"""
VoiceGuard Backend — Twilio Media Stream WebSocket Endpoint

Receives real-time audio from Twilio Media Streams over WebSocket.
Implements the Twilio Media Streams protocol:
  - 'connected': WebSocket connection established
  - 'start': Stream metadata (streamSid, callSid, track, etc.)
  - 'media': Base64-encoded mu-law audio payload
  - 'stop': Stream ended (call hung up)

Decodes mu-law 8kHz audio to 16-bit PCM and feeds it into the
AudioBufferManager for chunking.
"""

import asyncio
import base64
import json
from typing import Optional

import audioop
import numpy as np
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


class AudioBufferManager:
    """
    Manages incoming PCM audio for a single call.

    Accumulates decoded PCM frames into a rolling buffer and emits
    fixed-size chunks (default 2 seconds) with configurable overlap
    (default 0.5 seconds) to an asyncio queue for downstream processing.
    """

    def __init__(
        self,
        call_sid: str,
        sample_rate: int = 8000,
        chunk_duration_sec: float = 2.0,
        overlap_sec: float = 0.5,
    ):
        self.call_sid = call_sid
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec

        # Calculate sizes in samples (16-bit = 2 bytes per sample)
        self.chunk_samples = int(chunk_duration_sec * sample_rate)
        self.overlap_samples = int(overlap_sec * sample_rate)
        self.hop_samples = self.chunk_samples - self.overlap_samples

        # Internal buffer (raw bytes, 16-bit PCM = 2 bytes per sample)
        self._buffer = bytearray()
        self._total_samples_received: int = 0
        self._chunks_emitted: int = 0

        # Output queue for processed chunks
        self.chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=50)

        logger.info(
            "audio_buffer.initialized",
            call_sid=call_sid,
            chunk_samples=self.chunk_samples,
            overlap_samples=self.overlap_samples,
            hop_samples=self.hop_samples,
        )

    def add_audio(self, pcm_bytes: bytes, timestamp_ms: int) -> None:
        """
        Append decoded PCM bytes to the buffer.
        If enough data has accumulated, emit a chunk to the queue.

        Args:
            pcm_bytes: Raw 16-bit PCM audio bytes
            timestamp_ms: Twilio media timestamp in milliseconds
        """
        self._buffer.extend(pcm_bytes)
        self._total_samples_received += len(pcm_bytes) // 2  # 2 bytes per sample

        # Check if we have enough for a full chunk
        bytes_per_chunk = self.chunk_samples * 2  # 16-bit = 2 bytes/sample
        bytes_per_hop = self.hop_samples * 2

        while len(self._buffer) >= bytes_per_chunk:
            # Extract the chunk
            chunk_bytes = bytes(self._buffer[:bytes_per_chunk])

            # Convert to numpy float32 array normalized to [-1, 1]
            chunk_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Calculate the timestamp for this chunk
            chunk_start_sample = self._chunks_emitted * self.hop_samples
            chunk_timestamp_sec = chunk_start_sample / self.sample_rate

            # Emit chunk (non-blocking put)
            chunk_data = {
                "audio": chunk_array,
                "sample_rate": self.sample_rate,
                "call_sid": self.call_sid,
                "chunk_index": self._chunks_emitted,
                "timestamp_sec": chunk_timestamp_sec,
                "duration_sec": self.chunk_duration_sec,
            }

            try:
                self.chunk_queue.put_nowait(chunk_data)
                self._chunks_emitted += 1
                logger.debug(
                    "audio_buffer.chunk_emitted",
                    call_sid=self.call_sid,
                    chunk_index=chunk_data["chunk_index"],
                    timestamp_sec=round(chunk_timestamp_sec, 2),
                    buffer_remaining=len(self._buffer) // 2,
                )
            except asyncio.QueueFull:
                logger.warning(
                    "audio_buffer.queue_full",
                    call_sid=self.call_sid,
                    chunks_emitted=self._chunks_emitted,
                )

            # Advance buffer by hop size (keep overlap portion)
            self._buffer = self._buffer[bytes_per_hop:]

    @property
    def stats(self) -> dict:
        """Return buffer statistics."""
        return {
            "total_samples_received": self._total_samples_received,
            "total_duration_sec": round(self._total_samples_received / self.sample_rate, 2),
            "chunks_emitted": self._chunks_emitted,
            "buffer_samples": len(self._buffer) // 2,
        }


# ---------------------------------------------------------------------------
# Active call sessions
# ---------------------------------------------------------------------------
_active_sessions: dict[str, AudioBufferManager] = {}


@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio Media Streams WebSocket endpoint.

    Protocol events:
      - connected: WebSocket is open
      - start:     Stream metadata (streamSid, callSid, track, customParameters)
      - media:     Audio payload (base64 mu-law, 8kHz, mono)
      - stop:      Stream ended
    """
    await websocket.accept()

    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    buffer_manager: Optional[AudioBufferManager] = None

    logger.info("media_stream.websocket_connected")

    try:
        async for raw_message in websocket.iter_text():
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning("media_stream.invalid_json")
                continue

            event_type = message.get("event")

            # ----- CONNECTED -----
            if event_type == "connected":
                logger.info(
                    "media_stream.connected",
                    protocol=message.get("protocol"),
                    version=message.get("version"),
                )

            # ----- START -----
            elif event_type == "start":
                start_data = message.get("start", {})
                stream_sid = message.get("streamSid")
                call_sid = start_data.get("callSid", "unknown")
                track = start_data.get("track", "unknown")
                custom_params = start_data.get("customParameters", {})

                logger.info(
                    "media_stream.start",
                    stream_sid=stream_sid,
                    call_sid=call_sid,
                    track=track,
                    caller_number=custom_params.get("caller_number", ""),
                    called_number=custom_params.get("called_number", ""),
                    media_format=start_data.get("mediaFormat", {}),
                )

                # Create buffer manager for this call
                buffer_manager = AudioBufferManager(
                    call_sid=call_sid,
                    sample_rate=8000,
                    chunk_duration_sec=settings.audio_chunk_duration_sec,
                    overlap_sec=settings.audio_chunk_overlap_sec,
                )
                _active_sessions[call_sid] = buffer_manager

            # ----- MEDIA -----
            elif event_type == "media":
                if buffer_manager is None:
                    continue

                media_data = message.get("media", {})
                payload_b64 = media_data.get("payload", "")
                timestamp_ms = int(media_data.get("timestamp", "0"))

                if not payload_b64:
                    continue

                # Decode: base64 → mu-law bytes → 16-bit PCM
                mulaw_bytes = base64.b64decode(payload_b64)
                pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)  # 2 = 16-bit width

                # Feed into buffer manager
                buffer_manager.add_audio(pcm_bytes, timestamp_ms)

            # ----- STOP -----
            elif event_type == "stop":
                logger.info(
                    "media_stream.stop",
                    stream_sid=stream_sid,
                    call_sid=call_sid,
                    stats=buffer_manager.stats if buffer_manager else None,
                )

                # Clean up session
                if call_sid and call_sid in _active_sessions:
                    del _active_sessions[call_sid]

    except WebSocketDisconnect:
        logger.info(
            "media_stream.disconnected",
            stream_sid=stream_sid,
            call_sid=call_sid,
            stats=buffer_manager.stats if buffer_manager else None,
        )
    except Exception as e:
        logger.error(
            "media_stream.error",
            stream_sid=stream_sid,
            call_sid=call_sid,
            error=str(e),
            error_type=type(e).__name__,
        )
    finally:
        # Always clean up
        if call_sid and call_sid in _active_sessions:
            del _active_sessions[call_sid]
