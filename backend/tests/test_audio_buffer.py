"""
Unit tests for AudioBufferManager.

Verifies correct chunk size, overlap behavior, timestamp alignment,
and edge cases (silence, queue full, small inputs).
"""

import asyncio
import numpy as np
import pytest

from app.routes.media_stream import AudioBufferManager


class TestAudioBufferManager:
    """Tests for the AudioBufferManager class."""

    def _make_pcm_bytes(self, num_samples: int, value: int = 1000) -> bytes:
        """Generate fake 16-bit PCM bytes (constant value)."""
        samples = np.full(num_samples, value, dtype=np.int16)
        return samples.tobytes()

    def _make_sine_pcm_bytes(self, num_samples: int, freq: float = 440.0, sr: int = 8000) -> bytes:
        """Generate a sine wave as 16-bit PCM bytes."""
        t = np.arange(num_samples) / sr
        samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
        return samples.tobytes()

    # ---- Chunk size ----

    def test_no_chunk_emitted_when_buffer_too_small(self):
        """Buffer with less than 2 seconds of audio should not emit a chunk."""
        mgr = AudioBufferManager(call_sid="test-1", sample_rate=8000)
        # Feed 1 second of audio (8000 samples) — not enough for a 2-sec chunk
        pcm = self._make_pcm_bytes(8000)
        mgr.add_audio(pcm, timestamp_ms=0)

        assert mgr.chunk_queue.empty()
        assert mgr.stats["chunks_emitted"] == 0
        assert mgr.stats["total_samples_received"] == 8000

    def test_chunk_emitted_at_exactly_2_seconds(self):
        """Exactly 2 seconds (16000 samples at 8kHz) should emit one chunk."""
        mgr = AudioBufferManager(call_sid="test-2", sample_rate=8000)
        pcm = self._make_pcm_bytes(16000)
        mgr.add_audio(pcm, timestamp_ms=0)

        assert not mgr.chunk_queue.empty()
        assert mgr.stats["chunks_emitted"] == 1

    def test_chunk_has_correct_sample_count(self):
        """Emitted chunk should contain exactly chunk_samples float32 values."""
        mgr = AudioBufferManager(call_sid="test-3", sample_rate=8000)
        pcm = self._make_pcm_bytes(16000)
        mgr.add_audio(pcm, timestamp_ms=0)

        chunk = mgr.chunk_queue.get_nowait()
        assert chunk["audio"].shape == (16000,)
        assert chunk["audio"].dtype == np.float32

    def test_chunk_audio_is_normalized(self):
        """Audio values should be normalized to [-1, 1] range."""
        mgr = AudioBufferManager(call_sid="test-4", sample_rate=8000)
        pcm = self._make_sine_pcm_bytes(16000)
        mgr.add_audio(pcm, timestamp_ms=0)

        chunk = mgr.chunk_queue.get_nowait()
        assert chunk["audio"].max() <= 1.0
        assert chunk["audio"].min() >= -1.0

    # ---- Overlap ----

    def test_overlap_produces_correct_number_of_chunks(self):
        """
        With 2s chunks and 0.5s overlap (1.5s hop):
        - 2.0s audio → 1 chunk
        - 3.5s audio → 2 chunks (at 0s and 1.5s)
        - 5.0s audio → 3 chunks (at 0s, 1.5s, 3.0s)
        """
        mgr = AudioBufferManager(call_sid="test-5", sample_rate=8000)

        # Feed 5 seconds = 40000 samples
        pcm = self._make_pcm_bytes(40000)
        mgr.add_audio(pcm, timestamp_ms=0)

        assert mgr.stats["chunks_emitted"] == 3

    def test_overlap_data_is_shared(self):
        """
        The last 0.5s of chunk N should equal the first 0.5s of chunk N+1.
        """
        mgr = AudioBufferManager(call_sid="test-6", sample_rate=8000)

        # Feed exactly 3.5 seconds (28000 samples) → 2 chunks
        pcm = self._make_sine_pcm_bytes(28000, freq=440.0)
        mgr.add_audio(pcm, timestamp_ms=0)

        assert mgr.stats["chunks_emitted"] == 2

        chunk1 = mgr.chunk_queue.get_nowait()
        chunk2 = mgr.chunk_queue.get_nowait()

        # Last 4000 samples of chunk1 == first 4000 samples of chunk2
        overlap_from_chunk1 = chunk1["audio"][-4000:]
        overlap_from_chunk2 = chunk2["audio"][:4000]
        np.testing.assert_array_almost_equal(overlap_from_chunk1, overlap_from_chunk2)

    # ---- Timestamps ----

    def test_first_chunk_timestamp_is_zero(self):
        """First chunk should have timestamp_sec = 0.0."""
        mgr = AudioBufferManager(call_sid="test-7", sample_rate=8000)
        pcm = self._make_pcm_bytes(16000)
        mgr.add_audio(pcm, timestamp_ms=0)

        chunk = mgr.chunk_queue.get_nowait()
        assert chunk["timestamp_sec"] == 0.0
        assert chunk["chunk_index"] == 0

    def test_second_chunk_timestamp_is_hop_aligned(self):
        """Second chunk timestamp should be at 1.5 seconds (hop = 2.0 - 0.5)."""
        mgr = AudioBufferManager(call_sid="test-8", sample_rate=8000)
        pcm = self._make_pcm_bytes(28000)  # 3.5 seconds → 2 chunks
        mgr.add_audio(pcm, timestamp_ms=0)

        chunk1 = mgr.chunk_queue.get_nowait()
        chunk2 = mgr.chunk_queue.get_nowait()

        assert chunk1["timestamp_sec"] == 0.0
        assert chunk2["timestamp_sec"] == 1.5
        assert chunk2["chunk_index"] == 1

    def test_chunk_metadata_fields(self):
        """Each chunk should carry all required metadata."""
        mgr = AudioBufferManager(call_sid="test-9", sample_rate=8000)
        pcm = self._make_pcm_bytes(16000)
        mgr.add_audio(pcm, timestamp_ms=0)

        chunk = mgr.chunk_queue.get_nowait()
        assert "audio" in chunk
        assert "sample_rate" in chunk
        assert "call_sid" in chunk
        assert "chunk_index" in chunk
        assert "timestamp_sec" in chunk
        assert "duration_sec" in chunk

        assert chunk["sample_rate"] == 8000
        assert chunk["call_sid"] == "test-9"
        assert chunk["duration_sec"] == 2.0

    # ---- Incremental feeding ----

    def test_incremental_small_feeds(self):
        """Feeding audio in small increments should still produce correct chunks."""
        mgr = AudioBufferManager(call_sid="test-10", sample_rate=8000)

        # Feed 160 samples at a time (20ms frames, typical Twilio packet)
        for i in range(100):  # 100 × 160 = 16000 samples = 2 seconds
            pcm = self._make_pcm_bytes(160)
            mgr.add_audio(pcm, timestamp_ms=i * 20)

        assert mgr.stats["chunks_emitted"] == 1
        chunk = mgr.chunk_queue.get_nowait()
        assert chunk["audio"].shape == (16000,)

    # ---- Edge cases ----

    def test_empty_audio_does_nothing(self):
        """Feeding empty bytes should not crash or emit."""
        mgr = AudioBufferManager(call_sid="test-11", sample_rate=8000)
        mgr.add_audio(b"", timestamp_ms=0)

        assert mgr.chunk_queue.empty()
        assert mgr.stats["total_samples_received"] == 0

    def test_stats_are_accurate(self):
        """Stats should reflect actual buffer state."""
        mgr = AudioBufferManager(call_sid="test-12", sample_rate=8000)
        pcm = self._make_pcm_bytes(20000)  # 2.5 seconds
        mgr.add_audio(pcm, timestamp_ms=0)

        stats = mgr.stats
        assert stats["total_samples_received"] == 20000
        assert stats["total_duration_sec"] == 2.5
        assert stats["chunks_emitted"] == 1
        # After emitting 1 chunk, buffer should have: 20000 - 12000 (hop) = 8000 samples
        assert stats["buffer_samples"] == 8000

    def test_custom_chunk_duration(self):
        """Custom chunk duration should work correctly."""
        mgr = AudioBufferManager(
            call_sid="test-13",
            sample_rate=8000,
            chunk_duration_sec=1.0,
            overlap_sec=0.25,
        )
        # 1 second chunk, 0.25s overlap, 0.75s hop
        pcm = self._make_pcm_bytes(8000)  # 1 second
        mgr.add_audio(pcm, timestamp_ms=0)

        assert mgr.stats["chunks_emitted"] == 1
        chunk = mgr.chunk_queue.get_nowait()
        assert chunk["audio"].shape == (8000,)
