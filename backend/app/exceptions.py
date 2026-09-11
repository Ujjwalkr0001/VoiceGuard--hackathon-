"""
VoiceGuard Backend — Custom Exceptions

Organized by domain: audio processing, ML inference, external services, and API errors.
Each exception carries a machine-readable `error_code` for structured error responses.
"""


class VoiceGuardError(Exception):
    """Base exception for all VoiceGuard errors."""

    error_code: str = "VOICEGUARD_ERROR"
    status_code: int = 500

    def __init__(self, message: str = "An internal error occurred", **context):
        self.message = message
        self.context = context
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Audio Processing Errors
# ---------------------------------------------------------------------------
class AudioProcessingError(VoiceGuardError):
    """Base class for audio pipeline errors."""

    error_code = "AUDIO_ERROR"


class AudioDecodeError(AudioProcessingError):
    """Failed to decode mu-law or PCM audio from Twilio stream."""

    error_code = "AUDIO_DECODE_ERROR"

    def __init__(self, message: str = "Failed to decode audio payload", **context):
        super().__init__(message, **context)


class AudioBufferOverflow(AudioProcessingError):
    """Audio buffer exceeded maximum size — chunks not being consumed fast enough."""

    error_code = "AUDIO_BUFFER_OVERFLOW"

    def __init__(self, message: str = "Audio buffer overflow — processing too slow", **context):
        super().__init__(message, **context)


class FeatureExtractionError(AudioProcessingError):
    """DSP feature extraction failed (group delay, CQCC, pitch, VAD)."""

    error_code = "FEATURE_EXTRACTION_ERROR"

    def __init__(self, message: str = "Feature extraction failed", **context):
        super().__init__(message, **context)


# ---------------------------------------------------------------------------
# ML / Model Inference Errors
# ---------------------------------------------------------------------------
class ModelInferenceError(VoiceGuardError):
    """Base class for ML model errors."""

    error_code = "MODEL_ERROR"


class ModelNotLoadedError(ModelInferenceError):
    """Model was called before being loaded into memory."""

    error_code = "MODEL_NOT_LOADED"

    def __init__(self, model_name: str = "unknown", **context):
        super().__init__(f"Model '{model_name}' is not loaded", model_name=model_name, **context)


class ModelInferenceFailed(ModelInferenceError):
    """Model inference raised an unexpected error."""

    error_code = "MODEL_INFERENCE_FAILED"

    def __init__(self, model_name: str = "unknown", reason: str = "", **context):
        super().__init__(
            f"Inference failed for model '{model_name}': {reason}",
            model_name=model_name,
            **context,
        )


class EnsembleScoreError(ModelInferenceError):
    """Ensemble scoring (combining Model A + Model B) failed."""

    error_code = "ENSEMBLE_SCORE_ERROR"

    def __init__(self, message: str = "Ensemble scoring failed", **context):
        super().__init__(message, **context)


# ---------------------------------------------------------------------------
# External Service Errors
# ---------------------------------------------------------------------------
class ExternalServiceError(VoiceGuardError):
    """Base class for third-party service failures."""

    error_code = "EXTERNAL_SERVICE_ERROR"
    status_code = 502


class TwilioServiceError(ExternalServiceError):
    """Twilio API call failed."""

    error_code = "TWILIO_ERROR"

    def __init__(self, message: str = "Twilio service error", **context):
        super().__init__(message, **context)


class SarvamSTTError(ExternalServiceError):
    """Sarvam AI speech-to-text API failed or timed out."""

    error_code = "SARVAM_STT_ERROR"

    def __init__(self, message: str = "Sarvam AI STT failed", **context):
        super().__init__(message, **context)


class RedisConnectionError(ExternalServiceError):
    """Could not connect to or communicate with Redis."""

    error_code = "REDIS_ERROR"

    def __init__(self, message: str = "Redis connection failed", **context):
        super().__init__(message, **context)


class DynamoDBError(ExternalServiceError):
    """DynamoDB read/write operation failed."""

    error_code = "DYNAMODB_ERROR"

    def __init__(self, message: str = "DynamoDB operation failed", **context):
        super().__init__(message, **context)


class SNSPublishError(ExternalServiceError):
    """Failed to publish alert to AWS SNS."""

    error_code = "SNS_PUBLISH_ERROR"

    def __init__(self, message: str = "SNS publish failed", **context):
        super().__init__(message, **context)


class FCMDeliveryError(ExternalServiceError):
    """Failed to send push notification via Firebase Cloud Messaging."""

    error_code = "FCM_DELIVERY_ERROR"

    def __init__(self, message: str = "FCM push delivery failed", **context):
        super().__init__(message, **context)


# ---------------------------------------------------------------------------
# API / Request Errors
# ---------------------------------------------------------------------------
class APIError(VoiceGuardError):
    """Base class for API-level errors."""

    error_code = "API_ERROR"
    status_code = 400


class AuthenticationError(APIError):
    """Invalid or missing API key."""

    error_code = "AUTH_ERROR"
    status_code = 401

    def __init__(self, message: str = "Invalid or missing API key", **context):
        super().__init__(message, **context)


class RateLimitExceeded(APIError):
    """Client has exceeded the rate limit."""

    error_code = "RATE_LIMIT_EXCEEDED"
    status_code = 429

    def __init__(self, message: str = "Rate limit exceeded", **context):
        super().__init__(message, **context)


class SessionNotFoundError(APIError):
    """Requested call session does not exist."""

    error_code = "SESSION_NOT_FOUND"
    status_code = 404

    def __init__(self, call_sid: str = "", **context):
        super().__init__(f"Call session '{call_sid}' not found", call_sid=call_sid, **context)
