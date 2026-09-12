"""
VoiceGuard Backend — Configuration

All environment variables are loaded and validated here using Pydantic BaseSettings.
Import `settings` from this module anywhere in the app.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App / Server ----
    app_env: str = Field(default="development", description="Environment: development | staging | production")
    app_debug: bool = Field(default=True, description="Enable debug mode")
    app_host: str = Field(default="0.0.0.0", description="Server bind host")
    app_port: int = Field(default=8000, description="Server bind port")
    api_key: str = Field(default="dev-api-key", description="API key for authenticating REST endpoints")

    # ---- Twilio ----
    twilio_account_sid: str = Field(default="", description="Twilio Account SID")
    twilio_auth_token: str = Field(default="", description="Twilio Auth Token")
    twilio_phone_number: str = Field(default="", description="Twilio phone number (E.164 format)")

    # ---- Sarvam AI (STT) ----
    sarvam_api_key: str = Field(default="", description="Sarvam AI API key for speech-to-text")

    # ---- AWS ----
    aws_access_key_id: str = Field(default="", description="AWS Access Key ID")
    aws_secret_access_key: str = Field(default="", description="AWS Secret Access Key")
    aws_region: str = Field(default="ap-south-1", description="AWS region")

    # ---- Redis ----
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # ---- DynamoDB ----
    dynamodb_table_name: str = Field(default="call_sessions", description="DynamoDB table name")
    dynamodb_endpoint_url: Optional[str] = Field(
        default=None,
        description="Custom DynamoDB endpoint (e.g. http://localhost:8001 for DynamoDB Local). "
        "Leave unset to use real AWS.",
    )

    # ---- S3 (optional debug dumps) ----
    s3_bucket_name: str = Field(default="voiceguard-debug", description="S3 bucket for debug dumps")
    s3_enabled: bool = Field(default=False, description="Enable S3 audio/feature dumps")

    # ---- SNS ----
    sns_topic_arn: str = Field(default="", description="SNS topic ARN for alert fan-out")

    # ---- Firebase Cloud Messaging ----
    fcm_server_key: str = Field(default="", description="FCM server key for push notifications")

    # ---- Risk Engine Thresholds ----
    risk_threshold_medium: int = Field(default=70, ge=0, le=100, description="Medium risk threshold (0-100)")
    risk_threshold_high: int = Field(default=85, ge=0, le=100, description="High risk threshold (0-100)")

    # ---- Ensemble Weights ----
    ensemble_weight_model_a: float = Field(default=0.6, ge=0.0, le=1.0, description="Weight for deep model (AASIST)")
    ensemble_weight_model_b: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for XGBoost model")

    # ---- Caller Multipliers ----
    caller_multiplier_unknown: float = Field(default=1.3, ge=1.0, description="Risk multiplier for unknown callers")
    caller_multiplier_flagged: float = Field(default=1.5, ge=1.0, description="Risk multiplier for flagged callers")

    # ---- Pipeline ----
    rolling_window_size: int = Field(default=10, ge=1, description="Number of chunks for rolling average")
    audio_chunk_duration_sec: float = Field(default=2.0, gt=0.0, description="Audio chunk duration in seconds")
    audio_chunk_overlap_sec: float = Field(default=0.5, ge=0.0, description="Audio chunk overlap in seconds")

    # ---- Validators ----
    @field_validator("risk_threshold_high")
    @classmethod
    def high_threshold_above_medium(cls, v: int, info) -> int:
        """Ensure high threshold is >= medium threshold."""
        medium = info.data.get("risk_threshold_medium", 70)
        if v < medium:
            raise ValueError(
                f"risk_threshold_high ({v}) must be >= risk_threshold_medium ({medium})"
            )
        return v

    @field_validator("ensemble_weight_model_b")
    @classmethod
    def ensemble_weights_sum_to_one(cls, v: float, info) -> float:
        """Warn if ensemble weights don't approximately sum to 1.0."""
        weight_a = info.data.get("ensemble_weight_model_a", 0.6)
        total = weight_a + v
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Ensemble weights should sum to ~1.0, got {weight_a} + {v} = {total}"
            )
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"

    @property
    def audio_chunk_samples(self) -> int:
        """Number of PCM samples per chunk at 8 kHz (Twilio's sample rate)."""
        return int(self.audio_chunk_duration_sec * 8000)

    @property
    def audio_overlap_samples(self) -> int:
        """Number of overlapping PCM samples per chunk at 8 kHz."""
        return int(self.audio_chunk_overlap_sec * 8000)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache so the .env file is only read once.
    Call get_settings.cache_clear() if you need to reload.
    """
    return Settings()


# Convenience alias — import this directly
settings = get_settings()
