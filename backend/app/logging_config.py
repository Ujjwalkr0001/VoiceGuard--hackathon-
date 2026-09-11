"""
VoiceGuard Backend — Structured Logging Configuration

Sets up structlog for structured JSON logging in production
and pretty-printed colored logs in development.

Usage:
    from app.logging_config import setup_logging
    setup_logging()  # Call once at app startup

    import structlog
    logger = structlog.get_logger(__name__)
    logger.info("event.name", key="value")
"""

import logging
import sys

import structlog


def setup_logging(*, json_logs: bool = False, log_level: str = "INFO") -> None:
    """
    Configure structlog + stdlib logging.

    Args:
        json_logs: If True, output JSON lines (production). If False, pretty-print (dev).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    # Shared processors for both dev and prod
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        # Production: JSON output
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: colorful, human-readable output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            # Format exceptions nicely
            structlog.processors.format_exc_info,
            # If using stdlib logging as the final output
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the stdlib root logger to use structlog's formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Quiet down noisy third-party loggers
    for noisy_logger in ("uvicorn.access", "urllib3", "botocore", "boto3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    structlog.get_logger().info(
        "logging.configured",
        json_logs=json_logs,
        log_level=log_level,
    )
