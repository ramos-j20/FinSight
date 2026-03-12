"""Structured logging configuration using structlog."""
import logging
import sys

import structlog

from backend.core.config import get_settings


def setup_logging() -> None:
    """Configure structured logging using structlog."""
    settings = get_settings()
    
    # Map string log level to logging module integer level
    log_level_name = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Standard library logging configuration
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Determine execution environment based on log level (simplified for v1)
    is_development = log_level_name == "DEBUG"

    # Structlog configuration processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if is_development:
        # Pretty console output for development
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance for the given module name.
    
    Args:
        name: Name of the module/logger, usually __name__
        
    Returns:
        A structlog bound logger configured for the application.
    """
    # Ensure logging is setup when a logger is requested
    if not structlog.is_configured():
        setup_logging()
    
    return structlog.get_logger(name)
