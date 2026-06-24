"""Logging filters for the voice agent."""

import logging


class MaxLevelFilter(logging.Filter):
    """Filter to limit logging to a maximum level (exclusive upper bound)."""

    def __init__(self, max_level: str) -> None:
        """Initialize the filter with an inclusive maximum log level."""
        self.max_level: int = getattr(logging, max_level)
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:  # pyright: ignore[reportImplicitOverride]
        """Return True only if the record's level is below max_level."""
        return record.levelno <= self.max_level
