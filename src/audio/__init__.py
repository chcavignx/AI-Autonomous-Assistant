# audio/__init__.py
"""Audio package exports."""
from __future__ import annotations

from types import ModuleType

from lazy_loader import DelayedImportErrorModule

import lazy_loader as lazy

whisper: ModuleType | DelayedImportErrorModule | None = lazy.load(fullname="whisper", error_on_import=False)  # pyright: ignore[reportUnknownMemberType]
faster_whisper: ModuleType | DelayedImportErrorModule | None = lazy.load(fullname="faster_whisper", error_on_import=False)  # pyright: ignore[reportUnknownMemberType]

# Module-level attributes for backward compatibility
__all__ = ["whisper", "faster_whisper"]
