# !/usr/bin/env python3
"""Audio package exports."""
from __future__ import annotations
# mylib/__init__.py
from types import ModuleType

from lazy_loader import DelayedImportErrorModule

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import whisper
    import faster_whisper

import lazy_loader as lazy

whisper: ModuleType | DelayedImportErrorModule | None = lazy.load(fullname="whisper", error_on_import=True)  # whispers model is not loaded yet
faster_whisper: ModuleType | DelayedImportErrorModule | None = lazy.load(fullname="faster_whisper", error_on_import=True)  # faster-whisper model is not loaded yet
