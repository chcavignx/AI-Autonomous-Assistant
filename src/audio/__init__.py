# !/usr/bin/env python3
"""Audio package exports."""
from __future__ import annotations
# mylib/__init__.py
from types import ModuleType

from lazy_loader import DelayedImportErrorModule

import lazy_loader as lazy

whisper: ModuleType | DelayedImportErrorModule | None = lazy.load(fullname="whisper", error_on_import=False)  # pyright: ignore[reportUnknownMemberType]
faster_whisper: ModuleType | DelayedImportErrorModule | None = lazy.load(fullname="faster_whisper", error_on_import=False)  # pyright: ignore[reportUnknownMemberType]
