"""Base interfaces for modular vision library."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np


class DetectionDict(TypedDict):
    """Standardized detection result format."""

    box: list[int]  # [x1, y1, x2, y2]
    score: float
    class_id: int
    label: str


class BaseDetector(ABC):
    """Abstract base class for all object and face detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionDict]:
        """Detect objects or faces in the input BGR frame.

        Args:
            frame: Input image/frame in BGR format.

        Returns:
            List of standardized detection dictionaries.

        """
