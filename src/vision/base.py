"""Base interfaces and data structures for modular vision library."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np


class DetectionDict(TypedDict):
    """Standardized detection result dictionary format."""

    box: list[int]  # [x1, y1, x2, y2]
    score: float
    class_id: int
    label: str


@dataclass
class DetectedFace:
    """Standardized face detection output dataclass."""

    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 in pixels
    landmark5: np.ndarray | None = None  # (5,2) or None
    score: float = 1.0
    identity: str | None = None
    similarity: float | None = None
    embedding: np.ndarray | None = None  # Extracted embedding if available


@dataclass
class FaceEmbedding:
    """Registered face identity and its embedding representation."""

    id: str
    embedding: np.ndarray


@dataclass
class Detection:
    """Object detection bounding box result."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


class BaseDetector(ABC):
    """Abstract base class for all object and face detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect objects or faces in the input BGR frame.

        Args:
            frame: Input image/frame in BGR format.
            metadata: Optional camera/hardware metadata.

        Returns:
            List of standardized detection dictionaries.

        """

    def stop(self) -> None:
        """Release any hardware or runtime resources held by the detector."""
        # Default no-op hook for detectors without background threads/hardware.
        return
