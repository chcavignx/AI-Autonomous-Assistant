"""Metrics and performance profiling utilities for monitoring CPU/FPS."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

# Re-exported for backward compatibility

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ThroughputMeter:
    """Meter for measuring processing throughput (FPS)."""

    start_time: float = field(default_factory=perf_counter)
    count: int = 0

    def tick(self, n: int = 1) -> None:
        """Record the processing of n items."""
        self.count += n

    def fps(self) -> float:
        """Calculate the average throughput (items/sec) since start."""
        elapsed = perf_counter() - self.start_time
        return self.count / elapsed if elapsed > 0 else 0.0


@dataclass
class LatencyMeter:
    """Meter for tracking processing latencies and computing statistics."""

    samples: list[float] = field(default_factory=list)

    def record(self, start: float, end: float) -> None:
        """Record a latency sample in seconds.

        Args:
            start: Start timestamp.
            end: End timestamp.

        """
        self.samples.append(end - start)

    @property
    def mean_ms(self) -> float:
        """Compute the mean latency in milliseconds."""
        if not self.samples:
            return 0.0
        return 1000.0 * sum(self.samples) / len(self.samples)

    def percentile_ms(self, p: float) -> float:
        """Compute the specified percentile latency in milliseconds.

        Args:
            p: The percentile value (0 to 100).

        Returns:
            The percentile latency in milliseconds.

        """
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        k = int((p / 100.0) * (len(sorted_samples) - 1))
        return 1000.0 * sorted_samples[k]


class CsvLogger:
    """A logger for saving benchmark results in CSV format."""

    path: Path
    file: Any
    writer: Any
    start: float

    def __init__(self, path: Path, headers: list[str]) -> None:
        """Initialize the CSV logger and write the header row.

        Args:
            path: Destination file path.
            headers: Column names (excluding elapsed time_s).

        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time_s", *headers])
        self.start = perf_counter()

    def log(self, *values: float) -> None:
        """Log a new row of benchmark values.

        Args:
            values: Row values corresponding to headers.

        """
        t = perf_counter() - self.start
        self.writer.writerow([f"{t:.3f}"] + [f"{v:.3f}" for v in values])
        self.file.flush()

    def close(self) -> None:
        """Close the CSV file stream."""
        self.file.close()
