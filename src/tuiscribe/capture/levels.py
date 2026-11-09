"""Cheap per-chunk level measurement for the meters.

Runs on the capture thread before any VAD/STT work. The accumulator collects
energy and peak since the last read, so a UI polling at ~20 fps sees the true
RMS and peak over its frame interval rather than a single sampled chunk.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

FLOOR_DB = -100.0


def to_db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-10 else FLOOR_DB


@dataclass(frozen=True)
class Level:
    rms_db: float
    peak_db: float
    peak: float  # linear, for clip and digital-silence checks
    samples: int


class LevelAccumulator:
    """Thread-safe: `add` is called from the capture thread, `take` from the UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sumsq = 0.0
        self._peak = 0.0
        self._n = 0

    def add(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        sumsq = float(np.dot(samples, samples))
        peak = float(np.max(np.abs(samples)))
        with self._lock:
            self._sumsq += sumsq
            self._n += samples.size
            if peak > self._peak:
                self._peak = peak

    def take(self) -> Level | None:
        """Return the level since the previous call and reset, or None if no audio arrived."""
        with self._lock:
            n, sumsq, peak = self._n, self._sumsq, self._peak
            self._sumsq = 0.0
            self._peak = 0.0
            self._n = 0
        if n == 0:
            return None
        return Level(rms_db=to_db(math.sqrt(sumsq / n)), peak_db=to_db(peak), peak=peak, samples=n)
