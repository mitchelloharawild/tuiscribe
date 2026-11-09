"""Speech-to-text backends behind one interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Transcript:
    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


class Transcriber(Protocol):
    name: str  # e.g. "faster-whisper small.en"

    def load(self) -> None:
        """Load (and download if needed) the model. Slow; call off the UI thread."""

    def transcribe(self, audio: np.ndarray, prompt: str | None = None, final: bool = True) -> Transcript:
        """Transcribe 16 kHz mono float32 audio. `final=False` may trade accuracy for speed."""
