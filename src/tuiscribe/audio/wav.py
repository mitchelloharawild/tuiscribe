"""Time-aligned WAV writer for the saved `.mic.wav` and `.system.wav` files.

Chunks are placed by their capture time relative to the session start, so a
gap (source switch, device dropout) is filled with silence and the mic and
system WAVs stay aligned with each other for `replay`.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE

# Only pad when the clock says we are this far behind; smaller differences are
# clock jitter between the monotonic clock and the sample counter.
PAD_THRESHOLD = SAMPLE_RATE // 10  # 100 ms


class AlignedWavWriter:
    def __init__(self, path: Path, session_start: float) -> None:
        self.path = path
        self.session_start = session_start
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)
        self._written = 0
        self._lock = threading.Lock()

    @property
    def seconds(self) -> float:
        return self._written / SAMPLE_RATE

    def write(self, t0: float, samples: np.ndarray) -> None:
        expected = int(round((t0 - self.session_start) * SAMPLE_RATE))
        with self._lock:
            if self._wav is None:
                return
            if expected - self._written > PAD_THRESHOLD:
                pad = expected - self._written
                self._wav.writeframes(np.zeros(pad, dtype=np.int16).tobytes())
                self._written += pad
            pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
            self._wav.writeframes(pcm.tobytes())
            self._written += len(pcm)

    def close(self) -> None:
        with self._lock:
            if self._wav is not None:
                self._wav.close()
                self._wav = None


class WavSpans:
    """Reads stretches of saved WAVs by session time without loading whole files.

    Only the format `AlignedWavWriter` writes (16 kHz mono 16-bit) is read; other
    files give None, as does a channel with no file. Call `close()` when done.
    """

    def __init__(self, paths: dict[str, Path]) -> None:
        self._wavs: dict[str, wave.Wave_read] = {}
        for ch, path in paths.items():
            try:
                w = wave.open(str(path), "rb")
            except (OSError, EOFError, wave.Error):
                continue
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SAMPLE_RATE):
                self._wavs[ch] = w
            else:
                w.close()

    @property
    def channels(self) -> set[str]:
        return set(self._wavs)

    def __call__(self, channel: str, start: float, end: float) -> np.ndarray | None:
        w = self._wavs.get(channel)
        if w is None:
            return None
        a = max(0, int(round(start * SAMPLE_RATE)))
        b = min(w.getnframes(), int(round(end * SAMPLE_RATE)))
        if b <= a:
            return None
        w.setpos(a)
        return np.frombuffer(w.readframes(b - a), dtype=np.int16).astype(np.float32) / 32768.0

    def close(self) -> None:
        for w in self._wavs.values():
            w.close()
        self._wavs.clear()
