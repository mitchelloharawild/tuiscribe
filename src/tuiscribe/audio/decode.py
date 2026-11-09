"""Decode any audio/video file to 16 kHz mono float32 via ffmpeg (for `replay`)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE


def load_audio(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]  # fmt: skip
    try:
        out = subprocess.run(cmd, check=True, capture_output=True).stdout
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg is needed to read audio files") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg could not read {path}: {e.stderr.decode(errors='replace').strip()}") from e
    return np.frombuffer(out, dtype=np.float32).copy()
