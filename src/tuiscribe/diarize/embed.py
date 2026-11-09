"""Speaker embeddings from a small ONNX model (no torch).

Default: 3D-Speaker CAM++ trained on VoxCeleb (English), 512-dim, ~28 MB,
~0.1 s of one CPU core per 3.5 s utterance. Exported to ONNX by sherpa-onnx.
Input is 80-bin Kaldi fbank (25 ms / 10 ms, povey window) with the
per-utterance mean removed.

The model is downloaded on first use into `~/.cache/tuiscribe/models`.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"  # sic


@dataclass(frozen=True)
class EmbeddingModel:
    file: str
    window: str  # Kaldi window type used in training
    scale: float  # multiply [-1, 1] samples by this (wespeaker trained on int16 range)


MODELS = {
    "campplus": EmbeddingModel("3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx", "povey", 1.0),
    "wespeaker-resnet34": EmbeddingModel("wespeaker_en_voxceleb_resnet34_LM.onnx", "hamming", 32768.0),
}
DEFAULT_MODEL = "campplus"
MIN_AUDIO_S = 0.3  # shorter than this gives no embedding at all


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "tuiscribe" / "models"


def ensure_model(name: str = DEFAULT_MODEL) -> Path:
    m = MODELS[name]
    path = cache_dir() / m.file
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(f"{RELEASES}/{m.file}", timeout=60) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


class SpeakerEmbedder:
    def __init__(self, name: str = DEFAULT_MODEL, threads: int = 1) -> None:
        self.name = name
        self.model = MODELS[name]
        self.threads = threads
        self._session = None
        self._input = ""

    def load(self) -> None:
        """Download (first time) and load the model. Slow; call off the UI thread."""
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(ensure_model(self.name)), opts, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """L2-normalised embedding of 16 kHz mono float32 audio, or None if too short."""
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            return None
        if self._session is None:
            self.load()
        feats = self._fbank(audio)
        feats -= feats.mean(axis=0)
        emb = self._session.run(None, {self._input: feats[None]})[0][0]
        norm = float(np.linalg.norm(emb))
        return (emb / norm).astype(np.float32) if norm > 0 else None

    def _fbank(self, audio: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.window_type = self.model.window
        opts.mel_opts.num_bins = 80
        fb = knf.OnlineFbank(opts)
        fb.accept_waveform(SAMPLE_RATE, (np.asarray(audio, dtype=np.float32) * self.model.scale).tolist())
        fb.input_finished()
        return np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)
