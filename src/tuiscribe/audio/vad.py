"""Streaming Silero VAD (v6 ONNX, bundled with faster-whisper).

The model scores 512-sample windows (32 ms at 16 kHz). Each input row is the
previous 64 samples of context followed by the 512 new samples, and the LSTM
state (h, c) carries across calls, so audio can be fed in arbitrary chunk sizes.
One instance per channel: the state is per stream.
"""

from __future__ import annotations

import os

import numpy as np

WINDOW = 512
CONTEXT = 64


def _model_path() -> str:
    from faster_whisper.utils import get_assets_path

    return os.path.join(get_assets_path(), "silero_vad_v6.onnx")


class StreamingVAD:
    def __init__(self) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            _model_path(), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Feed audio; return (window_samples, speech_prob) for each complete window."""
        buf = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n = len(buf) // WINDOW
        self._pending = buf[n * WINDOW :]
        if n == 0:
            return []
        windows = buf[: n * WINDOW].reshape(n, WINDOW)
        ctx = np.empty((n, CONTEXT), dtype=np.float32)
        ctx[0] = self._context
        if n > 1:
            ctx[1:] = windows[:-1, -CONTEXT:]
        rows = np.concatenate([ctx, windows], axis=1)
        probs, self._h, self._c = self._session.run(
            None, {"input": rows, "h": self._h, "c": self._c}
        )
        self._context = windows[-1, -CONTEXT:].copy()
        probs = np.asarray(probs).reshape(-1)
        return [(windows[i], float(probs[i])) for i in range(n)]
