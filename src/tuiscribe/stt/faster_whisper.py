"""faster-whisper (CTranslate2) backend, int8 on CPU by default.

Audio arrives already VAD-gated, so the model's own VAD is off. Whisper is
prone to inventing text on near-silence; segments are dropped using the usual
Whisper heuristics (no-speech probability + low log-prob, repetition via
compression ratio) and a short list of stock hallucinations.
"""

from __future__ import annotations

import os
import re

import numpy as np

from . import Transcript

# Stock phrases Whisper produces from silence/noise. Dropped only when the
# model itself is unsure (see _is_hallucination), so a real "thank you" survives.
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
    "thank you for watching.", "thank you for watching!", "you", "bye.", ".",
    "please subscribe.", "subtitles by the amara.org community",
}  # fmt: skip


def default_threads() -> int:
    # Half the cores, at most 4, so capture and the UI keep up;
    # override with TUISCRIBE_STT_THREADS.
    env = os.environ.get("TUISCRIBE_STT_THREADS")
    if env and env.isdigit():
        return int(env)
    return max(1, min(4, (os.cpu_count() or 4) // 2))


class FasterWhisperTranscriber:
    def __init__(
        self,
        model: str = "small.en",
        compute_type: str = "int8",
        cpu_threads: int | None = None,
        beam_size: int = 5,
        language: str | None = None,
    ) -> None:
        self.model_name = model
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads or default_threads()
        self.beam_size = beam_size
        self.language = language or ("en" if model.endswith(".en") else None)
        self.name = f"faster-whisper {model}"
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    def transcribe(self, audio: np.ndarray, prompt: str | None = None, final: bool = True) -> Transcript:
        if self._model is None:
            self.load()
        assert self._model is not None
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size if final else 1,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            without_timestamps=False,  # timestamp mode is more robust on long chunks
        )
        kept, nsp, lp = [], [], []
        for s in segments:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            # High compression ratio alone is not dropped: faster-whisper already
            # retries at higher temperature, and legitimately repetitive speech
            # (paths, lists) scores high. Genuine loops are collapsed instead.
            text = collapse_repeats(s.text.strip())
            if not text or _is_hallucination(text, s.no_speech_prob, s.avg_logprob):
                continue
            kept.append(text)
            nsp.append(s.no_speech_prob)
            lp.append(s.avg_logprob)
        text = re.sub(r"\s+", " ", " ".join(kept)).strip()
        return Transcript(
            text=text,
            no_speech_prob=float(np.mean(nsp)) if nsp else None,
            avg_logprob=float(np.mean(lp)) if lp else None,
        )


_REPEAT = re.compile(r"\b((?:\S+\s+){0,5}?\S+?)([\s,.]+\1\b){3,}", re.IGNORECASE)


def collapse_repeats(text: str) -> str:
    """Collapse a phrase of 1-6 words repeated 4+ times in a row (Whisper loop) to one copy."""
    return _REPEAT.sub(r"\1", text)


def _is_hallucination(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
    return text.lower() in HALLUCINATIONS and (no_speech_prob > 0.3 or avg_logprob < -0.7)
