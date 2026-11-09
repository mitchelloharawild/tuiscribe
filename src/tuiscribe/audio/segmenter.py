"""Turn per-window VAD probabilities into utterances for one channel.

Pure state machine (no clock, no I/O): feed `(t, window, prob)` where `t` is
the window's start in session seconds, get back events.

- Speech starts when prob >= `threshold`; it ends after `min_silence_s` of
  prob < `neg_threshold`. Hysteresis avoids flapping.
- `pre_roll_s` of audio before the trigger is kept so first syllables survive.
- Utterances longer than `max_utterance_s` are cut at the quietest recent
  window and continue as a new utterance.
- While speaking, a `Partial` (the audio so far) is emitted every
  `partial_interval_s` for the live preview.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import SAMPLE_RATE


@dataclass
class Utterance:
    channel: str
    start: float  # session seconds (first speech window, excluding pre-roll)
    end: float  # session seconds (end of last speech window)
    audio: np.ndarray = field(repr=False)  # includes pre-roll and a short tail
    seq: int = 0  # per-channel sequence number, shared with its partials


@dataclass
class Partial:
    channel: str
    start: float
    audio: np.ndarray = field(repr=False)
    seq: int = 0


@dataclass
class SpeechStart:
    channel: str
    start: float
    seq: int = 0


@dataclass
class SpeechDiscarded:
    """Speech that ended too short to transcribe; lets the merger stop waiting."""

    channel: str
    start: float
    seq: int = 0


SegmenterEvent = Utterance | Partial | SpeechStart | SpeechDiscarded


class Segmenter:
    def __init__(
        self,
        channel: str,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        min_silence_s: float = 0.6,
        pre_roll_s: float = 0.3,
        tail_s: float = 0.2,
        min_speech_s: float = 0.3,
        max_utterance_s: float = 15.0,
        partial_interval_s: float = 2.0,
    ) -> None:
        self.channel = channel
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.min_silence_s = min_silence_s
        self.tail_s = tail_s
        self.min_speech_s = min_speech_s
        self.max_utterance_s = max_utterance_s
        self.partial_interval_s = partial_interval_s
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_s = pre_roll_s
        self._seq = 0
        self._reset_speech()

    # --- state ----------------------------------------------------------------

    def _reset_speech(self) -> None:
        self.in_speech = False
        self.speech_start: float | None = None
        self._audio: list[np.ndarray] = []
        self._probs: list[float] = []
        self._times: list[float] = []
        self._last_speech_end = 0.0
        self._silence_run = 0.0
        self._next_partial = 0.0

    @property
    def open_start(self) -> float | None:
        """Start of the utterance currently being collected, if any."""
        return self.speech_start if self.in_speech else None

    # --- feeding ----------------------------------------------------------------

    def push(self, t: float, window: np.ndarray, prob: float) -> list[SegmenterEvent]:
        dur = len(window) / SAMPLE_RATE
        events: list[SegmenterEvent] = []

        if not self.in_speech:
            if prob >= self.threshold:
                self._seq += 1
                self.in_speech = True
                self.speech_start = t
                self._audio = list(self._pre_roll)
                self._pre_roll.clear()
                self._probs = [0.0] * len(self._audio)
                self._times = [t - dur * (len(self._audio) - i) for i in range(len(self._audio))]
                self._last_speech_end = t + dur
                self._silence_run = 0.0
                self._next_partial = t + self.partial_interval_s
                events.append(SpeechStart(self.channel, t, self._seq))
            else:
                self._pre_roll.append(window)
                while len(self._pre_roll) * dur > self._pre_roll_s:
                    self._pre_roll.popleft()
                return events

        # in speech
        self._audio.append(window)
        self._probs.append(prob)
        self._times.append(t)
        if prob >= self.neg_threshold:
            self._last_speech_end = t + dur
            self._silence_run = 0.0
        else:
            self._silence_run += dur

        if self._silence_run >= self.min_silence_s:
            events.append(self._finish(trim_tail=True))
            return events

        assert self.speech_start is not None
        if t + dur - self.speech_start >= self.max_utterance_s:
            events.extend(self._split())
        elif t + dur >= self._next_partial:
            self._next_partial = t + dur + self.partial_interval_s
            events.append(Partial(self.channel, self.speech_start, np.concatenate(self._audio), self._seq))
        return events

    def flush(self) -> list[SegmenterEvent]:
        """End any open utterance (stop, pause, source switch)."""
        self._pre_roll.clear()
        if not self.in_speech:
            return []
        return [self._finish(trim_tail=True)]

    # --- internals ----------------------------------------------------------------

    def _finish(self, trim_tail: bool) -> Utterance | SpeechDiscarded:
        assert self.speech_start is not None
        start, end, seq = self.speech_start, self._last_speech_end, self._seq
        keep_until = end + self.tail_s if trim_tail else float("inf")
        audio = [a for a, ts in zip(self._audio, self._times) if ts < keep_until]
        self._reset_speech()
        if end - start < self.min_speech_s or not audio:
            return SpeechDiscarded(self.channel, start, seq)
        return Utterance(self.channel, start, end, np.concatenate(audio), seq)

    def _split(self) -> list[SegmenterEvent]:
        """Cut an over-long utterance at the quietest window in its last third."""
        n = len(self._probs)
        lo = max(1, n - max(n // 3, 1))
        cut = lo + int(np.argmin(self._probs[lo:]))
        head_audio, tail_audio = self._audio[:cut], self._audio[cut:]
        head_times, tail_times = self._times[:cut], self._times[cut:]
        tail_probs = self._probs[cut:]
        assert self.speech_start is not None
        cut_t = tail_times[0] if tail_times else self._last_speech_end
        head = Utterance(self.channel, self.speech_start, cut_t, np.concatenate(head_audio), self._seq)

        # continue speaking as a new utterance from the cut point
        self._seq += 1
        last_end = self._last_speech_end
        silence = self._silence_run
        self._audio, self._times, self._probs = list(tail_audio), list(tail_times), list(tail_probs)
        self.speech_start = cut_t
        self._last_speech_end = max(last_end, cut_t)
        self._silence_run = silence
        self._next_partial = cut_t + self.partial_interval_s
        return [head, SpeechStart(self.channel, cut_t, self._seq)]
