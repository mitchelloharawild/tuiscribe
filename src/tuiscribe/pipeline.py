"""VAD -> segmenter -> STT -> merger -> writer.

Live mode (`Pipeline`) runs two worker threads:

- **VAD thread**: consumes capture chunks for all channels, runs Silero VAD and
  the segmenter, queues finished utterances (finals) and keeps only the newest
  partial request per channel.
- **STT thread**: loads the model, then transcribes. Finals always go first;
  partial passes run only when no final is waiting.
  Finished items go through the merger, which releases them in start order
  once no earlier speech is still in flight, and are appended to the file.

Offline mode (`transcribe_offline`) runs the same stages synchronously for
`replay`.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .audio.segmenter import Partial, Segmenter, SpeechDiscarded, SpeechStart, Utterance
from .audio.vad import WINDOW, StreamingVAD
from .config import SAMPLE_RATE
from .events import EventBus, FinalEvent, PartialEvent, StatusEvent
from .merge import Merger
from .output import TranscriptWriter
from .records import Item, Note, Segment
from .stt import Transcriber

PROMPT_CHARS = 200  # previous text of the same channel given to Whisper as context
DISCONTINUITY_S = 0.25  # capture timeline jump that resets VAD state (switch/dropout)


@dataclass(frozen=True)
class PipelineStats:
    model_state: str  # "loading", "ready", "error"
    queue: int  # finals waiting for STT
    latency: float | None  # end of speech -> text written, last final (s)
    segments: int
    dropped_chunks: int
    busy: bool  # STT currently running


class Pipeline:
    def __init__(
        self,
        channels: list[str],
        transcriber: Transcriber,
        writer: TranscriptWriter | None,
        bus: EventBus,
        clock: Callable[[], float],  # current session time in seconds
        partials: bool = True,
        max_queue_chunks: int = 3000,  # ~60 s of 20 ms chunks
    ) -> None:
        self.channels = channels
        self.transcriber = transcriber
        self.writer = writer
        self.bus = bus
        self.clock = clock
        self.partials = partials

        self._chunks: queue.Queue = queue.Queue(maxsize=max_queue_chunks)
        self._finals: queue.Queue[Utterance | None] = queue.Queue()
        self._partials: dict[str, Partial] = {}
        self._partials_lock = threading.Lock()
        self._inflight: dict[tuple[str, int], float] = {}  # (channel, seq) -> start
        self._inflight_lock = threading.Lock()
        self._merger = Merger()
        self._merger_lock = threading.Lock()
        self._last_text: dict[str, str] = {ch: "" for ch in channels}

        self._vads = {ch: StreamingVAD() for ch in channels}
        self._segs = {ch: Segmenter(ch) for ch in channels}
        self._buf_t: dict[str, float | None] = {ch: None for ch in channels}

        self.model_state = "loading"
        self.latency: float | None = None
        self.dropped_chunks = 0
        self._busy = False
        self._vad_thread = threading.Thread(target=self._vad_loop, name="vad", daemon=True)
        self._stt_thread = threading.Thread(target=self._stt_loop, name="stt", daemon=True)
        self._vad_done = threading.Event()

    # --- public API (any thread) ------------------------------------------------

    def start(self) -> None:
        self._vad_thread.start()
        self._stt_thread.start()

    def feed(self, channel: str, t: float, samples: np.ndarray) -> None:
        """Capture thread: hand over a chunk starting at session time `t`."""
        try:
            self._chunks.put_nowait(("chunk", channel, t, samples))
        except queue.Full:
            self.dropped_chunks += 1

    def flush(self, channel: str | None = None) -> None:
        """End open utterances (mute, pause, source switch)."""
        self._chunks.put(("flush", channel, 0.0, None))

    def add_note(self, text: str) -> None:
        # Goes through the chunk queue so it is ordered after audio already captured.
        self._chunks.put(("note", None, 0.0, Note(at=self.clock(), text=text)))

    def stop(self, timeout: float = 120.0) -> None:
        """Finish all pending speech, write it, and stop the workers."""
        self._chunks.put(("stop", None, 0.0, None))
        self._vad_thread.join(timeout)
        self._stt_thread.join(timeout)

    def stats(self) -> PipelineStats:
        return PipelineStats(
            model_state=self.model_state,
            queue=self._finals.qsize(),
            latency=self.latency,
            segments=self.writer.segments if self.writer else 0,
            dropped_chunks=self.dropped_chunks,
            busy=self._busy,
        )

    # --- VAD thread ---------------------------------------------------------------

    def _vad_loop(self) -> None:
        while True:
            kind, channel, t, samples = self._chunks.get()
            if kind == "chunk":
                self._process_chunk(channel, t, samples)
            elif kind == "flush":
                for ch in [channel] if channel else self.channels:
                    self._flush_channel(ch)
            elif kind == "note":
                # Added here, after all audio captured before the note has been
                # through VAD, so any speech that started earlier is already in
                # flight and holds the watermark.
                with self._merger_lock:
                    self._merger.add(samples)
            elif kind == "stop":
                for ch in self.channels:
                    self._flush_channel(ch)
                self._vad_done.set()
                self._finals.put(None)  # tells the STT thread to drain and exit
                return

    def _process_chunk(self, ch: str, t: float, samples: np.ndarray) -> None:
        buf_t = self._buf_t[ch]
        vad = self._vads[ch]
        pending = len(vad._pending)
        if buf_t is None or abs((buf_t + pending / SAMPLE_RATE) - t) > DISCONTINUITY_S:
            self._flush_channel(ch)
            buf_t = t
        for window, prob in vad.process(samples):
            self._handle(self._segs[ch].push(buf_t, window, prob))
            buf_t += WINDOW / SAMPLE_RATE
        self._buf_t[ch] = buf_t

    def _flush_channel(self, ch: str) -> None:
        self._handle(self._segs[ch].flush())
        self._vads[ch].reset()
        self._buf_t[ch] = None
        with self._partials_lock:
            self._partials.pop(ch, None)

    def _handle(self, events: list) -> None:
        for ev in events:
            match ev:
                case SpeechStart(ch, start, seq):
                    with self._inflight_lock:
                        self._inflight[(ch, seq)] = start
                case SpeechDiscarded(ch, _start, seq):
                    with self._inflight_lock:
                        self._inflight.pop((ch, seq), None)
                    with self._partials_lock:
                        self._partials.pop(ch, None)
                    self.bus.publish(PartialEvent(ch, seq, _start, ""))
                case Utterance() as utt:
                    with self._partials_lock:
                        self._partials.pop(utt.channel, None)
                    self._finals.put(utt)
                case Partial() as p if self.partials:
                    with self._partials_lock:
                        self._partials[p.channel] = p

    # --- STT thread ---------------------------------------------------------------

    def _stt_loop(self) -> None:
        try:
            self.transcriber.load()
            self.model_state = "ready"
            self.bus.publish(StatusEvent("info", f"{self.transcriber.name} ready"))
        except Exception as e:  # noqa: BLE001 - report any load failure to the UI
            self.model_state = "error"
            self.bus.publish(StatusEvent("error", f"could not load {self.transcriber.name}: {e}"))
            self._drain_without_model()
            return

        draining = False
        while True:
            try:
                utt = self._finals.get(timeout=0.2)
            except queue.Empty:
                utt = False  # nothing final waiting
            if utt is None:
                draining = True
                continue
            if utt is not False:
                self._do_final(utt)
                continue
            if draining and self._vad_done.is_set() and self._finals.empty():
                self._release(final=True)
                return
            if not draining:
                partial = self._take_partial()
                if partial is not None:
                    self._do_partial(partial)
                    continue
            self._release()

    def _take_partial(self) -> Partial | None:
        with self._partials_lock:
            if not self._partials:
                return None
            ch = min(self._partials, key=lambda c: self._partials[c].start)
            return self._partials.pop(ch)

    def _do_partial(self, p: Partial) -> None:
        self._busy = True
        try:
            tr = self.transcriber.transcribe(p.audio, self._prompt(p.channel), final=False)
        except Exception as e:  # noqa: BLE001
            self.bus.publish(StatusEvent("warning", f"partial transcription failed: {e}", p.channel))
            return
        finally:
            self._busy = False
        with self._inflight_lock:
            still_open = (p.channel, p.seq) in self._inflight
        if still_open and tr.text:
            self.bus.publish(PartialEvent(p.channel, p.seq, p.start, tr.text))

    def _do_final(self, utt: Utterance) -> None:
        self._busy = True
        try:
            tr = self.transcriber.transcribe(utt.audio, self._prompt(utt.channel), final=True)
        except Exception as e:  # noqa: BLE001
            self.bus.publish(StatusEvent("error", f"transcription failed: {e}", utt.channel))
            tr = None
        finally:
            self._busy = False
        if tr is not None and tr.text:
            seg = Segment(
                channel=utt.channel,
                start=round(utt.start, 3),
                end=round(utt.end, 3),
                text=tr.text,
                no_speech_prob=None if tr.no_speech_prob is None else round(tr.no_speech_prob, 4),
                avg_logprob=None if tr.avg_logprob is None else round(tr.avg_logprob, 4),
            )
            self._last_text[utt.channel] = tr.text
            with self._merger_lock:
                self._merger.add(seg)
            self.latency = max(0.0, self.clock() - utt.end)
        # clears the live partial line for this utterance even if nothing was said
        self.bus.publish(PartialEvent(utt.channel, utt.seq, utt.start, ""))
        with self._inflight_lock:
            self._inflight.pop((utt.channel, utt.seq), None)
        self._release()

    def _watermark(self) -> float:
        with self._inflight_lock:
            return min(self._inflight.values(), default=float("inf"))

    def _release(self, final: bool = False) -> None:
        with self._merger_lock:
            items = self._merger.release_all() if final else self._merger.release(self._watermark())
        for item in items:
            self._emit(item)

    def _emit(self, item: Item) -> None:
        md = ""
        if self.writer is not None:
            try:
                md = self.writer.item(item)
            except OSError as e:
                self.bus.publish(StatusEvent("error", f"could not write transcript: {e}"))
        self.bus.publish(FinalEvent(item, md))

    def _prompt(self, channel: str) -> str | None:
        text = self._last_text.get(channel, "")
        return text[-PROMPT_CHARS:] or None

    def _drain_without_model(self) -> None:
        while True:
            utt = self._finals.get()
            if utt is None and self._vad_done.is_set():
                return


# --- offline (replay) ---------------------------------------------------------------


def segment_audio(channel: str, audio: np.ndarray, chunk: int = 4096) -> list[Utterance]:
    vad, seg = StreamingVAD(), Segmenter(channel, partial_interval_s=float("inf"))
    out: list[Utterance] = []
    t = 0.0
    for i in range(0, len(audio), chunk):
        for window, prob in vad.process(audio[i : i + chunk]):
            out += [e for e in seg.push(t, window, prob) if isinstance(e, Utterance)]
            t += WINDOW / SAMPLE_RATE
    out += [e for e in seg.flush() if isinstance(e, Utterance)]
    return out


def transcribe_offline(
    audio: dict[str, np.ndarray],
    transcriber: Transcriber,
    writer: TranscriptWriter | None,
    progress: Callable[[int, int, Segment | None], None] | None = None,
) -> list[Segment]:
    utterances = sorted(
        (u for ch, a in audio.items() for u in segment_audio(ch, a)), key=lambda u: u.start
    )
    last_text: dict[str, str] = {}
    segments: list[Segment] = []
    for i, utt in enumerate(utterances):
        tr = transcriber.transcribe(utt.audio, last_text.get(utt.channel, "")[-PROMPT_CHARS:] or None)
        seg = None
        if tr.text:
            seg = Segment(utt.channel, round(utt.start, 3), round(utt.end, 3), tr.text,
                          no_speech_prob=tr.no_speech_prob, avg_logprob=tr.avg_logprob)  # fmt: skip
            last_text[utt.channel] = tr.text
            segments.append(seg)
            if writer is not None:
                writer.item(seg)
        if progress is not None:
            progress(i + 1, len(utterances), seg)
    return segments
