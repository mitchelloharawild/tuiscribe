"""Owns the capture pipeline and applies commands from the UI.

No UI imports: the TUI and `--no-tui` both drive this class. Capture threads
call `_on_chunk`, which updates the meters and hands audio to the
transcription `Pipeline` (unless the channel is muted or transcription is
paused).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .audio.wav import AlignedWavWriter, WavSpans
from .capture.devices import Device
from .capture.levels import LevelAccumulator
from .capture.pw_record import PwRecordSource
from .config import CHANNEL_LABELS, DIGITAL_SILENCE
from .events import (
    AddNote,
    Command,
    EventBus,
    LevelEvent,
    NameSpeaker,
    Quit,
    SetDiarize,
    SetMute,
    SetPause,
    SourceEvent,
    StatusEvent,
    SwitchSource,
)
from .output import TranscriptWriter
from .pipeline import Pipeline, PipelineStats
from .records import Header
from .render import RenderOptions
from .stt import Transcriber

if TYPE_CHECKING:
    from .diarize import Diarizer

CLIP_LINEAR = 0.999


class Controller:
    def __init__(
        self,
        devices: dict[str, Device],  # channel -> device, e.g. {"mic": ..., "system": ...}
        output: Path | None = None,  # transcript .md (sidecar .jsonl next to it)
        transcriber: Transcriber | None = None,  # None: capture + meters only
        render_opts: RenderOptions = RenderOptions(),
        audio_paths: dict[str, Path] | None = None,  # channel -> .wav to write
        partials: bool = True,
        bus: EventBus | None = None,
        diarizer: Diarizer | None = None,  # speaker labels within channels (switchable live)
    ) -> None:
        self.bus = bus or EventBus()
        self.devices = dict(devices)
        self.output = output
        self.transcriber = transcriber
        self.render_opts = render_opts
        self.audio_paths = {ch: p for ch, p in (audio_paths or {}).items() if ch in devices}
        self.partials = partials
        self.diarizer = diarizer
        self.relabelled = 0  # segments whose speaker changed in the end-of-session re-clustering
        self.learned: dict[str, int] = {}  # voice samples saved at stop, per confirmed name
        self.muted: dict[str, bool] = {ch: False for ch in devices}
        self.paused = False
        self.session_start = 0.0
        self.started_at: datetime | None = None
        self.writer: TranscriptWriter | None = None
        self.pipeline: Pipeline | None = None
        self._levels = {ch: LevelAccumulator() for ch in devices}
        self._sources: dict[str, PwRecordSource] = {}
        self._writers: dict[str, AlignedWavWriter] = {}
        self._cmd_lock = threading.Lock()
        self._switch_lock = threading.Lock()
        self.stopping = threading.Event()
        self.stopped = threading.Event()

    @property
    def channels(self) -> list[str]:
        return list(self.devices)

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.session_start = time.monotonic()
        self.started_at = datetime.now().astimezone()
        for ch, path in self.audio_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writers[ch] = AlignedWavWriter(path, self.session_start)
        if self.transcriber is not None:
            if self.output is not None:
                self.writer = TranscriptWriter(self.output, self.render_opts)
                self.writer.header(
                    Header(
                        started_at=self.started_at,
                        sources={ch: d.description for ch, d in self.devices.items()},
                        model=self.transcriber.name,
                    )
                )
            self.pipeline = Pipeline(
                self.channels,
                self.transcriber,
                self.writer,
                self.bus,
                self.elapsed,
                partials=self.partials,
                diarizer=self.diarizer,
            )
            self.pipeline.start()
        for ch, dev in self.devices.items():
            src = PwRecordSource(ch, dev, self._on_chunk, self._on_source_error)
            self._sources[ch] = src
            self._start_source(src)
        system = self.devices.get("system")
        if system is not None and system.looks_like_speakers:
            self.bus.publish(
                StatusEvent(
                    "warning",
                    f"System audio is going to '{system.description}', which looks like "
                    "speakers. Without headphones the mic will also pick up remote voices.",
                    "system",
                )
            )

    def stop(self) -> None:
        """Stop capture, transcribe what is still pending, write the footer. May take a while."""
        with self._cmd_lock:  # no source switch or note can start mid-stop
            if self.stopping.is_set():
                already = True
            else:
                already = False
                self.stopping.set()
        if already:
            self.stopped.wait()
            return
        try:
            with self._switch_lock:  # wait for an in-flight switch to finish
                for src in self._sources.values():
                    src.stop()
            for w in self._writers.values():
                w.close()
            for ch, src in self._sources.items():
                self.bus.publish(SourceEvent(ch, src.device, "stopped"))
            finished = self.pipeline.stop() if self.pipeline is not None else True
            if self.writer is not None:
                if finished:
                    self._relabel()
                    self.writer.footer(datetime.now().astimezone(), self.elapsed())
                else:
                    self.writer.close()
        finally:
            self.stopped.set()

    def _relabel(self) -> None:
        """Rewrite this session's speaker labels from whole-session clustering."""
        if self.pipeline is None or self.writer is None:
            return
        try:
            speakers = self.pipeline.final_speakers() if self.diarizer is None else self._final_speakers()
            self.relabelled = self.writer.relabel(speakers)
        except Exception as e:  # noqa: BLE001 - keep the live labels; still write the footer
            self.bus.publish(StatusEvent("error", f"could not update speaker labels: {e}"))
            return
        if self.relabelled:
            self.bus.publish(StatusEvent("info", f"speaker labels revised for {self.relabelled} segments"))
        if self.diarizer is not None:
            self._learn_voices()

    def _learn_voices(self) -> None:
        """Save the voices of the speakers named by hand this session."""
        d = self.diarizer
        assert d is not None and self.writer is not None
        if not any(d.confirmed.values()):
            return
        if d.voices is None:
            self.bus.publish(StatusEvent("info", "voices not saved (--no-voices)"))
            return
        try:
            added = d.learn(str(self.writer.md_path.resolve()))
        except (OSError, ValueError) as e:
            self.bus.publish(StatusEvent("error", f"could not save voices: {e}"))
            return
        self.learned = added
        msg = ", ".join(f"{name} +{n}" for name, n in sorted(added.items()))
        self.bus.publish(StatusEvent("info", f"voice samples saved: {msg}"))

    def _final_speakers(self) -> dict[tuple[str, float], str | None]:
        """Whichever channels are diarized when the session stops get labels for the
        whole session: segments from before they were switched on are embedded from
        the saved audio. Channels switched off again lose their labels."""
        d = self.diarizer
        assert d is not None and self.writer is not None
        segs = self.writer.session_segments()
        spans = WavSpans({ch: p for ch, p in self.audio_paths.items() if ch in d.channels})
        try:
            todo = d.missing([s for s in segs if s.channel in spans.channels])
            if todo:
                self.bus.publish(StatusEvent("info", f"labelling speakers in {todo} earlier segments from the saved audio…"))
                d.load()
            labels = d.final_labels(segs, spans)
        finally:
            spans.close()
        return {(s.channel, s.start): labels.get((s.channel, s.start)) for s in segs}

    def elapsed(self) -> float:
        return time.monotonic() - self.session_start if self.session_start else 0.0

    # --- commands -----------------------------------------------------------

    def apply(self, cmd: Command) -> None:
        if isinstance(cmd, SwitchSource):
            self._switch(cmd.channel, cmd.device)
            return
        with self._cmd_lock:
            if self.stopping.is_set():
                return
            match cmd:
                case SetMute(channel, muted):
                    new = (not self.muted[channel]) if muted is None else muted
                    self.muted[channel] = new
                    if new and self.pipeline is not None:
                        self.pipeline.flush(channel)
                    self.bus.publish(
                        StatusEvent("info", f"{CHANNEL_LABELS[channel]} {'muted' if new else 'unmuted'}", channel)
                    )
                case SetPause(paused):
                    new = (not self.paused) if paused is None else paused
                    self.paused = new
                    if new and self.pipeline is not None:
                        self.pipeline.flush()
                    self.bus.publish(StatusEvent("info", "transcription paused" if new else "transcription resumed"))
                case SetDiarize(channel, on):
                    self._set_diarize(channel, on)
                case NameSpeaker(channel, index, name):
                    self._name_speaker(channel, index, name)
                case AddNote(text):
                    if self.pipeline is None:
                        self.bus.publish(StatusEvent("warning", "notes need transcription enabled"))
                    elif text.strip():
                        self.pipeline.add_note(text)
                case Quit():
                    pass  # the caller runs stop(), which can block while STT drains

    def _set_diarize(self, channel: str, on: bool | None) -> None:
        d = self.diarizer
        label = CHANNEL_LABELS[channel]
        if d is None or self.pipeline is None:
            self.bus.publish(StatusEvent("warning", "speaker labels need transcription enabled"))
            return
        new = (not d.applies(channel)) if on is None else on
        if new == d.applies(channel):
            return
        d.set_enabled(channel, new)
        if not new:
            self.bus.publish(StatusEvent("info", f"{label} speaker labels off (none in the transcript for {label})", channel))
            return
        if channel in self.audio_paths:
            msg = f"{label} speaker labels on; earlier {label} speech is labelled when you stop"
        else:
            msg = f"{label} speaker labels on from now (audio not saved, so earlier speech stays unlabelled)"
        self.bus.publish(StatusEvent("info", msg, channel))
        if not d.ready:
            threading.Thread(target=self._load_diarizer, args=(channel,), name="speaker-model", daemon=True).start()

    def _name_speaker(self, channel: str, index: int, name: str | None) -> None:
        d = self.diarizer
        if d is None or not d.applies(channel):
            return
        before = next((sp.label for sp in d.live_speakers() if sp.channel == channel and sp.index == index), None)
        d.name_speaker(channel, index, name)
        after = next((sp.label for sp in d.live_speakers() if sp.channel == channel and sp.index == index), None)
        if name == before:
            msg = f"{name} confirmed" + ("" if d.voices is None else f"; {name}'s saved voice is updated when you stop")
        elif name:
            keep = "" if d.voices is None else f" and {name}'s voice is saved"
            msg = f"{before} is {name} from now; when you stop, earlier lines are relabelled{keep}"
        else:
            msg = f"{before} is no longer named; now {after}"
        self.bus.publish(StatusEvent("info", msg, channel))

    def _load_diarizer(self, channel: str) -> None:
        d = self.diarizer
        assert d is not None
        try:
            d.load()
        except Exception as e:  # noqa: BLE001 - transcription goes on without speaker labels
            for ch in list(d.channels):
                d.set_enabled(ch, False)
            self.bus.publish(StatusEvent("warning", f"speaker labels off: could not load the speaker model: {e}", channel))

    def _switch(self, channel: str, device: Device) -> None:
        # Slow (stops and starts pw-record), so it has its own lock and does not
        # block mute/pause/note, which only take _cmd_lock briefly.
        with self._switch_lock:
            if self.stopping.is_set():
                return
            src = self._sources[channel]
            self.devices[channel] = device
            src.stop()
            if self.pipeline is not None:
                self.pipeline.flush(channel)
            src.device = device
            if not self.stopping.is_set():
                self._start_source(src)

    # --- polling (UI timer) ------------------------------------------------

    def poll_levels(self) -> dict[str, LevelEvent | None]:
        out: dict[str, LevelEvent | None] = {}
        for ch, acc in self._levels.items():
            lvl = acc.take()
            out[ch] = (
                None
                if lvl is None
                else LevelEvent(
                    ch,
                    lvl.rms_db,
                    lvl.peak_db,
                    clipped=lvl.peak >= CLIP_LINEAR,
                    digital_silence=lvl.peak < DIGITAL_SILENCE,
                )
            )
        return out

    def stats(self) -> PipelineStats | None:
        return self.pipeline.stats() if self.pipeline is not None else None

    def saved_seconds(self) -> dict[str, float]:
        return {ch: w.seconds for ch, w in self._writers.items()}

    # --- internals ----------------------------------------------------------

    def _start_source(self, src: PwRecordSource) -> None:
        self.bus.publish(SourceEvent(src.channel, src.device, "starting"))
        try:
            src.start()
        except OSError as e:
            self.bus.publish(SourceEvent(src.channel, src.device, "lost", str(e)))
            return
        self.bus.publish(SourceEvent(src.channel, src.device, "running"))

    def _on_chunk(self, channel: str, t0: float, samples: np.ndarray) -> None:
        # Capture thread. Levels are measured even when muted so the meter
        # still shows the device is alive; muted audio goes no further.
        self._levels[channel].add(samples)
        if self.muted[channel]:
            return
        writer = self._writers.get(channel)
        if writer is not None:
            writer.write(t0, samples)
        if self.pipeline is not None and not self.paused:
            self.pipeline.feed(channel, t0 - self.session_start, samples)

    def _on_source_error(self, channel: str, message: str) -> None:
        if self.pipeline is not None:
            self.pipeline.flush(channel)  # otherwise its open speech holds back the other channel
        self.bus.publish(SourceEvent(channel, self.devices[channel], "lost", message))
