"""Owns the capture pipeline and applies commands from the UI.

No UI imports: the TUI, `--no-tui` and (later) `replay` all drive this class.
Build steps 1-2 cover capture, levels, mute, source switching and optional
raw-audio saving; VAD/STT stages attach to `_on_chunk` in later steps.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio.wav import AlignedWavWriter
from .capture.devices import Device
from .capture.levels import LevelAccumulator
from .capture.pw_record import PwRecordSource
from .config import DIGITAL_SILENCE
from .events import (
    Command,
    EventBus,
    LevelEvent,
    Quit,
    SetMute,
    SourceEvent,
    StatusEvent,
    SwitchSource,
)

CLIP_LINEAR = 0.999


class Controller:
    def __init__(
        self,
        devices: dict[str, Device],  # channel -> device, e.g. {"mic": ..., "system": ...}
        save_audio: Path | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.bus = bus or EventBus()
        self.devices = dict(devices)
        self.save_audio = save_audio
        self.muted: dict[str, bool] = {ch: False for ch in devices}
        self.session_start = 0.0
        self.started_at: datetime | None = None
        self._levels = {ch: LevelAccumulator() for ch in devices}
        self._sources: dict[str, PwRecordSource] = {}
        self._writers: dict[str, AlignedWavWriter] = {}
        self._cmd_lock = threading.Lock()
        self.stopped = threading.Event()

    @property
    def channels(self) -> list[str]:
        return list(self.devices)

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.session_start = time.monotonic()
        self.started_at = datetime.now().astimezone()
        if self.save_audio is not None:
            self.save_audio.mkdir(parents=True, exist_ok=True)
            for ch in self.channels:
                self._writers[ch] = AlignedWavWriter(
                    self.save_audio / f"{ch}.wav", self.session_start
                )
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
        for src in self._sources.values():
            src.stop()
        for w in self._writers.values():
            w.close()
        for ch, src in self._sources.items():
            self.bus.publish(SourceEvent(ch, src.device, "stopped"))
        self.stopped.set()

    def elapsed(self) -> float:
        return time.monotonic() - self.session_start if self.session_start else 0.0

    # --- commands -----------------------------------------------------------

    def apply(self, cmd: Command) -> None:
        with self._cmd_lock:
            match cmd:
                case SetMute(channel, muted):
                    new = (not self.muted[channel]) if muted is None else muted
                    self.muted[channel] = new
                    self.bus.publish(
                        StatusEvent("info", f"{channel} {'muted' if new else 'unmuted'}", channel)
                    )
                case SwitchSource(channel, device):
                    src = self._sources[channel]
                    self.devices[channel] = device
                    src.stop()
                    src.device = device
                    self._start_source(src)
                case Quit():
                    self.stop()

    # --- levels (polled by the UI at a fixed rate) --------------------------

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
        # Build step 3+: hand (channel, t0, samples) to the VAD/segmenter here.

    def _on_source_error(self, channel: str, message: str) -> None:
        self.bus.publish(SourceEvent(channel, self.devices[channel], "lost", message))
