"""Animated level meter for one channel: scrolling sparkline + dB readout."""

from __future__ import annotations

import time
from collections import deque

from rich.text import Text
from textual.widget import Widget

from ..capture.devices import Device
from ..config import CHANNEL_LABELS, METER_FPS, SILENCE_DB
from ..events import LevelEvent, SourceState

BLOCKS = "▁▂▃▄▅▆▇█"  # silence draws a flat ▁ baseline; a blank means no data
METER_FLOOR_DB = -60.0
PEAK_HOLD_S = 1.5
ZERO_WARN_S = 3.0  # exact digital silence for this long -> "wrong device?" hint
HISTORY = 240  # samples kept; the widget shows as many as fit
DATA_HOLD_S = 0.5  # bridge empty frames shorter than this


def db_to_block(db: float) -> str:
    frac = (db - METER_FLOOR_DB) / -METER_FLOOR_DB
    frac = min(max(frac, 0.0), 1.0)
    return BLOCKS[round(frac * (len(BLOCKS) - 1))]


def level_style(db: float) -> str:
    if db >= -3.0:
        return "bold red"
    if db >= -12.0:
        return "yellow"
    return "green"


class Meter(Widget):
    DEFAULT_CSS = """
    Meter {
        height: 3;
        padding: 0 1;
    }
    """

    def __init__(self, channel: str, device: Device, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channel = channel
        self.device = device
        self.state: SourceState = "starting"
        self.state_detail = ""
        self.muted = False
        self.history: deque[float | None] = deque(maxlen=HISTORY)
        self.rms_db: float | None = None
        self.peak_db: float | None = None
        self.peak_hold: float | None = None
        self._peak_hold_at = 0.0
        self._last_sound = time.monotonic()
        self._last_data = 0.0
        self._zero_since: float | None = None
        self.clipped_at = 0.0

    # --- updates from the app timer ----------------------------------------

    def push(self, level: LevelEvent | None) -> None:
        now = time.monotonic()
        if level is None:
            # Sources deliver in bursts (Bluetooth especially), so an empty
            # frame is normal: hold the last value unless data has really stopped.
            held = now - self._last_data < DATA_HOLD_S
            self.history.append(self.rms_db if held else None)
        else:
            self._last_data = now
            self.history.append(level.rms_db)
            self.rms_db = level.rms_db
            self.peak_db = level.peak_db
            if self.peak_hold is None or level.peak_db >= self.peak_hold or now - self._peak_hold_at > PEAK_HOLD_S:
                self.peak_hold = level.peak_db
                self._peak_hold_at = now
            if level.rms_db > SILENCE_DB:
                self._last_sound = now
            if level.digital_silence:
                self._zero_since = self._zero_since or now
            else:
                self._zero_since = None
            if level.clipped:
                self.clipped_at = now
        self.refresh()

    def set_source(self, device: Device, state: SourceState, detail: str = "") -> None:
        if device != self.device:
            self.history.clear()
            self._zero_since = None
        self.device = device
        self.state = state
        self.state_detail = detail
        self.refresh()

    # --- rendering ----------------------------------------------------------

    def _flags(self, now: float) -> Text:
        t = Text()
        if self.state == "lost":
            t.append(" LOST ", "bold white on red")
        elif self.state in ("starting", "stopped"):
            t.append(f" {self.state} ", "dim")
        if self.muted:
            t.append(" MUTED ", "bold black on yellow")
        if now - self.clipped_at < PEAK_HOLD_S:
            t.append(" CLIP ", "bold white on red")
        return t

    def _note(self, now: float) -> Text:
        if self.state == "lost":
            return Text(self.state_detail or "source lost", "red")
        if self._zero_since is not None and now - self._zero_since >= ZERO_WARN_S:
            return Text("no signal (exact zeros): device muted or wrong source?", "yellow")
        silent = now - self._last_sound
        if silent >= 2.0:
            return Text(f"silent {silent:.0f} s", "dim")
        return Text("")

    def render(self) -> Text:
        now = time.monotonic()
        width = max(self.size.width - 2, 20)  # minus padding

        # line 1: label, device, flags
        line1 = Text()
        line1.append(f"{CHANNEL_LABELS[self.channel]:<7}", "bold")
        line1.append(" ")
        flags = self._flags(now)
        room = width - line1.cell_len - flags.cell_len - 1
        line1.append(self.device.description[: max(room, 0)], "cyan" if not self.muted else "dim")
        line1.pad_right(width - line1.cell_len - flags.cell_len)
        line1.append_text(flags)

        # line 2: sparkline history + readout
        readout = Text()
        if self.rms_db is None:
            readout.append("   -- dB", "dim")
        else:
            readout.append(f"{self.rms_db:6.1f} dB", "bold" if not self.muted else "dim")
            if self.peak_hold is not None:
                readout.append(f"  peak {self.peak_hold:5.1f}", level_style(self.peak_hold) if not self.muted else "dim")
        spark_w = max(width - readout.cell_len - 2, 10)
        hist = list(self.history)[-spark_w:]
        spark = Text(" " * (spark_w - len(hist)))
        for db in hist:
            if db is None:
                spark.append(" ")
            else:
                spark.append(db_to_block(db), "grey50" if self.muted else level_style(db))
        line2 = spark + Text("  ") + readout

        # line 3: note (silence duration, warnings)
        return Text("\n").join([line1, line2, self._note(now)])


# Frame interval used by the app timer
FRAME_S = 1.0 / METER_FPS
