"""Textual live monitor. Build step 2: level meters, activity log, mute and quit.

The app never touches audio directly: it polls `Controller.poll_levels()` on a
fixed timer and drains the controller's event bus. Commands go back through
`Controller.apply()`.
"""

from __future__ import annotations

import time
from datetime import datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, RichLog, Static

from ..config import CHANNEL_LABELS
from ..controller import Controller
from ..events import SetMute, SourceEvent, StatusEvent
from .meters import FRAME_S, Meter


def fmt_elapsed(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class TuiscribeApp(App):
    TITLE = "tuiscribe"
    CSS = """
    Screen { layout: vertical; }
    #title { height: 1; padding: 0 1; background: $boost; }
    #meters { height: auto; border: round $primary 50%; }
    #activity { height: 1fr; border: round $primary 50%; padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("M", "toggle_mute('mic')", "mute mic"),
        Binding("S", "toggle_mute('system')", "mute system"),
        Binding("q", "quit_clean", "quit"),
        Binding("ctrl+c", "quit_clean", "quit", show=False, priority=True),
    ]

    def __init__(self, controller: Controller, duration: float | None = None) -> None:
        super().__init__()
        self.controller = controller
        self.duration = duration
        self._cpu_last = (time.monotonic(), time.process_time())
        self._cpu_pct = 0.0

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        with Vertical(id="meters"):
            for ch, dev in self.controller.devices.items():
                yield Meter(ch, dev, id=f"meter-{ch}")
        log = RichLog(id="activity", wrap=True, markup=False, max_lines=500)
        log.border_title = "activity"
        yield log
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#meters").border_title = "input levels"
        self.controller.start()
        self._log("info", "Capturing. Transcription is not wired in yet (build steps 3-5).")
        if self.controller.save_audio is not None:
            self._log("info", f"Saving raw audio to {self.controller.save_audio}/")
        self.set_interval(FRAME_S, self._tick)

    # --- timer ----------------------------------------------------------------

    def _tick(self) -> None:
        c = self.controller
        for ch, level in c.poll_levels().items():
            self.query_one(f"#meter-{ch}", Meter).push(level)
        for ev in c.bus.drain():
            self._handle_event(ev)
        self._update_title()
        self._update_status()
        if self.duration is not None and c.elapsed() >= self.duration:
            self.action_quit_clean()

    def _handle_event(self, ev: object) -> None:
        match ev:
            case SourceEvent(channel, device, state, detail):
                self.query_one(f"#meter-{channel}", Meter).set_source(device, state, detail)
                if state == "lost":
                    self._log("error", f"{CHANNEL_LABELS[channel]} lost: {detail}")
                elif state == "running":
                    self._log("info", f"{CHANNEL_LABELS[channel]} ← {device.description}")
            case StatusEvent(severity, message, _):
                self._log(severity, message)

    def _update_title(self) -> None:
        c = self.controller
        t = Text()
        t.append("tuiscribe", "bold")
        t.append("  ·  ")
        t.append("● REC ", "bold red")
        t.append(fmt_elapsed(c.elapsed()))
        t.append("  ·  capture only (STT not yet wired)", "dim")
        self.query_one("#title", Static).update(t)

    def _update_status(self) -> None:
        now, cpu = time.monotonic(), time.process_time()
        last_wall, last_cpu = self._cpu_last
        if now - last_wall >= 1.0:
            self._cpu_pct = 100.0 * (cpu - last_cpu) / (now - last_wall)
            self._cpu_last = (now, cpu)
        parts = [f"CPU {self._cpu_pct:4.1f}%"]
        saved = self.controller.saved_seconds()
        if saved:
            parts.append("saved " + ", ".join(f"{ch} {s:.0f}s" for ch, s in saved.items()))
        if self.controller.bus.dropped:
            parts.append(f"dropped events {self.controller.bus.dropped}")
        self.query_one("#status", Static).update("  ·  ".join(parts))

    def _log(self, severity: str, message: str) -> None:
        style = {"info": "", "warning": "yellow", "error": "bold red"}.get(severity, "")
        line = Text(datetime.now().strftime("%H:%M:%S "), "dim")
        line.append(message, style)
        self.query_one("#activity", RichLog).write(line)

    # --- actions --------------------------------------------------------------

    def action_toggle_mute(self, channel: str) -> None:
        if channel not in self.controller.devices:
            return
        self.controller.apply(SetMute(channel))
        self.query_one(f"#meter-{channel}", Meter).muted = self.controller.muted[channel]

    def action_quit_clean(self) -> None:
        if not self.controller.stopped.is_set():
            self.controller.stop()
        self.exit()
