"""Textual live monitor: meters, transcript preview, activity log, controls.

The app never touches audio directly: it polls `Controller.poll_levels()` and
`Controller.stats()` on a fixed timer and drains the controller's event bus.
Commands go back through `Controller.apply()`; anything that can block
(switching a source, stopping) runs in a worker thread.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, RichLog, Static

from ..capture.devices import Device
from ..config import CHANNEL_LABELS
from ..controller import Controller
from ..events import AddNote, FinalEvent, PartialEvent, SetMute, SetPause, SourceEvent, StatusEvent, SwitchSource
from .meters import FRAME_S, Meter
from .screens import HelpScreen, NoteInput, SourcePicker
from .transcript import TranscriptView


def fmt_elapsed(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class TuiscribeApp(App):
    TITLE = "tuiscribe"
    CSS = """
    Screen { layout: vertical; }
    #title { height: 1; padding: 0 1; background: $boost; }
    #meters { height: auto; border: round $primary 50%; }
    #activity { height: 6; border: round $primary 30%; padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("m", "pick_source('mic')", "mic"),
        Binding("s", "pick_source('system')", "system"),
        Binding("M", "toggle_mute('mic')", "mute mic"),
        Binding("S", "toggle_mute('system')", "mute sys"),
        Binding("space", "toggle_pause", "pause"),
        Binding("n", "note", "note"),
        Binding("question_mark", "help", "help"),
        Binding("q", "quit_clean", "quit"),
        Binding("ctrl+c", "quit_clean", "quit", show=False, priority=True),
    ]

    def __init__(self, controller: Controller, duration: float | None = None) -> None:
        super().__init__()
        self.controller = controller
        self.duration = duration
        self._cpu_last = (time.monotonic(), time.process_time())
        self._cpu_pct = 0.0
        self._finishing = False
        self._quit_presses = 0
        self.abandoned = False

    # --- layout -----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        with Vertical(id="meters"):
            for ch, dev in self.controller.devices.items():
                yield Meter(ch, dev, id=f"meter-{ch}")
        if self.controller.pipeline is not None or self.controller.transcriber is not None:
            yield TranscriptView(self.controller.render_opts, id="transcript")
        log = RichLog(id="activity", wrap=True, markup=False, max_lines=500)
        log.border_title = "activity"
        yield log
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#meters").border_title = "input levels"
        self.controller.start()
        if self.controller.started_at is not None and self.controller.transcriber is not None:
            self.query_one(TranscriptView).started_at = self.controller.started_at
        c = self.controller
        if c.writer is not None:
            self._log("info", f"Writing {c.writer.md_path} (+ {c.writer.jsonl_path.name})")
        elif c.transcriber is None:
            self._log("info", "Meters only (--no-transcribe).")
        if c.transcriber is not None:
            self._log("info", f"Loading {c.transcriber.name} ... (first run downloads the model)")
        if c.save_audio is not None:
            self._log("info", f"Saving raw audio to {c.save_audio}/")
        self._log("info", "Press ? for keys.")
        self.set_interval(FRAME_S, self._tick)

    # --- timer ------------------------------------------------------------------

    def _tick(self) -> None:
        c = self.controller
        for ch, level in c.poll_levels().items():
            self.query_one(f"#meter-{ch}", Meter).push(level)
        for ev in c.bus.drain():
            self._handle_event(ev)
        self._update_title()
        self._update_status()
        if self.duration is not None and c.elapsed() >= self.duration and not self._finishing:
            self.action_quit_clean()

    def _handle_event(self, ev: object) -> None:
        match ev:
            case SourceEvent(channel, device, state, detail):
                self.query_one(f"#meter-{channel}", Meter).set_source(device, state, detail)
                if state == "lost":
                    self._log("error", f"{CHANNEL_LABELS[channel]} lost: {detail}. Press "
                              f"{'m' if channel == 'mic' else 's'} to pick another source.")
                elif state == "running":
                    self._log("info", f"{CHANNEL_LABELS[channel]} ← {device.description}")
            case StatusEvent(severity, message, _):
                self._log(severity, message)
            case PartialEvent(channel, seq, start, text):
                self._transcript().set_partial(channel, seq, start, text)
            case FinalEvent(item, _md):
                self._transcript().add(item)

    def _transcript(self) -> TranscriptView:
        return self.query_one(TranscriptView)

    def _update_title(self) -> None:
        c = self.controller
        t = Text()
        t.append("tuiscribe", "bold")
        if c.writer is not None:
            t.append(f" · {c.writer.md_path.name}")
        t.append("  ·  ")
        if self._finishing:
            t.append("■ FINISHING ", "bold yellow")
        elif c.paused:
            t.append("❚❚ PAUSED ", "bold yellow")
        else:
            t.append("● REC ", "bold red")
        t.append(fmt_elapsed(c.elapsed()))
        if c.transcriber is not None:
            t.append(f"  ·  {c.transcriber.name}", "dim")
        self.query_one("#title", Static).update(t)

    def _update_status(self) -> None:
        now, cpu = time.monotonic(), time.process_time()
        last_wall, last_cpu = self._cpu_last
        if now - last_wall >= 1.0:
            self._cpu_pct = 100.0 * (cpu - last_cpu) / (now - last_wall)
            self._cpu_last = (now, cpu)
        parts: list[str] = []
        st = self.controller.stats()
        if st is not None:
            if st.model_state == "loading":
                parts.append("STT loading model…")
            elif st.model_state == "error":
                parts.append("STT unavailable")
            else:
                parts.append(f"STT queue {st.queue}{' ●' if st.busy else ''}")
                if st.latency is not None:
                    parts.append(f"latency {st.latency:.1f} s")
                parts.append(f"{st.segments} segments")
            if st.dropped_chunks:
                parts.append(f"dropped {st.dropped_chunks} chunks")
        parts.append(f"CPU {self._cpu_pct:.0f}%")
        saved = self.controller.saved_seconds()
        if saved:
            parts.append("audio " + ", ".join(f"{ch} {s:.0f}s" for ch, s in saved.items()))
        if self.controller.bus.dropped:
            parts.append(f"dropped events {self.controller.bus.dropped}")
        self.query_one("#status", Static).update("  ·  ".join(parts))

    def _log(self, severity: str, message: str) -> None:
        style = {"info": "", "warning": "yellow", "error": "bold red"}.get(severity, "")
        line = Text(datetime.now().strftime("%H:%M:%S "), "dim")
        line.append(message, style)
        self.query_one("#activity", RichLog).write(line)

    # --- actions ----------------------------------------------------------------

    def action_toggle_mute(self, channel: str) -> None:
        if channel not in self.controller.devices or self._finishing:
            return
        self.controller.apply(SetMute(channel))
        self.query_one(f"#meter-{channel}", Meter).muted = self.controller.muted[channel]

    def action_toggle_pause(self) -> None:
        if self.controller.transcriber is None or self._finishing:
            return
        self.controller.apply(SetPause())

    def action_pick_source(self, channel: str) -> None:
        if channel not in self.controller.devices:
            self._log("warning", f"{CHANNEL_LABELS[channel]} is not being recorded in this session.")
            return
        if self._finishing:
            return

        def chosen(device: Device | None) -> None:
            if self._finishing:
                return
            if device is not None and device.name != self.controller.devices[channel].name:
                self._switch(channel, device)

        self.push_screen(SourcePicker(channel, self.controller.devices[channel]), chosen)

    @work(thread=True, exclusive=False)
    def _switch(self, channel: str, device: Device) -> None:
        self.controller.apply(SwitchSource(channel, device))

    def action_note(self) -> None:
        if self.controller.pipeline is None or self._finishing:
            return

        def done(text: str | None) -> None:
            if self._finishing:
                self._log("warning", "Stopping: note not added.")
                return
            if text:
                self.controller.apply(AddNote(text))
                self._log("info", f"note queued: {text}")

        self.push_screen(NoteInput(), done)

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_quit_clean(self) -> None:
        self._quit_presses += 1
        if self._finishing:
            if self._quit_presses >= 3:
                self.abandoned = True  # cli exits the process; pending speech is lost
                self.exit()
            else:
                self._log("warning", "Still transcribing the last utterances. Press again to abandon them.")
            return
        self._finishing = True
        self._log("info", "Stopping capture; transcribing what is left…")
        self._stop_and_exit()

    def _stop_and_exit(self) -> None:
        # A plain daemon thread, not a Textual worker: Textual waits for its
        # executor threads at exit, which would defeat "press q to abandon".
        def run() -> None:
            self.controller.stop()
            try:
                self.call_from_thread(self._final_drain_and_exit)
            except RuntimeError:
                pass  # app already exited

        threading.Thread(target=run, name="stop", daemon=True).start()

    def _final_drain_and_exit(self) -> None:
        for ev in self.controller.bus.drain():
            self._handle_event(ev)
        self.exit()
