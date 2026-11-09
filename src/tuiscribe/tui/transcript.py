"""Transcript preview: finished items (as in the .md) plus dimmed live partials."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import RichLog, Static

from ..records import SPEAKER_FOR_CHANNEL, Item, Note, Segment
from ..render import RenderOptions, fmt_clock, fmt_duration

SPEAKER_STYLE = {"mic": "bold cyan", "system": "bold magenta"}


class TranscriptView(Vertical):
    DEFAULT_CSS = """
    TranscriptView { height: 1fr; border: round $primary 50%; padding: 0 1; }
    TranscriptView RichLog { height: 1fr; background: transparent; }
    TranscriptView #partials { height: auto; }
    """

    def __init__(self, opts: RenderOptions, **kwargs) -> None:
        super().__init__(**kwargs)
        self.opts = opts
        self.started_at: datetime | None = None
        self._prev: Item | None = None
        self._partials: dict[str, tuple[int, float, str]] = {}  # channel -> (seq, start, text)
        self._cleared: dict[str, int] = {}  # channel -> highest seq that has ended

    def compose(self):
        yield RichLog(wrap=True, markup=False, max_lines=2000, auto_scroll=True, id="finals")
        yield Static(id="partials")

    def on_mount(self) -> None:
        self.border_title = "transcript"

    def add(self, item: Item) -> None:
        log = self.query_one("#finals", RichLog)
        prev = self._prev
        self._prev = item
        if isinstance(item, Note):
            log.write(Text(item.text, "bold yellow"))
            return
        seg = item
        new_block = prev is None or isinstance(prev, Note) or seg.label != prev.label
        if isinstance(prev, Segment):
            gap = seg.start - prev.end
            if gap > self.opts.gap_long:
                log.write(Text(f"⋯ {fmt_duration(gap)} of silence ⋯", "dim italic"))
                new_block = True
            elif gap >= self.opts.gap_short and not new_block:
                log.write(Text(""))
        if new_block:
            head = Text()
            head.append(seg.label, SPEAKER_STYLE.get(seg.channel, "bold"))
            if self.started_at is not None:
                head.append(f" · {fmt_clock(self.started_at, seg.start)}", "dim")
            if isinstance(prev, Segment) and seg.start < prev.end and seg.channel != prev.channel:
                head.append(" (overlapping)", "dim")
            log.write(head)
        log.write(Text(seg.text))

    def set_partial(self, channel: str, seq: int, start: float, text: str) -> None:
        cur = self._partials.get(channel)
        if text and seq <= self._cleared.get(channel, 0):
            return  # a late partial for an utterance that already ended
        if not text:
            self._cleared[channel] = max(seq, self._cleared.get(channel, 0))
            # clear only if it belongs to this utterance (or an older one)
            if cur is not None and cur[0] <= seq:
                del self._partials[channel]
        elif cur is None or cur[0] <= seq:
            self._partials[channel] = (seq, start, text)
        self._render_partials()

    def _render_partials(self) -> None:
        t = Text()
        for ch, (_seq, _start, text) in sorted(self._partials.items(), key=lambda kv: kv[1][1]):
            if t:
                t.append("\n")
            t.append(f"{SPEAKER_FOR_CHANNEL.get(ch, ch)} (live) · ", "dim bold")
            t.append(text + " …", "dim italic")
        self.query_one("#partials", Static).update(t)
