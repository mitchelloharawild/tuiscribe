"""Events (pipeline -> UI) and commands (UI -> controller), plus a thread-safe bus.

Worker threads never touch the UI. They publish events here and the UI drains
the bus on its own loop. Level meters do not go through the queue: the UI
polls `Controller.poll_levels()` at a fixed rate (see capture/levels.py), so a
slow UI can never back up the audio path.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Literal

from .capture.devices import Device
from .records import Item

# --- events -----------------------------------------------------------------


@dataclass(frozen=True)
class LevelEvent:
    channel: str
    rms_db: float
    peak_db: float
    clipped: bool
    digital_silence: bool  # exact zeros: muted hardware or wrong device


SourceState = Literal["starting", "running", "lost", "stopped"]


@dataclass(frozen=True)
class SourceEvent:
    channel: str
    device: Device
    state: SourceState
    detail: str = ""


@dataclass(frozen=True)
class StatusEvent:
    severity: Literal["info", "warning", "error"]
    message: str
    channel: str | None = None


@dataclass(frozen=True)
class PartialEvent:
    """In-progress text for an utterance; empty text clears the live line."""

    channel: str
    seq: int
    start: float
    text: str


@dataclass(frozen=True)
class FinalEvent:
    item: Item  # Segment or Note, in file order
    markdown: str  # exactly what was appended to the .md file


Event = SourceEvent | StatusEvent | PartialEvent | FinalEvent


# --- commands ---------------------------------------------------------------


@dataclass(frozen=True)
class SetMute:
    channel: str
    muted: bool | None = None  # None toggles


@dataclass(frozen=True)
class SwitchSource:
    channel: str
    device: Device


@dataclass(frozen=True)
class SetPause:
    paused: bool | None = None  # None toggles


@dataclass(frozen=True)
class AddNote:
    text: str  # raw Markdown, e.g. "## Action items"


@dataclass(frozen=True)
class Quit:
    pass


Command = SetMute | SwitchSource | SetPause | AddNote | Quit


# --- bus --------------------------------------------------------------------


class EventBus:
    """Bounded, non-blocking. Publishers never wait; if the consumer stalls,
    new events are dropped and counted rather than blocking a capture thread."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._q: queue.Queue[Event] = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def publish(self, event: Event) -> None:
        try:
            self._q.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def drain(self, limit: int = 500) -> list[Event]:
        out: list[Event] = []
        for _ in range(limit):
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out
