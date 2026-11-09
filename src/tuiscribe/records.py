"""Transcript records: what goes into the `.jsonl` sidecar and gets rendered.

Times are session offsets in seconds; `Header.started_at` anchors them to wall
time. The sidecar is the source of truth; the Markdown is derived from it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

SPEAKER_FOR_CHANNEL = {"mic": "You", "system": "Remote"}


@dataclass(frozen=True)
class Header:
    started_at: datetime
    sources: dict[str, str]  # channel -> device description
    model: str
    type: str = field(default="header", init=False)


@dataclass(frozen=True)
class Segment:
    channel: str
    start: float
    end: float
    text: str
    speaker: str | None = None  # None -> derived from channel
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    type: str = field(default="segment", init=False)

    @property
    def label(self) -> str:
        return self.speaker or SPEAKER_FOR_CHANNEL.get(self.channel, self.channel)


@dataclass(frozen=True)
class Note:
    """Manual note or heading inserted from the UI; `text` is raw Markdown."""

    at: float
    text: str
    type: str = field(default="note", init=False)

    @property
    def start(self) -> float:
        return self.at


@dataclass(frozen=True)
class Footer:
    ended_at: datetime
    duration: float
    segments: int
    type: str = field(default="footer", init=False)


Record = Header | Segment | Note | Footer
Item = Segment | Note  # things ordered on the timeline


def to_json(rec: Record) -> str:
    d = asdict(rec)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return json.dumps(d, ensure_ascii=False)


def from_json(line: str) -> Record:
    d: dict[str, Any] = json.loads(line)
    kind = d.pop("type")
    if kind == "header":
        return Header(started_at=datetime.fromisoformat(d["started_at"]), sources=d["sources"], model=d["model"])
    if kind == "segment":
        return Segment(**d)
    if kind == "note":
        return Note(**d)
    if kind == "footer":
        return Footer(ended_at=datetime.fromisoformat(d["ended_at"]), duration=d["duration"], segments=d["segments"])
    raise ValueError(f"unknown record type {kind!r}")
