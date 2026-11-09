"""Append-only Markdown + JSONL sidecar writer, flushed per item.

`meeting.md` and `meeting.jsonl` sit side by side. If the Markdown file
already has content, the new session is appended after it.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

from .records import Footer, Header, Item, Segment, to_json
from .render import RenderOptions, render_footer, render_header, render_item


def sidecar_path(md: Path) -> Path:
    return md.with_suffix(".jsonl")


class TranscriptWriter:
    def __init__(self, md_path: Path, opts: RenderOptions = RenderOptions()) -> None:
        self.md_path = md_path
        self.jsonl_path = sidecar_path(md_path)
        self.opts = opts
        self.segments = 0
        self._prev: Item | None = None
        self._started_at: datetime | None = None
        self._lock = threading.Lock()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        existing = md_path.exists() and md_path.stat().st_size > 0
        self._md = md_path.open("a", encoding="utf-8")
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._needs_sep = existing
        self.closed = False

    def _write(self, md: str, rec) -> None:
        self._md.write(md)
        self._md.flush()
        self._jsonl.write(to_json(rec) + "\n")
        self._jsonl.flush()

    def header(self, h: Header) -> None:
        with self._lock:
            self._started_at = h.started_at
            sep = "\n\n" if self._needs_sep else ""
            self._write(sep + render_header(h), h)

    def item(self, item: Item) -> str:
        """Append one item; returns the Markdown that was appended."""
        with self._lock:
            assert self._started_at is not None, "header() first"
            if self.closed:
                raise ValueError(f"{self.md_path} is already closed (footer written)")
            md = render_item(self._prev, item, self._started_at, self.opts)
            self._write(md, item)
            self._prev = item
            if isinstance(item, Segment):
                self.segments += 1
            return md

    def footer(self, ended_at: datetime, duration: float) -> None:
        with self._lock:
            if self.closed:
                return
            f = Footer(ended_at=ended_at, duration=duration, segments=self.segments)
            self._write(render_footer(f), f)
            self._close()

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self.closed:
            return
        for fh in (self._md, self._jsonl):
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
        self.closed = True
