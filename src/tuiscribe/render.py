"""Records -> Markdown. Pure functions: no I/O, no clock.

The file is append-only, so each rendered item is the text to *append* after
the previous one. Paragraphs are left open (no trailing newline) so a
continuing speaker can extend the same paragraph; the next item decides
whether to continue, break the paragraph, or start a new speaker block.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .records import Footer, Header, Item, Note, Record, Segment


@dataclass(frozen=True)
class RenderOptions:
    gap_short: float = 2.0  # pause that starts a new paragraph (same speaker)
    gap_long: float = 10.0  # pause that gets a silence marker
    rule_after: float = 60.0  # pause that also gets a horizontal rule


def fmt_duration(s: float) -> str:
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m} min {sec} s" if sec else f"{m} min"
    h, rem = divmod(s, 3600)
    return f"{h} h {rem // 60} min"


def fmt_clock(started_at: datetime, offset: float) -> str:
    return (started_at + timedelta(seconds=offset)).strftime("%H:%M:%S")


def render_header(h: Header) -> str:
    sources = ", ".join(f'{ch} = "{desc}"' for ch, desc in h.sources.items())
    return (
        f"# Meeting: {h.started_at.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"**Sources:** {sources} · **Model:** {h.model}"
    )


def _label(seg: Segment, started_at: datetime, overlapping: bool = False) -> str:
    extra = " (overlapping)" if overlapping else ""
    return f"\n\n**{seg.label}** · {fmt_clock(started_at, seg.start)}{extra}\n{seg.text}"


def _item_end(item: Item) -> float:
    return item.end if isinstance(item, Segment) else item.at


def render_item(prev: Item | None, item: Item, started_at: datetime, opts: RenderOptions = RenderOptions()) -> str:
    if isinstance(item, Note):
        return f"\n\n{item.text.strip()}"

    seg = item
    if prev is None or isinstance(prev, Note):
        return _label(seg, started_at)

    gap = seg.start - _item_end(prev)
    if gap > opts.gap_long:
        marker = "\n\n---" if gap > opts.rule_after else ""
        return f"{marker}\n\n*⋯ {fmt_duration(gap)} of silence ⋯*" + _label(seg, started_at)
    if seg.label != prev.label:
        return _label(seg, started_at, overlapping=seg.start < prev.end and seg.channel != prev.channel)
    if gap >= opts.gap_short:
        return f"\n\n{seg.text}"
    return f" {seg.text}"


def render_footer(f: Footer) -> str:
    return (
        f"\n\n---\n\n*Ended {f.ended_at.strftime('%H:%M:%S')} · "
        f"duration {fmt_duration(f.duration)} · {f.segments} segments*\n"
    )


def render_records(records: list[Record], opts: RenderOptions = RenderOptions()) -> str:
    """Render a full sidecar (one or more sessions) to Markdown."""
    out: list[str] = []
    started_at: datetime | None = None
    prev: Item | None = None
    for rec in records:
        if isinstance(rec, Header):
            if out:
                out.append("\n\n")
            out.append(render_header(rec))
            started_at, prev = rec.started_at, None
        elif isinstance(rec, Footer):
            out.append(render_footer(rec))
        else:
            if started_at is None:
                raise ValueError("sidecar has items before a header")
            out.append(render_item(prev, rec, started_at, opts))
            prev = rec
    text = "".join(out)
    return text if text.endswith("\n") else text + "\n"
