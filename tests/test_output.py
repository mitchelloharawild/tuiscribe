from datetime import datetime, timezone

from tuiscribe.output import TranscriptWriter, sidecar_path
from tuiscribe.records import Footer, Header, Note, Segment, from_json
from tuiscribe.render import render_records

T0 = datetime(2024, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2024, 3, 5, 14, 0, 0, tzinfo=timezone.utc)


def header(t=T0):
    return Header(t, {"mic": "Mic", "system": "Sys"}, "fake")


def session(w, t=T0):
    w.header(header(t))
    w.item(Segment("mic", 1.0, 2.0, "hello"))
    w.item(Segment("mic", 2.5, 3.0, "there"))
    w.item(Segment("system", 4.0, 5.0, "hi"))
    w.item(Note(6.0, "## Notes"))
    w.item(Segment("mic", 7.0, 8.0, "after note"))
    w.footer(t, 10.0)


def parsed(path):
    return [from_json(line) for line in sidecar_path(path).read_text().splitlines()]


def test_writes_md_and_jsonl(tmp_path):
    md = tmp_path / "sub" / "meeting.md"
    w = TranscriptWriter(md)
    session(w)
    assert md.exists() and md.with_suffix(".jsonl").exists()
    recs = parsed(md)
    assert isinstance(recs[0], Header) and isinstance(recs[-1], Footer)
    assert [type(r).__name__ for r in recs] == ["Header", "Segment", "Segment", "Segment", "Note", "Segment", "Footer"]
    text = md.read_text()
    assert "# Meeting: 2024-03-05 12:00" in text
    assert "**You** · 12:00:01\nhello there" in text


def test_md_equals_render_of_jsonl(tmp_path):
    md = tmp_path / "m.md"
    w = TranscriptWriter(md)
    session(w)
    assert md.read_text() == render_records(parsed(md))


def test_item_returns_appended_markdown(tmp_path):
    w = TranscriptWriter(tmp_path / "m.md")
    w.header(header())
    out = w.item(Segment("mic", 1.0, 2.0, "hello"))
    assert out == "\n\n**You** · 12:00:01\nhello"
    assert w.segments == 1
    w.item(Note(3.0, "x"))
    assert w.segments == 1


def test_append_second_session(tmp_path):
    md = tmp_path / "m.md"
    w1 = TranscriptWriter(md)
    session(w1)
    first = md.read_text()
    w2 = TranscriptWriter(md)
    session(w2, T1)
    text = md.read_text()
    assert text.startswith(first)
    assert text.count("# Meeting:") == 2
    assert "# Meeting: 2024-03-05 14:00" in text
    recs = parsed(md)
    assert sum(isinstance(r, Header) for r in recs) == 2
    assert text == render_records(recs)
    # the new header is separated from the previous session by a blank line
    assert "\n\n# Meeting: 2024-03-05 14:00" in text
    assert w2.segments == 4


def test_footer_closes_and_second_footer_noop(tmp_path):
    md = tmp_path / "m.md"
    w = TranscriptWriter(md)
    w.header(header())
    w.item(Segment("mic", 1.0, 2.0, "hello"))
    w.footer(T0, 5.0)
    assert w.closed
    before_md, before_js = md.read_text(), md.with_suffix(".jsonl").read_text()
    footers = [r for r in parsed(md) if isinstance(r, Footer)]
    assert len(footers) == 1 and footers[0].segments == 1 and footers[0].duration == 5.0
    w.footer(T0, 9.0)  # no-op, must not raise on closed files
    assert md.read_text() == before_md
    assert md.with_suffix(".jsonl").read_text() == before_js
    w.close()  # idempotent


def test_close_without_footer(tmp_path):
    w = TranscriptWriter(tmp_path / "m.md")
    w.header(header())
    w.close()
    assert w.closed
    assert len(parsed(tmp_path / "m.md")) == 1
