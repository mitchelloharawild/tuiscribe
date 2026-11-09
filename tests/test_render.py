from datetime import datetime, timezone

import pytest

from tuiscribe.records import Footer, Header, Note, Segment, from_json, to_json
from tuiscribe.render import RenderOptions, fmt_clock, fmt_duration, render_item, render_records

T0 = datetime(2024, 3, 5, 12, 0, 0, tzinfo=timezone.utc)


def seg(start, end, text="hi", channel="mic", **kw):
    return Segment(channel, start, end, text, **kw)


def test_first_segment_has_label_and_clock():
    out = render_item(None, seg(5, 7, "Hello"), T0)
    assert out == "\n\n**You** · 12:00:05\nHello"


def test_remote_label_and_custom_speaker():
    assert "**Remote** · 12:01:05" in render_item(None, seg(65, 66, channel="system"), T0)
    assert "**Alice** ·" in render_item(None, seg(1, 2, speaker="Alice"), T0)


def test_fmt_clock():
    assert fmt_clock(T0, 3725) == "13:02:05"


def test_same_speaker_short_gap_continues_paragraph():
    prev = seg(0, 2)
    assert render_item(prev, seg(3, 4, "world"), T0) == " world"


def test_gap_between_short_and_long_new_paragraph_no_label():
    prev = seg(0, 2)
    out = render_item(prev, seg(6, 7, "world"), T0)  # gap 4 s
    assert out == "\n\nworld"
    assert "**" not in out


def test_gap_exactly_short_starts_paragraph():
    assert render_item(seg(0, 2), seg(4, 5, "x"), T0) == "\n\nx"


def test_long_gap_silence_marker_and_fresh_label():
    out = render_item(seg(0, 2), seg(32, 33, "again"), T0)  # gap 30 s
    assert "*⋯ 30 s of silence ⋯*" in out
    assert "---" not in out
    assert out.endswith("**You** · 12:00:32\nagain")


def test_rule_after_adds_horizontal_rule():
    out = render_item(seg(0, 2), seg(92, 93, "later"), T0)  # gap 90 s
    assert out.startswith("\n\n---\n\n*⋯ 1 min 30 s of silence ⋯*")
    assert "**You** · 12:01:32" in out


def test_custom_options():
    opts = RenderOptions(gap_short=0.5, gap_long=100, rule_after=200)
    assert render_item(seg(0, 1), seg(2, 3, "x"), T0, opts) == "\n\nx"
    assert render_item(seg(0, 1), seg(50, 51, "x"), T0, opts) == "\n\nx"


def test_speaker_change_gets_label():
    out = render_item(seg(0, 2), seg(3, 4, "yo", channel="system"), T0)
    assert out == "\n\n**Remote** · 12:00:03\nyo"


def test_overlapping_different_channel():
    prev = seg(5, 10)
    out = render_item(prev, seg(8, 11, "talking over", channel="system"), T0)
    assert "**Remote** · 12:00:08 (overlapping)" in out


def test_not_overlapping_when_after_prev_end():
    out = render_item(seg(5, 10), seg(11, 12, channel="system"), T0)
    assert "(overlapping)" not in out


def test_note_renders_raw_text_and_next_segment_labelled():
    note = Note(at=4, text="  ## Action items  ")
    assert render_item(seg(0, 2), note, T0) == "\n\n## Action items"
    out = render_item(note, seg(4.5, 5, "ok"), T0)  # same speaker would otherwise continue
    assert out == "\n\n**You** · 12:00:04\nok"


def test_note_first_item():
    assert render_item(None, Note(0, "# Hi"), T0) == "\n\n# Hi"


@pytest.mark.parametrize(
    "s,expected",
    [
        (0, "0 s"),
        (5, "5 s"),
        (59.4, "59 s"),
        (60, "1 min"),
        (75, "1 min 15 s"),
        (600, "10 min"),
        (3599, "59 min 59 s"),
        (3600, "1 h 0 min"),
        (7380, "2 h 3 min"),
    ],
)
def test_fmt_duration(s, expected):
    assert fmt_duration(s) == expected


def test_render_records_two_sessions_and_footer():
    t1 = datetime(2024, 3, 5, 9, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 3, 5, 15, 30, tzinfo=timezone.utc)
    recs = [
        Header(t1, {"mic": "Mic", "system": "Sys"}, "small.en"),
        seg(1, 2, "one"),
        Footer(datetime(2024, 3, 5, 9, 5, tzinfo=timezone.utc), 300.0, 1),
        Header(t2, {"mic": "Mic"}, "base"),
        seg(1, 2, "two"),
        Footer(datetime(2024, 3, 5, 15, 31, tzinfo=timezone.utc), 61.0, 1),
    ]
    out = render_records(recs)
    assert out.count("# Meeting:") == 2
    assert "# Meeting: 2024-03-05 09:00" in out
    assert "# Meeting: 2024-03-05 15:30" in out
    assert '**Sources:** mic = "Mic", system = "Sys" · **Model:** small.en' in out
    assert "**You** · 09:00:01\none" in out
    assert "**You** · 15:30:01\ntwo" in out  # clock anchored to the second header
    assert "*Ended 09:05:00 · duration 5 min · 1 segments*" in out
    assert "duration 1 min 1 s" in out
    assert out.endswith("\n")
    # first segment of session 2 gets a label (no stale prev)
    assert out.index("# Meeting: 2024-03-05 15:30") < out.index("two")


def test_render_records_adds_trailing_newline():
    out = render_records([Header(T0, {}, "m"), seg(1, 2)])
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_render_records_item_before_header_raises():
    with pytest.raises(ValueError):
        render_records([seg(1, 2)])


@pytest.mark.parametrize(
    "rec",
    [
        Header(T0, {"mic": "Built-in", "system": "Monitor"}, "faster-whisper small.en"),
        Header(datetime(2024, 1, 1, 8, 0, 0, 123456, tzinfo=timezone.utc), {}, "m"),
        Segment("mic", 1.5, 2.25, "héllo ✓"),
        Segment("system", 0.0, 1.0, "x", speaker="Bob", no_speech_prob=0.01, avg_logprob=-0.3),
        Note(3.5, "## Heading\n\n- item"),
        Footer(T0, 12.5, 7),
    ],
)
def test_json_round_trip(rec):
    assert from_json(to_json(rec)) == rec
    assert "\n" not in to_json(rec)


def test_json_round_trip_keeps_timezone():
    h = Header(datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc), {}, "m")
    assert from_json(to_json(h)).started_at.tzinfo is not None


def test_from_json_unknown_type():
    with pytest.raises(ValueError):
        from_json('{"type": "bogus"}')
