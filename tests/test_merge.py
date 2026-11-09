from tuiscribe.merge import Merger
from tuiscribe.records import Note, Segment


def s(start, text="x", channel="mic"):
    return Segment(channel, start, start + 1, text)


def test_release_in_start_order():
    m = Merger()
    for st in (5.0, 1.0, 3.0):
        m.add(s(st))
    assert [i.start for i in m.release(10)] == [1.0, 3.0, 5.0]
    assert len(m) == 0


def test_holds_items_after_watermark():
    m = Merger()
    for st in (1.0, 3.0, 5.0):
        m.add(s(st))
    assert [i.start for i in m.release(3.0)] == [1.0, 3.0]  # inclusive
    assert len(m) == 1
    assert m.release(4.0) == []
    assert [i.start for i in m.release(5.0)] == [5.0]


def test_release_all_drains():
    m = Merger()
    m.add(s(9.0))
    m.add(Note(2.0, "n"))
    out = m.release_all()
    assert [i.start for i in out] == [2.0, 9.0]
    assert isinstance(out[0], Note)
    assert m.release_all() == []


def test_stable_for_equal_starts():
    m = Merger()
    items = [s(1.0, "a"), s(1.0, "b", "system"), Note(1.0, "c"), s(1.0, "d")]
    for i in items:
        m.add(i)
    assert m.release_all() == items


def test_empty():
    m = Merger()
    assert m.release(100) == []
    assert len(m) == 0
