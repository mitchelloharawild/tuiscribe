import numpy as np
import pytest

from tuiscribe.audio.segmenter import Partial, Segmenter, SpeechDiscarded, SpeechStart, Utterance

W = 512
DUR = W / 16000  # 0.032 s


def win(v=0.0):
    return np.full(W, v, dtype=np.float32)


def run(seg, probs, t0=0.0, start_index=0):
    events = []
    for i, p in enumerate(probs):
        events += seg.push(t0 + (start_index + i) * DUR, win(), p)
    return events


def of(events, cls):
    return [e for e in events if isinstance(e, cls)]


def test_speech_start_at_threshold():
    seg = Segmenter("mic")
    assert seg.push(0.0, win(), 0.49) == []
    assert seg.open_start is None
    ev = seg.push(DUR, win(), 0.5)
    assert ev == [SpeechStart("mic", DUR, 1)]
    assert seg.open_start == DUR


def test_utterance_ends_after_min_silence():
    seg = Segmenter("mic")
    ev = run(seg, [0.9] * 20 + [0.0] * 30)
    utts = of(ev, Utterance)
    assert len(utts) == 1
    u = utts[0]
    assert u.channel == "mic" and u.seq == 1
    assert u.start == 0.0
    assert u.end == pytest.approx(20 * DUR)
    assert seg.open_start is None
    # nothing more once finished
    assert of(ev, SpeechStart) == [SpeechStart("mic", 0.0, 1)]


def test_not_ended_before_min_silence():
    seg = Segmenter("mic")
    ev = run(seg, [0.9] * 20 + [0.0] * 18)  # 18 * 32 ms = 0.576 < 0.6
    assert of(ev, Utterance) == []
    assert seg.open_start == 0.0


def test_hysteresis_keeps_speech_going():
    seg = Segmenter("mic")
    # in-between probs (>= neg_threshold, < threshold) are not silence
    ev = run(seg, [0.9] * 5 + [0.4] * 100)
    assert of(ev, Utterance) == []
    assert seg.open_start == 0.0
    # and in-between probs never *start* speech
    idle = Segmenter("mic")
    assert run(idle, [0.4] * 50) == []
    assert idle.open_start is None


def test_hysteresis_extends_end_time():
    seg = Segmenter("mic")
    ev = run(seg, [0.9] * 10 + [0.4] * 10 + [0.0] * 25)
    (u,) = of(ev, Utterance)
    assert u.end == pytest.approx(20 * DUR)


def test_pre_roll_included():
    seg = Segmenter("mic")
    ev = run(seg, [0.0] * 20 + [0.9] * 20 + [0.0] * 30)
    (u,) = of(ev, Utterance)
    assert u.start == pytest.approx(20 * DUR)
    span = u.end - u.start
    assert len(u.audio) / 16000 > span
    # 20 speech windows + ~0.3 s pre-roll + ~0.2 s tail (each +-1 window)
    assert len(u.audio) / 16000 == pytest.approx(20 * DUR + 0.3 + 0.2, abs=0.08)


def test_pre_roll_audio_content():
    seg = Segmenter("mic", pre_roll_s=0.3)
    seg.push(0.0, win(0.1), 0.0)
    seg.push(DUR, win(0.2), 0.0)
    for i in range(2, 22):
        seg.push(i * DUR, win(0.5), 0.9)
    (u,) = of(seg.flush(), Utterance)
    assert u.audio[0] == pytest.approx(0.1)
    assert u.audio[W] == pytest.approx(0.2)
    assert len(u.audio) == 22 * W


def test_tail_trimmed():
    seg = Segmenter("mic", tail_s=0.2)
    ev = run(seg, [0.9] * 20 + [0.0] * 30)
    (u,) = of(ev, Utterance)
    tail = len(u.audio) / 16000 - 20 * DUR
    # ~0.2 s of tail, not the whole 0.6 s of silence
    assert 0.15 <= tail <= 0.25


def test_too_short_discarded():
    seg = Segmenter("mic")
    ev = run(seg, [0.9] * 3 + [0.0] * 30)
    assert of(ev, Utterance) == []
    assert of(ev, SpeechDiscarded) == [SpeechDiscarded("mic", 0.0, 1)]
    assert seg.open_start is None


def test_seq_increments_between_utterances():
    seg = Segmenter("mic")
    ev = run(seg, ([0.9] * 20 + [0.0] * 30) * 2)
    assert [u.seq for u in of(ev, Utterance)] == [1, 2]
    assert [s.seq for s in of(ev, SpeechStart)] == [1, 2]


def test_partials_every_interval_with_growing_audio():
    seg = Segmenter("mic", partial_interval_s=2.0)
    ev = run(seg, [0.9] * 140 + [0.0] * 30)
    parts = of(ev, Partial)
    (u,) = of(ev, Utterance)
    assert len(parts) == 2
    assert all(p.seq == u.seq == 1 and p.start == 0.0 for p in parts)
    assert len(parts[0].audio) < len(parts[1].audio)
    assert len(parts[0].audio) / 16000 == pytest.approx(2.0, abs=0.05)
    assert len(parts[1].audio) / 16000 == pytest.approx(4.0, abs=0.07)
    assert len(parts[1].audio) < len(u.audio)


def test_no_partial_before_interval():
    seg = Segmenter("mic", partial_interval_s=2.0)
    assert of(run(seg, [0.9] * 50), Partial) == []


def test_long_speech_split():
    seg = Segmenter("mic", max_utterance_s=15.0)
    probs = [0.9] * 600
    probs[400] = 0.4  # quietest window, in the last third
    ev = run(seg, probs)
    utts = of(ev, Utterance)
    starts = of(ev, SpeechStart)
    assert len(utts) >= 1
    head = utts[0]
    assert head.seq == 1 and head.start == 0.0
    assert starts[1].seq == 2
    assert head.end == starts[1].start  # contiguous
    assert head.end == pytest.approx(400 * DUR)  # cut at the lowest-prob window
    assert seg.open_start is not None


def test_split_ignores_dip_outside_last_third():
    seg = Segmenter("mic", max_utterance_s=15.0)
    probs = [0.9] * 500
    probs[100] = 0.4  # early dip, not eligible
    ev = run(seg, probs)
    head = of(ev, Utterance)[0]
    assert head.end > 100 * DUR + 1.0
    assert head.end > 300 * DUR


def test_split_continues_with_later_utterances_contiguous():
    seg = Segmenter("mic", max_utterance_s=5.0)
    ev = run(seg, [0.9] * 400 + [0.0] * 30)
    utts = of(ev, Utterance)
    assert len(utts) >= 3
    for a, b in zip(utts, utts[1:]):
        assert a.end == b.start
        assert b.seq == a.seq + 1


def test_flush_closes_open_utterance():
    seg = Segmenter("mic")
    run(seg, [0.9] * 20)
    ev = seg.flush()
    assert len(ev) == 1 and isinstance(ev[0], Utterance)
    assert ev[0].start == 0.0 and ev[0].end == pytest.approx(20 * DUR)
    assert seg.open_start is None
    assert seg.flush() == []


def test_flush_idle_returns_empty():
    seg = Segmenter("mic")
    assert seg.flush() == []
    run(seg, [0.0] * 5)
    assert seg.flush() == []


def test_flush_short_speech_discards():
    seg = Segmenter("mic")
    run(seg, [0.9] * 2)
    ev = seg.flush()
    assert ev == [SpeechDiscarded("mic", 0.0, 1)]


def test_flush_clears_pre_roll():
    seg = Segmenter("mic")
    seg.push(0.0, win(0.1), 0.0)
    seg.flush()
    for i in range(1, 21):
        seg.push(i * DUR, win(0.5), 0.9)
    (u,) = of(seg.flush(), Utterance)
    assert len(u.audio) == 20 * W  # no pre-roll carried over
