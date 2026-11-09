import threading
import time
from datetime import datetime, timezone

import numpy as np
import pytest

import tuiscribe.pipeline as pl
from tuiscribe.events import EventBus, FinalEvent, PartialEvent
from tuiscribe.output import TranscriptWriter, sidecar_path
from tuiscribe.pipeline import Pipeline, transcribe_offline
from tuiscribe.records import Header, Note, Segment, from_json
from tuiscribe.stt import Transcript

W = 512
SR = 16000
T0 = datetime(2024, 3, 5, 12, 0, 0, tzinfo=timezone.utc)


class FakeVAD:
    def __init__(self):
        self.reset()

    def reset(self):
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, samples):
        buf = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32)])
        n = len(buf) // W
        self._pending = buf[n * W :]
        out = []
        for i in range(n):
            w = buf[i * W : (i + 1) * W]
            out.append((w, 0.9 if float(np.abs(w).mean()) > 0.05 else 0.0))
        return out


class FakeTranscriber:
    name = "fake"

    def __init__(self, gate=None):
        self.loaded = False
        self.calls = []
        self.gate = gate

    def load(self):
        self.loaded = True

    def transcribe(self, audio, prompt=None, final=True):
        self.calls.append((len(audio), prompt, final))
        if final and self.gate is not None:
            self.gate.wait(10)
        return Transcript(f"utt {len(audio)}")


@pytest.fixture(autouse=True)
def fake_vad(monkeypatch):
    monkeypatch.setattr(pl, "StreamingVAD", FakeVAD)


def burst(windows, amp=0.3):
    x = np.full(windows * W, amp, dtype=np.float32)
    x[1::2] *= -1
    return x


def quiet(windows):
    return np.zeros(windows * W, dtype=np.float32)


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


# --- offline ----------------------------------------------------------------


def test_offline_orders_and_writes(tmp_path):
    audio = {
        "system": np.concatenate([quiet(60), burst(40), quiet(40)]),
        "mic": np.concatenate([quiet(16), burst(32), quiet(40), burst(30), quiet(40)]),
    }
    md = tmp_path / "off.md"
    writer = TranscriptWriter(md)
    writer.header(Header(T0, {}, "fake"))
    tr = FakeTranscriber()
    progress = []
    segs = transcribe_offline(audio, tr, writer, progress=lambda i, n, s: progress.append((i, n, s)))
    writer.close()

    starts = [s.start for s in segs]
    assert starts == sorted(starts)
    assert len(segs) == 3
    assert segs[0].start == pytest.approx(16 * W / SR)
    assert all(s.text.startswith("utt ") for s in segs)
    assert [p[0] for p in progress] == [1, 2, 3] and all(p[1] == 3 for p in progress)
    written = [from_json(l) for l in sidecar_path(md).read_text().splitlines()]
    assert [r for r in written if isinstance(r, Segment)] == segs
    # prompt: previous text of the same channel
    mic_prompts = [p for (_n, p, _f) in tr.calls]
    assert mic_prompts[0] is None
    assert any(p and p.startswith("utt ") for p in mic_prompts[1:])


def test_offline_skips_empty_text_and_no_writer():
    class Empty(FakeTranscriber):
        def transcribe(self, audio, prompt=None, final=True):
            return Transcript("")

    audio = {"mic": np.concatenate([quiet(10), burst(30), quiet(40)])}
    assert transcribe_offline(audio, Empty(), None) == []


def test_offline_silence_only():
    assert transcribe_offline({"mic": quiet(50)}, FakeTranscriber(), None) == []


# --- live -------------------------------------------------------------------


def make_pipeline(tmp_path, transcriber, clock_box):
    bus = EventBus()
    md = tmp_path / "live.md"
    writer = TranscriptWriter(md)
    writer.header(Header(T0, {}, "fake"))
    pipe = Pipeline(["mic", "system"], transcriber, writer, bus, clock=lambda: clock_box[0])
    return pipe, bus, writer, md


def items_from(bus_events):
    return [e.item for e in bus_events if isinstance(e, FinalEvent)]


def test_live_ordering_across_channels_partials_cleared_and_note(tmp_path):
    gate = threading.Event()
    tr = FakeTranscriber(gate)
    clock = [0.0]
    pipe, bus, writer, md = make_pipeline(tmp_path, tr, clock)

    # system speaks windows 32..100 (long); mic speaks windows 50..70 and finishes first
    sysaud = np.concatenate([quiet(32), burst(68), quiet(50)])
    micaud = np.concatenate([quiet(50), burst(20), quiet(80)])
    assert len(sysaud) == len(micaud) == 150 * W
    pipe.start()
    step = 320
    for i in range(0, len(sysaud), step):
        t = i / SR
        pipe.feed("mic", t, micaud[i : i + step])
        pipe.feed("system", t, sysaud[i : i + step])
    end_t = len(sysaud) / SR
    assert wait_for(lambda: pipe._buf_t["system"] is not None and abs(pipe._buf_t["system"] - end_t) < 1e-6)
    assert wait_for(lambda: pipe._buf_t["mic"] is not None and abs(pipe._buf_t["mic"] - end_t) < 1e-6)

    # mic final is queued first; STT is blocked on the gate. Add a note between the two starts.
    clock[0] = 1.3
    pipe.add_note("## Mid note")
    gate.set()
    pipe.stop()

    assert tr.loaded
    events = bus.drain()
    items = items_from(events)
    kinds = [(type(i).__name__, getattr(i, "channel", None)) for i in items]
    assert kinds == [("Segment", "system"), ("Note", None), ("Segment", "mic")]
    starts = [i.start for i in items]
    assert starts == sorted(starts)
    assert starts[0] == pytest.approx(32 * W / SR)
    assert items[1] == Note(at=1.3, text="## Mid note")

    # file order == event order
    recs = [from_json(l) for l in sidecar_path(md).read_text().splitlines()]
    assert [r for r in recs if isinstance(r, (Segment, Note))] == items
    assert writer.segments == 2

    # empty PartialEvents clear the live line for each utterance
    clears = {(e.channel, e.seq) for e in events if isinstance(e, PartialEvent) and e.text == ""}
    assert {("mic", 1), ("system", 1)} <= clears
    assert pipe.model_state == "ready"


def test_live_discontinuity_flushes_open_utterance(tmp_path):
    tr = FakeTranscriber()
    pipe, bus, writer, md = make_pipeline(tmp_path, tr, [0.0])
    pipe.start()
    pipe.feed("mic", 0.0, burst(40))  # speech, never goes silent
    pipe.feed("mic", 0.1, quiet(1))  # expected t is 1.28: a jump back > 0.25 s
    assert wait_for(lambda: writer.segments == 1), "discontinuity should have flushed the utterance"
    pipe.stop()
    (seg,) = items_from(bus.drain())
    assert seg.start == 0.0
    assert seg.end == pytest.approx(40 * W / SR)


def test_live_small_gap_is_not_discontinuity(tmp_path):
    tr = FakeTranscriber()
    pipe, bus, writer, md = make_pipeline(tmp_path, tr, [0.0])
    pipe.start()
    pipe.feed("mic", 0.0, burst(40))
    pipe.feed("mic", 40 * W / SR + 0.2, burst(5))  # 0.2 s jump < 0.25: same utterance
    assert not wait_for(lambda: writer.segments > 0, timeout=0.5)
    pipe.stop()
    segs = items_from(bus.drain())
    assert len(segs) == 1
    assert segs[0].start == 0.0


def test_live_stop_flushes_open_speech(tmp_path):
    pipe, bus, writer, md = make_pipeline(tmp_path, FakeTranscriber(), [0.0])
    pipe.start()
    pipe.feed("system", 0.0, burst(30))
    pipe.stop()
    (seg,) = items_from(bus.drain())
    assert seg.channel == "system" and seg.text.startswith("utt ")
