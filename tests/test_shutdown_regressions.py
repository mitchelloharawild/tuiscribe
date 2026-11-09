"""Shutdown and failure paths: stopping while loading or transcribing, failed loads, notes, source switches."""

import threading
import time

import pytest

import tuiscribe.pipeline as pl
from tuiscribe.capture.devices import Device
from tuiscribe.controller import Controller
from tuiscribe.events import AddNote, EventBus, FinalEvent, StatusEvent, SwitchSource
from tuiscribe.output import TranscriptWriter
from tuiscribe.records import Header, Note
from tuiscribe.stt import Transcript

from test_pipeline_offline import T0, FakeTranscriber, FakeVAD, burst, quiet


@pytest.fixture(autouse=True)
def fake_vad(monkeypatch):
    monkeypatch.setattr(pl, "StreamingVAD", FakeVAD)


def make(tmp_path, tr, channels=("mic", "system")):
    w = TranscriptWriter(tmp_path / "x.md")
    w.header(Header(T0, {}, "fake"))
    bus = EventBus()
    return pl.Pipeline(list(channels), tr, w, bus, clock=lambda: 1.0), w, bus


class SlowLoad(FakeTranscriber):
    def load(self):
        time.sleep(0.5)


def test_stop_waits_for_model_load_and_writes_speech(tmp_path):
    p, w, bus = make(tmp_path, SlowLoad())
    p.start()
    p.feed("mic", 0.0, burst(30))
    assert p.stop() is True
    assert w.segments == 1


def test_stop_timeout_reports_unfinished(tmp_path):
    p, w, bus = make(tmp_path, SlowLoad())
    p.start()
    p.feed("mic", 0.0, burst(30))
    assert p.stop(timeout=0.05) is False
    w.footer(T0, 1.0)  # writer closed while STT still runs
    time.sleep(0.8)
    errors = [e for e in bus.drain() if isinstance(e, StatusEvent) and e.severity == "error"]
    assert errors and "closed" in errors[0].message  # reported, thread did not crash


def test_notes_written_when_model_fails_to_load(tmp_path):
    class FailLoad(FakeTranscriber):
        def load(self):
            raise RuntimeError("no model")

    p, w, bus = make(tmp_path, FailLoad())
    p.start()
    p.feed("mic", 0.0, burst(30))  # speech is dropped, but must not hold the note back
    p.add_note("## Action items")
    p.stop()
    assert "## Action items" in (tmp_path / "x.md").read_text()
    assert any(isinstance(e, FinalEvent) and isinstance(e.item, Note) for e in bus.drain())


def test_vad_crash_does_not_hang_stop(tmp_path, monkeypatch):
    class BadVAD(FakeVAD):
        def process(self, samples):
            raise RuntimeError("onnx boom")

    monkeypatch.setattr(pl, "StreamingVAD", BadVAD)
    p = pl.Pipeline(["mic"], FakeTranscriber(), None, EventBus(), clock=lambda: 0.0, max_queue_chunks=10)
    p.start()
    for i in range(50):
        p.feed("mic", i * 0.02, quiet(1))
    t = threading.Thread(target=p.stop, daemon=True)
    t.start()
    t.join(3)
    assert not t.is_alive()


def test_control_messages_never_block_when_audio_queue_full(tmp_path):
    gate = threading.Event()
    p, w, bus = make(tmp_path, FakeTranscriber())
    # don't start: nothing consumes the queue
    for i in range(p.max_queue_chunks + 10):
        p.feed("mic", i * 0.032, quiet(1))
    assert p.dropped_chunks == 10
    t = threading.Thread(target=lambda: (p.flush(), p.add_note("x"), gate.set()), daemon=True)
    t.start()
    assert gate.wait(1)


def test_small_gap_resyncs_timeline(tmp_path):
    p, w, bus = make(tmp_path, FakeTranscriber(), channels=("mic",))
    p.start()
    p.feed("mic", 0.0, quiet(10))
    p.feed("mic", 10 * 512 / 16000 + 0.2, burst(30))  # 0.2 s jump: no cut, but resync
    p.stop()
    (seg,) = [e.item for e in bus.drain() if isinstance(e, FinalEvent)]
    assert seg.start == pytest.approx(10 * 512 / 16000 + 0.2, abs=0.01)


# --- controller -------------------------------------------------------------


def dev(i, kind):
    return Device(i, f"node{i}", f"dev {i}", kind, None, "running")


def test_controller_ignores_commands_after_stop(tmp_path):
    c = Controller({"mic": dev(1, "source")})
    c.stop()
    c.apply(SwitchSource("mic", dev(2, "source")))  # must not start a new pw-record
    c.apply(AddNote("late"))
    assert c.devices["mic"].id == 1
    assert c._sources == {}


def test_source_lost_flushes_channel(tmp_path):
    class P:
        flushed = []

        def flush(self, ch=None):
            self.flushed.append(ch)

    c = Controller({"mic": dev(1, "source"), "system": dev(2, "sink")})
    c.pipeline = P()
    c._on_source_error("mic", "pw-record exited (1)")
    assert P.flushed == ["mic"]
