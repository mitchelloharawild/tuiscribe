"""Diarization wired through Pipeline, transcribe_offline and Controller."""

import numpy as np
import pytest

import tuiscribe.pipeline as pl
from tuiscribe.controller import Controller
from tuiscribe.diarize import Diarizer
from tuiscribe.events import EventBus, FinalEvent, StatusEvent
from tuiscribe.capture.devices import Device
from tuiscribe.output import TranscriptWriter, sidecar_path
from tuiscribe.pipeline import Pipeline, transcribe_offline
from tuiscribe.records import Header, Segment, from_json
from tuiscribe.render import render_records

from test_diarize import A, B, AmpEmbedder, vec
from test_pipeline_offline import T0, FakeTranscriber, FakeVAD, burst, quiet


@pytest.fixture(autouse=True)
def fake_vad(monkeypatch):
    monkeypatch.setattr(pl, "StreamingVAD", FakeVAD)


def two_voices():
    """quiet, voice A (0.3), quiet, voice B (0.6), quiet, voice A again."""
    return np.concatenate(
        [quiet(20), burst(40, 0.3), quiet(40), burst(40, 0.6), quiet(40), burst(40, 0.3), quiet(40)]
    )


def run_live(tmp_path, diarizer, audio, channels=("mic", "system")):
    w = TranscriptWriter(tmp_path / "x.md")
    w.header(Header(T0, {}, "fake"))
    bus = EventBus()
    p = Pipeline(list(channels), FakeTranscriber(), w, bus, clock=lambda: 100.0, partials=False, diarizer=diarizer)
    p.start()
    for ch, a in audio.items():
        p.feed(ch, 0.0, a)
    assert p.stop(timeout=10) is True
    events = bus.drain()
    segs = [e.item for e in events if isinstance(e, FinalEvent) and isinstance(e.item, Segment)]
    return p, w, events, segs


def statuses(events, severity=None):
    return [e for e in events if isinstance(e, StatusEvent) and (severity is None or e.severity == severity)]


# --- live ---------------------------------------------------------------------------


def test_live_system_labelled_mic_not(tmp_path):
    d = Diarizer(["system"], embedder=AmpEmbedder())
    p, w, events, segs = run_live(tmp_path, d, {"system": two_voices(), "mic": two_voices()})
    w.close()
    system = [s for s in segs if s.channel == "system"]
    mic = [s for s in segs if s.channel == "mic"]
    assert [s.speaker for s in system] == ["Remote 1", "Remote 2", "Remote 1"]
    assert len(mic) == 3 and all(s.speaker is None for s in mic)
    assert any("speaker labels on" in e.message and "system" in e.message for e in statuses(events, "info"))
    # written files carry the labels
    written = [from_json(l) for l in sidecar_path(tmp_path / "x.md").read_text().splitlines()]
    assert [r.speaker for r in written if isinstance(r, Segment) and r.channel == "system"] == [
        "Remote 1", "Remote 2", "Remote 1"]
    assert "**Remote 2**" in (tmp_path / "x.md").read_text()


def test_live_both_channels(tmp_path):
    d = Diarizer(["mic", "system"], embedder=AmpEmbedder())
    p, w, events, segs = run_live(tmp_path, d, {"system": two_voices(), "mic": two_voices()})
    assert {s.speaker for s in segs if s.channel == "mic"} == {"Local 1", "Local 2"}
    assert {s.speaker for s in segs if s.channel == "system"} == {"Remote 1", "Remote 2"}


def test_final_speakers_after_stop(tmp_path):
    d = Diarizer(["system"], embedder=AmpEmbedder())
    p, w, events, segs = run_live(tmp_path, d, {"system": two_voices()})
    final = p.final_speakers()
    assert final == {("system", s.start): s.speaker for s in segs}
    assert list(final.values()) == ["Remote 1", "Remote 2", "Remote 1"]


def test_final_speakers_without_diarizer(tmp_path):
    p, w, events, segs = run_live(tmp_path, None, {"system": two_voices()})
    assert p.final_speakers() == {}
    assert all(s.speaker is None for s in segs)


def test_diarizer_load_failure_warns_and_continues(tmp_path):
    class BadLoad(AmpEmbedder):
        def load(self):
            raise RuntimeError("no onnx")

    d = Diarizer(["system"], embedder=BadLoad())
    p, w, events, segs = run_live(tmp_path, d, {"system": two_voices()})
    warns = statuses(events, "warning")
    assert any("speaker labels off" in e.message and "no onnx" in e.message for e in warns)
    assert not statuses(events, "error")
    assert len(segs) == 3 and all(s.speaker is None for s in segs)
    assert p.diarizer is None
    assert p.final_speakers() == {}


def test_embed_exception_warns_speaker_none(tmp_path):
    class BadEmbed(AmpEmbedder):
        def embed(self, audio):
            raise RuntimeError("bad audio")

    d = Diarizer(["system"], embedder=BadEmbed())
    p, w, events, segs = run_live(tmp_path, d, {"system": two_voices()})
    assert len(segs) == 3 and all(s.speaker is None for s in segs)
    warns = [e for e in statuses(events, "warning") if "speaker label failed" in e.message]
    assert 1 <= len(warns) <= 3  # rate-limited
    assert warns[0].channel == "system"
    assert not statuses(events, "error")


def test_embed_exception_once_then_recovers(tmp_path):
    class Flaky(AmpEmbedder):
        def embed(self, audio):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("blip")
            return super().embed(audio)

    d = Diarizer(["system"], embedder=Flaky())
    p, w, events, segs = run_live(tmp_path, d, {"system": two_voices()})
    assert [s.speaker for s in segs] == [None, "Remote 1", "Remote 2"] or [s.speaker for s in segs][0] is None
    assert all(s.speaker for s in segs[1:])


def test_embedder_none_result_keeps_going(tmp_path):
    d = Diarizer(["system"], embedder=AmpEmbedder({}))  # every embedding is None
    p, w, events, segs = run_live(tmp_path, d, {"system": two_voices()})
    assert [s.speaker for s in segs] == ["Remote 1"] * 3


# --- offline ------------------------------------------------------------------------


def test_offline_final_labels_in_segments_and_file(tmp_path):
    # Noisy first utterance: online splits it off, whole-session clustering merges it.
    voices = {0.2: vec(0), 0.3: vec(62), 0.4: vec(10), 0.5: vec(20)}
    audio = {
        "system": np.concatenate(
            [quiet(20)] + [np.concatenate([burst(40, a), quiet(40)]) for a in (0.2, 0.3, 0.4, 0.5)]
        )
    }
    d = Diarizer(["system"], embedder=AmpEmbedder(voices))
    md = tmp_path / "off.md"
    w = TranscriptWriter(md)
    w.header(Header(T0, {}, "fake"))
    segs = transcribe_offline(audio, FakeTranscriber(), w, diarizer=d)
    w.footer(T0, 1.0)
    assert [s.speaker for s in segs] == ["Remote 1"] * 4
    assert d._online["system"].n_speakers == 2  # live pass had split it
    recs = [from_json(l) for l in sidecar_path(md).read_text().splitlines()]
    assert [r.speaker for r in recs if isinstance(r, Segment)] == ["Remote 1"] * 4
    assert md.read_text() == render_records(recs)
    assert "Remote 2" not in md.read_text()


def test_offline_diarizer_only_for_applicable_channels(tmp_path):
    audio = {"system": two_voices(), "mic": two_voices()}
    d = Diarizer(["system"], embedder=AmpEmbedder())
    segs = transcribe_offline(audio, FakeTranscriber(), None, diarizer=d)
    assert {s.speaker for s in segs if s.channel == "mic"} == {None}
    assert [s.speaker for s in segs if s.channel == "system"] == ["Remote 1", "Remote 2", "Remote 1"]


def test_offline_without_diarizer_unchanged(tmp_path):
    segs = transcribe_offline({"system": two_voices()}, FakeTranscriber(), None)
    assert all(s.speaker is None for s in segs)


# --- Controller._relabel --------------------------------------------------------------


def dev(i, kind):
    return Device(i, f"node{i}", f"dev {i}", kind, None, "running")


class FakePipeline:
    def __init__(self, labels):
        self.labels = labels

    def final_speakers(self):
        return self.labels


def writer_with_two(tmp_path):
    w = TranscriptWriter(tmp_path / "c.md")
    w.header(Header(T0, {}, "fake"))
    w.item(Segment("system", 1.0, 2.0, "a", speaker="Remote 1"))
    w.item(Segment("system", 5.0, 6.0, "b", speaker="Remote 1"))
    return w


def test_controller_relabel_updates_and_publishes(tmp_path):
    c = Controller({"system": dev(1, "sink")})
    c.writer = writer_with_two(tmp_path)
    c.pipeline = FakePipeline({("system", 1.0): "Remote 1", ("system", 5.0): "Remote 2"})
    c._relabel()
    assert c.relabelled == 1
    ev = [e for e in c.bus.drain() if isinstance(e, StatusEvent)]
    assert len(ev) == 1 and ev[0].severity == "info" and "1 segments" in ev[0].message
    c.writer.footer(T0, 1.0)
    assert "**Remote 2**" in (tmp_path / "c.md").read_text()


def test_controller_relabel_no_change_is_silent(tmp_path):
    c = Controller({"system": dev(1, "sink")})
    c.writer = writer_with_two(tmp_path)
    c.pipeline = FakePipeline({})
    c._relabel()
    assert c.relabelled == 0
    assert c.bus.drain() == []


def test_controller_relabel_error_publishes_and_does_not_raise(tmp_path):
    class Boom:
        def relabel(self, labels):
            raise OSError("disk full")

    c = Controller({"system": dev(1, "sink")})
    c.writer = Boom()
    c.pipeline = FakePipeline({("system", 1.0): "X"})
    c._relabel()
    assert c.relabelled == 0
    ev = c.bus.drain()
    assert len(ev) == 1 and ev[0].severity == "error" and "disk full" in ev[0].message


def test_controller_relabel_without_pipeline_or_writer():
    c = Controller({"system": dev(1, "sink")})
    c._relabel()
    assert c.relabelled == 0 and c.bus.drain() == []


def test_controller_stop_relabels_before_footer(tmp_path):
    order = []

    class W:
        def relabel(self, labels):
            order.append("relabel")
            return 0

        def footer(self, ended, dur):
            order.append("footer")

        def close(self):
            order.append("close")

    class P:
        def stop(self):
            return True

        def final_speakers(self):
            return {}

    c = Controller({"system": dev(1, "sink")})
    c.pipeline, c.writer = P(), W()
    c.stop()
    assert order == ["relabel", "footer"]


def test_controller_stop_unfinished_skips_relabel(tmp_path):
    order = []

    class W:
        def relabel(self, labels):
            order.append("relabel")

        def footer(self, *a):
            order.append("footer")

        def close(self):
            order.append("close")

    class P:
        def stop(self):
            return False

    c = Controller({"system": dev(1, "sink")})
    c.pipeline, c.writer = P(), W()
    c.stop()
    assert order == ["close"]
