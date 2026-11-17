"""Switching speaker labels on mid-session, labelling from saved audio, `tuiscribe diarize`."""

import time
import wave

import numpy as np
import pytest
from typer.testing import CliRunner

import tuiscribe.pipeline as pl
from tuiscribe.audio.wav import AlignedWavWriter, WavSpans
from tuiscribe.capture.devices import Device
from tuiscribe.cli import app
from tuiscribe.config import SAMPLE_RATE
from tuiscribe.controller import Controller
from tuiscribe.diarize import Diarizer
from tuiscribe.events import EventBus, SetDiarize, StatusEvent
from tuiscribe.output import TranscriptWriter, sidecar_path
from tuiscribe.pipeline import Pipeline
from tuiscribe.records import Header, Note, Segment, from_json, to_json
from tuiscribe.render import render_records

from test_diarize import AmpEmbedder, tone
from test_pipeline_offline import T0, FakeTranscriber, FakeVAD, burst, quiet

# mic audio: voice A (0.3) at 0-2 s, voice B (0.6) at 3-5 s, voice A at 6-8 s
SPANS = [(0.0, 2.0, 0.3), (3.0, 5.0, 0.6), (6.0, 8.0, 0.3)]


def mic_audio():
    x = np.zeros(9 * SAMPLE_RATE, dtype=np.float32)
    for a, b, amp in SPANS:
        x[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)] = amp
    return x


def save_wav(path, audio):
    w = AlignedWavWriter(path, 0.0)
    w.write(0.0, audio)
    w.close()


def segs(channel="mic"):
    return [Segment(channel, a, b, f"s{i}") for i, (a, b, _) in enumerate(SPANS)]


def dev(i, kind):
    return Device(i, f"node{i}", f"dev {i}", kind, None, "running")


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# --- Diarizer -----------------------------------------------------------------------


def test_set_enabled_toggles_and_keeps_voiceprints():
    d = Diarizer(embedder=AmpEmbedder())
    assert not d.channels and not d.applies("mic")
    d.set_enabled("mic", True)
    assert d.applies("mic")
    assert d.label(("mic", 0.0), tone(0.3), 2.0) == "Local 1"
    d.set_enabled("mic", False)
    assert not d.applies("mic") and d.final_labels() == {}
    d.set_enabled("mic", True)
    assert d.label(("mic", 5.0), tone(0.6), 2.0) == "Local 2"  # same clusterer as before
    assert d.label(("mic", 9.0), tone(0.3), 2.0) == "Local 1"


def test_load_is_idempotent():
    emb = AmpEmbedder()
    d = Diarizer(embedder=emb)
    calls = []
    emb.load = lambda: calls.append(1)
    d.load()
    d.load()
    assert d.ready and calls == [1]


def test_final_labels_backfills_from_audio():
    audio = mic_audio()
    d = Diarizer(["mic"], embedder=AmpEmbedder())
    d.label(("mic", 6.0), tone(0.3), 2.0)  # only the last one was heard live
    s = segs()
    assert d.missing(s) == 2
    labels = d.final_labels(s, lambda ch, a, b: audio[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)])
    assert labels == {("mic", 0.0): "Local 1", ("mic", 3.0): "Local 2", ("mic", 6.0): "Local 1"}


def test_final_labels_without_audio_leaves_unheard_out():
    d = Diarizer(["mic"], embedder=AmpEmbedder())
    d.label(("mic", 6.0), tone(0.3), 2.0)
    assert d.final_labels(segs(), lambda *a: None) == {("mic", 6.0): "Local 1"}
    assert d.final_labels(segs()) == {("mic", 6.0): "Local 1"}


def test_final_labels_ignores_channels_not_diarized():
    d = Diarizer(["system"], embedder=AmpEmbedder())
    assert d.final_labels(segs("mic"), lambda *a: tone(0.3)) == {}
    assert d.missing(segs("mic")) == 0


# --- WavSpans -----------------------------------------------------------------------


def test_wav_spans_reads_by_session_time(tmp_path):
    save_wav(tmp_path / "a.mic.wav", mic_audio())
    spans = WavSpans({"mic": tmp_path / "a.mic.wav", "system": tmp_path / "missing.wav"})
    assert spans.channels == {"mic"}
    clip = spans("mic", 3.0, 5.0)
    assert len(clip) == 2 * SAMPLE_RATE and round(float(clip.max()), 2) == 0.6
    assert len(spans("mic", 8.5, 20.0)) == SAMPLE_RATE // 2  # clipped to the file
    assert spans("mic", 20.0, 21.0) is None
    assert spans("system", 0.0, 1.0) is None
    spans.close()


def test_wav_spans_skips_other_formats(tmp_path):
    p = tmp_path / "stereo.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\0" * 400)
    assert WavSpans({"mic": p}).channels == set()


# --- Pipeline -----------------------------------------------------------------------


def test_pipeline_skips_live_labels_until_model_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "StreamingVAD", FakeVAD)
    d = Diarizer(embedder=AmpEmbedder())  # nothing on at start: pipeline does not load it
    w = TranscriptWriter(tmp_path / "x.md")
    w.header(Header(T0, {}, "fake"))
    p = Pipeline(["mic"], FakeTranscriber(), w, EventBus(), clock=lambda: 100.0, partials=False, diarizer=d)
    p.start()
    assert wait_for(lambda: p.model_state == "ready")
    d.set_enabled("mic", True)  # switched on later; its model load has not finished
    p.feed("mic", 0.0, np.concatenate([quiet(20), burst(40), quiet(40)]))
    assert p.stop(timeout=10)
    assert not d.ready
    assert [s.speaker for s in w.session_segments()] == [None]


# --- Controller ---------------------------------------------------------------------


class FakePipeline:
    def final_speakers(self):
        raise AssertionError("controller with a diarizer labels the session itself")


def session(tmp_path, save_audio=True):
    path = tmp_path / "m.mic.wav"
    save_wav(path, mic_audio())
    c = Controller(
        {"mic": dev(1, "source")},
        bus=EventBus(),
        audio_paths={"mic": path} if save_audio else None,
        diarizer=Diarizer(embedder=AmpEmbedder()),
    )
    c.pipeline = FakePipeline()
    c.writer = TranscriptWriter(tmp_path / "m.md")
    c.writer.header(Header(T0, {"mic": "dev 1"}, "fake"))
    return c


def infos(c):
    return [e.message for e in c.bus.drain() if isinstance(e, StatusEvent)]


def test_switch_on_mid_session_labels_whole_session(tmp_path):
    c = session(tmp_path)
    for s in segs():  # all said before speaker labels were switched on
        c.writer.item(s)
    c.apply(SetDiarize("mic"))
    assert c.diarizer.applies("mic")
    assert wait_for(lambda: c.diarizer.ready)
    assert any("earlier MIC speech is labelled when you stop" in m for m in infos(c))
    c._relabel()
    assert c.relabelled == 3
    assert any("labelling speakers in 3 earlier segments" in m for m in infos(c))
    c.writer.footer(T0, 9.0)
    recs = [from_json(l) for l in sidecar_path(tmp_path / "m.md").read_text().splitlines()]
    assert [r.speaker for r in recs if isinstance(r, Segment)] == ["Local 1", "Local 2", "Local 1"]
    assert "**Local 2**" in (tmp_path / "m.md").read_text()


def test_switch_off_before_stop_removes_labels(tmp_path):
    c = session(tmp_path)
    c.apply(SetDiarize("mic", True))
    assert wait_for(lambda: c.diarizer.ready)
    for (a, b, amp), s in zip(SPANS, segs()):
        c.writer.item(Segment("mic", a, b, s.text, speaker=c.diarizer.label(("mic", a), tone(amp), b - a)))
    c.apply(SetDiarize("mic"))  # toggle off
    assert not c.diarizer.applies("mic")
    assert any("speaker labels off" in m for m in infos(c))
    c._relabel()
    assert c.relabelled == 3
    assert all(s.speaker is None for s in c.writer.session_segments())


def test_switch_on_without_saved_audio_labels_from_then_on(tmp_path):
    c = session(tmp_path, save_audio=False)
    c.writer.item(segs()[0])
    c.apply(SetDiarize("mic", True))
    assert wait_for(lambda: c.diarizer.ready)
    assert any("audio not saved" in m for m in infos(c))
    for (a, b, amp), s in list(zip(SPANS, segs()))[1:]:
        c.writer.item(Segment("mic", a, b, s.text, speaker=c.diarizer.label(("mic", a), tone(amp), b - a)))
    c._relabel()
    assert [s.speaker for s in c.writer.session_segments()] == [None, "Local 1", "Local 2"]


def test_switch_on_load_failure_switches_off(tmp_path):
    c = session(tmp_path)

    def boom():
        raise RuntimeError("no onnx")

    c.diarizer.embedder.load = boom
    c.apply(SetDiarize("mic", True))
    assert wait_for(lambda: not c.diarizer.applies("mic"))
    assert wait_for(lambda: any("no onnx" in e.message for e in c.bus.drain() if isinstance(e, StatusEvent)))


def test_set_diarize_needs_transcription():
    c = Controller({"mic": dev(1, "source")}, bus=EventBus())
    c.apply(SetDiarize("mic"))
    ev = c.bus.drain()
    assert len(ev) == 1 and ev[0].severity == "warning"


def test_set_diarize_same_state_is_silent(tmp_path):
    c = session(tmp_path)
    c.apply(SetDiarize("mic", False))
    assert c.bus.drain() == []


# --- tuiscribe diarize --------------------------------------------------------------


@pytest.fixture
def fake_model(monkeypatch):
    import tuiscribe.diarize as dz

    monkeypatch.setattr(dz, "SpeakerEmbedder", lambda name: AmpEmbedder())


def recorded(tmp_path, extra=()):
    md = tmp_path / "talk.md"
    records = [Header(T0, {"mic": "dev"}, "fake"), *segs(), Note(8.5, "## end"), *extra]
    sidecar_path(md).write_text("".join(to_json(r) + "\n" for r in records))
    md.write_text(render_records(records))
    save_wav(tmp_path / "talk.mic.wav", mic_audio())
    return md


def test_cli_diarize_in_place(tmp_path, fake_model):
    md = recorded(tmp_path)
    r = CliRunner().invoke(app, ["diarize", str(md), "--diarize", "mic"])
    assert r.exit_code == 0, r.output
    assert "MIC: 2 speakers" in r.output
    recs = [from_json(l) for l in sidecar_path(md).read_text().splitlines()]
    assert [x.speaker for x in recs if isinstance(x, Segment)] == ["Local 1", "Local 2", "Local 1"]
    assert any(isinstance(x, Note) for x in recs)
    assert md.read_text() == render_records(recs)


def test_cli_diarize_refuses_hand_edited_md_without_output(tmp_path, fake_model):
    md = recorded(tmp_path)
    md.write_text(md.read_text() + "my edits\n")
    r = CliRunner().invoke(app, ["diarize", str(md), "-d", "mic"])
    assert r.exit_code == 1 and "differs" in r.output
    out = tmp_path / "labelled.md"
    r = CliRunner().invoke(app, ["diarize", str(md), "-d", "mic", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert "**Local 2**" in out.read_text() and sidecar_path(out).exists()
    assert "my edits" in md.read_text()  # original untouched


def test_cli_diarize_explicit_audio_and_missing_audio(tmp_path, fake_model):
    md = recorded(tmp_path)
    (tmp_path / "talk.mic.wav").rename(tmp_path / "other.wav")
    r = CliRunner().invoke(app, ["diarize", str(md), "-d", "mic"])
    assert r.exit_code == 1 and "no audio for mic" in r.output
    r = CliRunner().invoke(app, ["diarize", str(md), "-d", "mic", "--audio", f"mic={tmp_path / 'other.wav'}"])
    assert r.exit_code == 0, r.output


def test_cli_diarize_multi_session_refused(tmp_path, fake_model):
    md = recorded(tmp_path, extra=[Header(T0, {}, "fake")])
    r = CliRunner().invoke(app, ["diarize", str(md), "-d", "mic"])
    assert r.exit_code == 1 and "2 sessions" in r.output
