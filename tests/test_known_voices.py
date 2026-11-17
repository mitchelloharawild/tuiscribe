"""Naming speakers from saved voices: Diarizer, `tuiscribe diarize`, `tuiscribe voices`."""

import numpy as np
from typer.testing import CliRunner

from tuiscribe.cli import app
from tuiscribe.diarize import Diarizer
from tuiscribe.diarize.voices import VoiceBank
from tuiscribe.output import sidecar_path
from tuiscribe.records import Segment, from_json

from test_diarize import A, B, C, AmpEmbedder, tone
from test_diarize_toggle import fake_model, recorded  # noqa: F401 - fixture


def speakers(md):
    return [r.speaker for r in map(from_json, sidecar_path(md).read_text().splitlines()) if isinstance(r, Segment)]


def bank_with(**people):
    bank = VoiceBank()
    for name, voice in people.items():
        bank.add(name, [(voice, 0.0, 5.0), (voice, 10.0, 15.0)], "mic", "earlier.md")
    return bank


def test_final_labels_name_known_voices_and_renumber_the_rest():
    d = Diarizer(["mic"], embedder=AmpEmbedder(), voices=bank_with(Bea=B))
    for i, amp in enumerate([0.3, 0.6, 0.3]):
        d.label(("mic", float(i)), tone(amp, 32000), 2.0)
    labels = d.final_labels()
    assert [labels[("mic", float(i))] for i in range(3)] == ["Local 1", "Bea", "Local 1"]
    assert set(d.matched["mic"]) == {"Bea"} and d.matched["mic"]["Bea"] > 0.99
    assert {k: len(v) for k, v in d.groups["mic"].items()} == {"Local 1": 2, "Bea": 1}


def test_final_labels_without_voices_are_unchanged():
    d = Diarizer(["mic"], embedder=AmpEmbedder(), voices=VoiceBank())  # empty bank
    for i, amp in enumerate([0.6, 0.3]):
        d.label(("mic", float(i)), tone(amp, 32000), 2.0)
    assert list(d.final_labels().values()) == ["Local 1", "Local 2"]


def test_cli_voices_add_then_diarize_names_them(tmp_path, fake_model):
    md = recorded(tmp_path)
    run = CliRunner().invoke
    assert run(app, ["diarize", str(md), "-d", "mic"]).exit_code == 0
    assert speakers(md) == ["Local 1", "Local 2", "Local 1"]

    r = run(app, ["voices", "add", "Bea", str(md), "--label", "Local 2"])
    assert r.exit_code == 0, r.output
    assert "added 1 samples" in r.output and "relabelled 'Local 2' as 'Bea'" in r.output
    assert speakers(md) == ["Local 1", "Bea", "Local 1"]
    v = VoiceBank().get("Bea")
    assert len(v) == 1 and v.source[0] == str(md.resolve()) and np.allclose(v.emb[0], B)

    # a fresh labelling names her from the saved voice
    out = tmp_path / "again.md"
    r = run(app, ["diarize", str(md), "-d", "mic", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert "known: Bea (1.00)" in r.output
    assert speakers(out) == ["Local 1", "Bea", "Local 1"]
    r = run(app, ["diarize", str(md), "-d", "mic", "-o", str(out), "--no-voices"])
    assert speakers(out) == ["Local 1", "Local 2", "Local 1"]

    r = run(app, ["voices", "list"])
    assert "Bea: 1 samples from 1 transcript (MIC 0.0 min)" in r.output
    assert run(app, ["voices", "remove", "Bea"]).exit_code == 0
    assert VoiceBank().get("Bea") is None
    assert run(app, ["voices", "remove", "Bea"]).exit_code == 1


def test_cli_voices_add_label_defaults_to_name_and_is_idempotent(tmp_path, fake_model):
    md = recorded(tmp_path)
    CliRunner().invoke(app, ["diarize", str(md), "-d", "mic"])
    CliRunner().invoke(app, ["voices", "add", "Al", str(md), "-l", "Local 1"])
    r = CliRunner().invoke(app, ["voices", "add", "Al", str(md)])  # now labelled Al
    assert r.exit_code == 0 and "added 0 samples" in r.output and "2 saved in all" in r.output


def test_cli_voices_add_unknown_label(tmp_path, fake_model):
    md = recorded(tmp_path)
    r = CliRunner().invoke(app, ["voices", "add", "Bea", str(md)])
    assert r.exit_code == 1 and "no segments labelled 'Bea'" in r.output


def test_cli_voices_add_keeps_hand_edits(tmp_path, fake_model):
    md = recorded(tmp_path)
    CliRunner().invoke(app, ["diarize", str(md), "-d", "mic"])
    md.write_text(md.read_text() + "my edits\n")
    r = CliRunner().invoke(app, ["voices", "add", "Bea", str(md), "-l", "Local 2"])
    assert r.exit_code == 0 and "not relabelled" in r.output
    assert "my edits" in md.read_text() and speakers(md)[1] == "Local 2"
    assert VoiceBank().get("Bea") is not None


def test_cli_voices_list_empty():
    r = CliRunner().invoke(app, ["voices", "list"])
    assert r.exit_code == 0 and "no saved voices" in r.output


# --- live --------------------------------------------------------------------------


def test_live_names_known_voices_from_the_first_utterance():
    d = Diarizer(["mic"], embedder=AmpEmbedder(), voices=bank_with(Bea=B))
    got = [d.label(("mic", float(i)), tone(amp), dur) for i, (amp, dur) in enumerate(
        [(0.6, 3.0), (0.3, 3.0), (0.6, 0.5), (0.3, 3.0)]
    )]
    assert got == ["Bea", "Local 1", "Bea", "Local 1"]
    assert d.speakers("mic") == 2


def test_live_unknown_voice_is_not_taken_for_a_known_one():
    from test_diarize import A_NEAR

    # A_NEAR is 0.87 similar to A: joins Al. C is opposite: a new speaker.
    d = Diarizer(["mic"], embedder=AmpEmbedder({0.3: A_NEAR, 0.6: C}), voices=bank_with(Al=A), threshold=0.3)
    got = [d.label(("mic", float(i)), tone(amp), 3.0) for i, amp in enumerate([0.6, 0.3, 0.6])]
    assert got == ["Local 1", "Al", "Local 1"]
    assert d.speakers("mic") == 2


def test_live_without_voices_numbers_from_one():
    d = Diarizer(["mic"], embedder=AmpEmbedder())
    assert d.label(("mic", 0.0), tone(0.6), 0.5) == "Local 1"  # too short to found a speaker
    assert d.speakers("mic") == 0


def test_live_known_voice_needs_seed_threshold_even_when_speaker_threshold_is_low():
    from test_diarize import vec

    d = Diarizer(["mic"], embedder=AmpEmbedder({0.3: vec(70)}), voices=bank_with(Al=A), threshold=0.3)
    assert d.label(("mic", 0.0), tone(0.3), 3.0) == "Local 1"  # cos 70 = 0.34: not Al


# --- naming by hand ----------------------------------------------------------------


def labelled(d, amps, dur=3.0):
    return [d.label(("mic", float(i)), tone(amp), dur) for i, amp in enumerate(amps)]


def test_name_speaker_live_then_final_and_learn():
    bank = VoiceBank()
    d = Diarizer(["mic"], embedder=AmpEmbedder(), voices=bank)
    assert labelled(d, [0.3, 0.6]) == ["Local 1", "Local 2"]
    sp = {s.label: s for s in d.live_speakers()}
    assert set(sp) == {"Local 1", "Local 2"} and sp["Local 1"].seconds == 3.0 and not sp["Local 1"].confirmed
    d.name_speaker("mic", sp["Local 2"].index, "Bea")
    assert d.label(("mic", 2.0), tone(0.6), 3.0) == "Bea"
    assert [s.label for s in d.live_speakers() if s.confirmed] == ["Bea"]
    labels = d.final_labels()
    assert [labels[("mic", float(i))] for i in range(3)] == ["Local 1", "Bea", "Bea"]
    assert d.learn("talk.md") == {"Bea": 2}  # only confirmed names are learned
    assert bank.names() == ["Bea"] and sorted(bank.get("Bea").start) == [1.0, 2.0]
    assert d.learn("talk.md") == {"Bea": 0}


def test_confirming_a_known_voice_adds_to_it():
    bank = bank_with(Bea=B)
    d = Diarizer(["mic"], embedder=AmpEmbedder(), voices=bank)
    assert labelled(d, [0.6]) == ["Bea"]
    (sp,) = d.live_speakers()
    assert sp.suggestion[0] == "Bea" and sp.suggestion[1] > 0.99
    d.name_speaker("mic", sp.index, "Bea")
    d.final_labels()
    assert d.learn("new.md") == {"Bea": 1} and len(bank.get("Bea")) == 3


def test_unnaming_and_automatic_matches_are_not_learned():
    bank = bank_with(Bea=B)
    d = Diarizer(["mic"], embedder=AmpEmbedder(), voices=bank)
    labelled(d, [0.6, 0.3])
    local = next(s for s in d.live_speakers() if s.label == "Local 1")
    d.name_speaker("mic", local.index, "Al")
    d.name_speaker("mic", local.index, None)
    assert d.live_speakers()[1].label == "Local 1"
    d.final_labels()
    assert d.learn("x.md") == {} and len(bank.get("Bea")) == 2


def test_hand_name_beats_a_saved_voice_for_the_same_group():
    bank = bank_with(Bea=B)
    d = Diarizer(["mic"], embedder=AmpEmbedder(), voices=bank)
    labelled(d, [0.6, 0.6])
    d.name_speaker("mic", 0, "Bianca")  # "Bea" was wrong
    labels = d.final_labels()
    assert set(labels.values()) == {"Bianca"}
    assert d.learn("x.md") == {"Bianca": 2}


def test_learn_without_voices_does_nothing():
    d = Diarizer(["mic"], embedder=AmpEmbedder())
    labelled(d, [0.3])
    d.name_speaker("mic", 0, "Al")
    assert d.final_labels() == {("mic", 0.0): "Al"}
    assert d.learn("x.md") == {}


def test_controller_names_speaker_and_saves_voice_at_stop(tmp_path):
    from tuiscribe.events import NameSpeaker, SetDiarize, StatusEvent
    from test_diarize_toggle import SPANS, session, wait_for

    c = session(tmp_path)
    c.diarizer.voices = VoiceBank()
    c.apply(SetDiarize("mic", True))
    assert wait_for(lambda: c.diarizer.ready)
    for a, b, amp in SPANS:
        c.writer.item(Segment("mic", a, b, "x", speaker=c.diarizer.label(("mic", a), tone(amp), b - a)))
    c.bus.drain()
    c.apply(NameSpeaker("mic", 1, "Bea"))
    msgs = [e.message for e in c.bus.drain() if isinstance(e, StatusEvent)]
    assert msgs == ["Local 2 is Bea from now; when you stop, earlier lines are relabelled and Bea's voice is saved"]
    c._relabel()
    assert [s.speaker for s in c.writer.session_segments()] == ["Local 1", "Bea", "Local 1"]
    assert c.learned == {"Bea": 1}
    assert any("voice samples saved: Bea +1" in e.message for e in c.bus.drain() if isinstance(e, StatusEvent))
    v = VoiceBank().get("Bea")
    assert v.source[0] == str((tmp_path / "m.md").resolve())
