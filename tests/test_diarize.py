"""Speaker diarization: clustering, Diarizer, embedder basics, writer relabel, CLI parsing."""

import json
from datetime import timedelta

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from tuiscribe.cli import _diarize_channels, app
from tuiscribe.diarize import DIARIZED_LABEL, Diarizer, speaker_label
from tuiscribe.diarize import embed as embed_mod
from tuiscribe.diarize.cluster import OnlineClusterer, _ahc, cluster_all
from tuiscribe.diarize.embed import MODELS, SpeakerEmbedder, cache_dir
from tuiscribe.output import TranscriptWriter, sidecar_path
from tuiscribe.records import Footer, Header, Note, Segment, from_json
from tuiscribe.render import render_records

from test_pipeline_offline import T0


def vec(deg):
    """Unit vector at an angle; cosine similarity between two is cos(angle difference)."""
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)], dtype=np.float32)


A, B, C = vec(0), vec(90), vec(180)
A_NEAR = vec(30)  # cos 30 = 0.87
A_FAR = vec(70)  # cos 70 = 0.34 (below 0.5)


class AmpEmbedder:
    """Fake embedder: voice is chosen by the audio's peak amplitude."""

    def __init__(self, voices=None):
        self.voices = {0.3: A, 0.6: B} if voices is None else voices
        self.loaded = False
        self.calls = 0

    def load(self):
        self.loaded = True

    def embed(self, audio):
        self.calls += 1
        if len(audio) == 0:
            return None
        amp = round(float(np.abs(audio).max()), 2)
        return self.voices.get(amp)


def tone(amp, n=16000):
    return np.full(n, amp, dtype=np.float32)


# --- OnlineClusterer ------------------------------------------------------------


def test_online_two_voices_and_repeat():
    c = OnlineClusterer()
    assert c.assign(A, 2.0) == 0
    assert c.assign(B, 2.0) == 1
    assert c.assign(A, 2.0) == 0
    assert c.assign(B, 2.0) == 1
    assert c.n_speakers == 2


def test_online_similar_above_threshold_joins():
    c = OnlineClusterer(threshold=0.5)
    c.assign(A, 2.0)
    assert c.assign(A_NEAR, 2.0) == 0
    assert c.n_speakers == 1


def test_online_below_threshold_starts_new_speaker():
    c = OnlineClusterer(threshold=0.5)
    c.assign(A, 2.0)
    assert c.assign(A_FAR, 2.0) == 1
    assert c.n_speakers == 2


def test_online_threshold_is_configurable():
    c = OnlineClusterer(threshold=0.95)
    c.assign(A, 2.0)
    assert c.assign(A_NEAR, 2.0) == 1


def test_online_short_utterance_keeps_previous_label():
    c = OnlineClusterer()
    c.assign(A, 2.0)
    c.assign(B, 2.0)  # last = 1
    # Short, unlike anyone: no new speaker, previous label.
    assert c.assign(C, 0.5) == 1
    assert c.n_speakers == 2


def test_online_short_utterance_joins_if_clearly_similar():
    c = OnlineClusterer()
    c.assign(A, 2.0)
    c.assign(B, 2.0)  # last = 1
    assert c.assign(A_NEAR, 0.5) == 0
    assert c.n_speakers == 2


def test_online_short_does_not_move_centroid():
    c = OnlineClusterer()
    c.assign(A, 2.0)
    before = c._sums[0].copy()
    c.assign(A_NEAR, 0.5)
    np.testing.assert_array_equal(c._sums[0], before)


def test_online_none_embedding_never_creates_speaker():
    c = OnlineClusterer()
    c.assign(A, 2.0)
    c.assign(B, 2.0)
    assert c.assign(None, 2.0) == 1
    assert c.assign(A, 2.0) == 0
    assert c.assign(None, 5.0) == 0
    assert c.n_speakers == 2


def test_online_first_utterance_none():
    c = OnlineClusterer()
    assert c.assign(None, 3.0) == 0
    assert c.n_speakers == 0
    assert c.assign(A, 2.0) == 0  # the first real voice is speaker 0
    assert c.assign(B, 2.0) == 1


def test_online_first_utterance_reliable():
    c = OnlineClusterer()
    assert c.assign(A, 2.0) == 0
    assert c.n_speakers == 1


def test_online_short_first_utterance_does_not_create_speaker():
    c = OnlineClusterer()
    assert c.assign(A, 0.4) == 0  # still labelled speaker 0 ...
    assert c.n_speakers == 0  # ... but founds no voiceprint
    assert c.assign(A, 2.0) == 0
    assert c.n_speakers == 1


def test_online_no_embedding_first_then_short():
    c = OnlineClusterer()
    c.assign(None, 3.0)
    c.assign(None, 3.0)
    assert c.n_speakers == 0


# --- cluster_all ----------------------------------------------------------------


def test_cluster_all_two_groups_any_order():
    embs = [A, B, A, B, B, A]
    assert cluster_all(embs, [2.0] * 6) == [0, 1, 0, 1, 1, 0]
    embs = [B, B, A, A, B]
    assert cluster_all(embs, [2.0] * 5) == [0, 0, 1, 1, 0]


def test_cluster_all_three_groups_numbered_by_first_appearance():
    embs = [C, A, B, A, C, B]
    assert cluster_all(embs, [2.0] * 6) == [0, 1, 2, 1, 0, 2]


def test_cluster_all_merges_early_live_mistake():
    # Noisy first utterance of speaker X (vec 60) looks unlike the later ones
    # (vec 0, 10, 20): online splits it off, AHC puts them all together.
    x = [vec(0), vec(62), vec(10), vec(20), vec(5)]
    # cos(62deg)=0.47 < 0.5 -> online starts a new speaker for it
    live = OnlineClusterer()
    online = [live.assign(e, 2.0) for e in x]
    assert online == [0, 1, 0, 0, 0]
    # average linkage: 62 deg is close enough (mean sim ~0.65) to the others.
    assert cluster_all(x, [2.0] * 5) == [0, 0, 0, 0, 0]


def test_cluster_all_short_attached_to_nearest_cluster():
    embs = [A, B, A_NEAR, A_NEAR]
    durs = [2.0, 2.0, 0.4, 0.4]
    assert cluster_all(embs, durs) == [0, 1, 0, 0]
    # nearest is B
    assert cluster_all([A, B, vec(80)], [2.0, 2.0, 0.4]) == [0, 1, 1]


def test_cluster_all_short_unlike_everyone_takes_previous_label():
    embs = [A, B, C]
    assert cluster_all(embs, [2.0, 2.0, 0.4]) == [0, 1, 1]
    embs = [B, A, vec(-90)]
    assert cluster_all(embs, [2.0, 2.0, 0.4]) == [0, 1, 1]


def test_cluster_all_none_takes_previous_label():
    assert cluster_all([A, B, None, A, None], [2.0] * 5) == [0, 1, 1, 0, 0]


def test_cluster_all_leading_none_takes_next_label():
    assert cluster_all([None, None, B, A], [2.0] * 4) == [0, 0, 0, 1]
    # leading short one with no cluster to join falls forward
    assert cluster_all([A, A], [0.2, 0.2]) == [0, 0]


def test_cluster_all_all_none_and_empty_and_single():
    assert cluster_all([None, None, None], [1.0, 2.0, 3.0]) == [0, 0, 0]
    assert cluster_all([], []) == []
    assert cluster_all([A], [2.0]) == [0]
    assert cluster_all([None], [2.0]) == [0]
    assert cluster_all([A], [0.1]) == [0]


def test_cluster_all_all_short_uses_previous_labels():
    # no reliable utterance -> nothing to cluster; all share label 0
    assert cluster_all([A, B, C], [0.5, 0.5, 0.5]) == [0, 0, 0]


def test_cluster_all_threshold():
    embs = [A, A_NEAR]
    assert cluster_all(embs, [2.0, 2.0], threshold=0.5) == [0, 0]
    assert cluster_all(embs, [2.0, 2.0], threshold=0.95) == [0, 1]


def test_ahc_single_and_by_first_appearance():
    assert _ahc(np.stack([A]), 0.5) == [0]
    assert _ahc(np.stack([B, A, B]), 0.5) == [0, 1, 0]


# --- Diarizer -------------------------------------------------------------------


def test_speaker_label_names():
    assert speaker_label("system", 0) == "Remote 1"
    assert speaker_label("mic", 1) == "Local 2"
    assert speaker_label("other", 2) == "other 3"
    assert DIARIZED_LABEL == {"mic": "Local", "system": "Remote"}


def test_diarizer_labels_per_channel():
    emb = AmpEmbedder()
    d = Diarizer(["system", "mic"], embedder=emb)
    assert d.applies("system") and d.applies("mic") and not d.applies("other")
    assert d.label(("system", 1.0), tone(0.3), 2.0) == "Remote 1"
    assert d.label(("system", 5.0), tone(0.6), 2.0) == "Remote 2"
    assert d.label(("system", 9.0), tone(0.3), 2.0) == "Remote 1"
    assert d.label(("mic", 2.0), tone(0.6), 2.0) == "Local 1"  # separate clusterer per channel
    assert d.speakers("system") == 2
    assert d.speakers("mic") == 1
    assert d.speakers("nope") == 0


def test_diarizer_applies_only_configured_channels():
    d = Diarizer(["system"], embedder=AmpEmbedder())
    assert d.applies("system") and not d.applies("mic")


def test_diarizer_load_delegates():
    emb = AmpEmbedder()
    Diarizer(["system"], embedder=emb).load()
    assert emb.loaded


def test_diarizer_final_labels_keyed_by_channel_start():
    d = Diarizer(["system", "mic"], embedder=AmpEmbedder())
    d.label(("system", 1.0), tone(0.3), 2.0)
    d.label(("mic", 1.0), tone(0.6), 2.0)  # same start, different channel
    d.label(("system", 4.0), tone(0.6), 2.0)
    d.label(("system", 8.0), tone(0.3), 2.0)
    assert d.final_labels() == {
        ("system", 1.0): "Remote 1",
        ("mic", 1.0): "Local 1",
        ("system", 4.0): "Remote 2",
        ("system", 8.0): "Remote 1",
    }


def test_diarizer_final_labels_empty():
    assert Diarizer(["system"], embedder=AmpEmbedder()).final_labels() == {}


def test_diarizer_final_labels_correct_live_mistake():
    voices = {0.1: vec(0), 0.2: vec(62), 0.3: vec(10), 0.4: vec(20), 0.5: vec(5)}
    d = Diarizer(["system"], embedder=AmpEmbedder(voices))
    live = [d.label(("system", float(i)), tone(a), 2.0) for i, a in enumerate(voices)]
    assert live == ["Remote 1", "Remote 2", "Remote 1", "Remote 1", "Remote 1"]
    assert set(d.final_labels().values()) == {"Remote 1"}


def test_diarizer_none_embedding_falls_back():
    d = Diarizer(["system"], embedder=AmpEmbedder())
    assert d.label(("system", 1.0), tone(0.3), 2.0) == "Remote 1"
    assert d.label(("system", 4.0), tone(0.99), 2.0) == "Remote 1"  # unknown amp -> None
    assert d.final_labels()[("system", 4.0)] == "Remote 1"


# --- SpeakerEmbedder (pure parts only) -------------------------------------------


def test_embed_short_audio_returns_none_without_loading(monkeypatch):
    e = SpeakerEmbedder()
    monkeypatch.setattr(e, "load", lambda: pytest.fail("must not load"))
    assert e.embed(np.zeros(int(0.29 * 16000), dtype=np.float32)) is None
    assert e.embed(np.zeros(0, dtype=np.float32)) is None


def test_embed_threshold_is_exactly_min_audio(monkeypatch):
    e = SpeakerEmbedder()
    called = []
    monkeypatch.setattr(e, "load", lambda: called.append(1) or (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        e.embed(np.zeros(int(0.3 * 16000), dtype=np.float32))
    assert called == [1]


def test_models_table():
    assert embed_mod.DEFAULT_MODEL in MODELS
    for m in MODELS.values():
        assert m.file.endswith(".onnx")
        assert m.window in ("povey", "hamming")
        assert m.scale > 0
    assert MODELS["wespeaker-resnet34"].scale == 32768.0
    assert SpeakerEmbedder("campplus").model is MODELS["campplus"]


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        SpeakerEmbedder("nope")


def test_cache_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache_dir() == tmp_path / "tuiscribe" / "models"


def test_cache_dir_default(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache_dir() == tmp_path / ".cache" / "tuiscribe" / "models"
    monkeypatch.setenv("XDG_CACHE_HOME", "")  # empty counts as unset
    assert cache_dir() == tmp_path / ".cache" / "tuiscribe" / "models"


# --- TranscriptWriter.relabel -----------------------------------------------------


def seg(ch, start, end, text, speaker=None):
    return Segment(ch, start, end, text, speaker=speaker)


def read_records(md):
    return [from_json(l) for l in sidecar_path(md).read_text().splitlines()]


def first_session(md):
    w = TranscriptWriter(md)
    w.header(Header(T0, {"mic": "m"}, "fake"))
    w.item(seg("mic", 1.0, 2.0, "earlier one", "Local 1"))
    w.item(seg("mic", 2.5, 3.5, "earlier two", "Local 1"))
    w.footer(T0 + timedelta(seconds=10), 10.0)
    return md.read_bytes(), sidecar_path(md).read_bytes()


def second_session(md, labels=("Remote 1", "Remote 1", "Remote 2")):
    w = TranscriptWriter(md)
    w.header(Header(T0 + timedelta(hours=1), {"system": "s"}, "fake"))
    w.item(seg("system", 1.0, 2.0, "hello", labels[0]))
    w.item(seg("system", 2.5, 3.5, "again", labels[1]))
    w.item(Note(4.0, "## Note"))
    w.item(seg("system", 5.0, 6.0, "other", labels[2]))
    return w


def test_relabel_rewrites_only_current_session(tmp_path):
    md = tmp_path / "m.md"
    md0, js0 = first_session(md)
    w = second_session(md)
    n = w.relabel({("system", 2.5): "Remote 2", ("system", 5.0): "Remote 1"})
    assert n == 2
    w.close()
    assert md.read_bytes().startswith(md0)
    assert sidecar_path(md).read_bytes().startswith(js0)
    recs = read_records(md)
    segs = [r for r in recs if isinstance(r, Segment)]
    assert [s.speaker for s in segs] == ["Local 1", "Local 1", "Remote 1", "Remote 2", "Remote 1"]
    assert "earlier one" in md.read_text()


def test_relabel_md_equals_render_of_jsonl(tmp_path):
    md = tmp_path / "m.md"
    first_session(md)
    w = second_session(md)
    w.relabel({("system", 2.5): "Remote 2", ("system", 5.0): "Remote 1", ("system", 1.0): "Remote 3"})
    w.footer(T0 + timedelta(hours=1, seconds=9), 9.0)
    assert md.read_text() == render_records(read_records(md))


def test_relabel_no_change_returns_zero_and_leaves_files(tmp_path):
    md = tmp_path / "m.md"
    w = second_session(md)
    before = (md.read_bytes(), sidecar_path(md).read_bytes())
    assert w.relabel({}) == 0
    assert w.relabel({("system", 1.0): "Remote 1", ("system", 5.0): "Remote 2"}) == 0  # same labels
    assert w.relabel({("system", 99.0): "X", ("mic", 1.0): "Y"}) == 0  # unknown keys / channel mismatch
    assert (md.read_bytes(), sidecar_path(md).read_bytes()) == before


def test_relabel_then_footer_and_more_items(tmp_path):
    md = tmp_path / "m.md"
    w = second_session(md)
    assert w.relabel({("system", 5.0): "Remote 1"}) == 1
    w.item(seg("system", 7.0, 8.0, "tail", "Remote 1"))  # appending after a relabel works
    w.footer(T0 + timedelta(hours=1, seconds=9), 9.0)
    recs = read_records(md)
    assert isinstance(recs[-1], Footer)
    assert recs[-1].segments == 4
    assert [type(r).__name__ for r in recs] == ["Header", "Segment", "Segment", "Note", "Segment", "Segment", "Footer"]
    assert md.read_text() == render_records(recs)


def test_relabel_changes_paragraph_grouping(tmp_path):
    md = tmp_path / "m.md"
    w = TranscriptWriter(md)
    w.header(Header(T0, {"system": "s"}, "fake"))
    w.item(seg("system", 1.0, 2.0, "first", "Remote 1"))
    w.item(seg("system", 2.2, 3.0, "second", "Remote 1"))
    assert "**Remote 1** · 12:00:01\nfirst second" in md.read_text()
    assert w.relabel({("system", 2.2): "Remote 2"}) == 1
    text = md.read_text()
    assert "**Remote 1** · 12:00:01\nfirst" in text
    assert "first second" not in text
    assert "**Remote 2** · 12:00:02\nsecond" in text
    w.footer(T0, 4.0)
    assert md.read_text() == render_records(read_records(md))


def test_relabel_merges_speaker_blocks(tmp_path):
    md = tmp_path / "m.md"
    w = TranscriptWriter(md)
    w.header(Header(T0, {"system": "s"}, "fake"))
    w.item(seg("system", 1.0, 2.0, "first", "Remote 1"))
    w.item(seg("system", 2.2, 3.0, "second", "Remote 2"))
    assert w.relabel({("system", 2.2): "Remote 1"}) == 1
    assert "first second" in md.read_text()
    assert "Remote 2" not in md.read_text()


def test_relabel_sets_speaker_on_unlabelled_segments(tmp_path):
    md = tmp_path / "m.md"
    w = TranscriptWriter(md)
    w.header(Header(T0, {"system": "s"}, "fake"))
    w.item(seg("system", 1.0, 2.0, "hi"))
    assert "**Remote**" in md.read_text()
    assert w.relabel({("system", 1.0): "Remote 1"}) == 1
    assert "**Remote 1**" in md.read_text()
    assert json.loads(sidecar_path(md).read_text().splitlines()[1])["speaker"] == "Remote 1"


def test_relabel_before_header_or_after_close_is_noop(tmp_path):
    md = tmp_path / "m.md"
    w = TranscriptWriter(md)
    assert w.relabel({("system", 1.0): "X"}) == 0
    w.header(Header(T0, {}, "fake"))
    w.item(seg("system", 1.0, 2.0, "hi", "Remote 1"))
    w.footer(T0, 1.0)
    before = md.read_bytes()
    assert w.relabel({("system", 1.0): "Remote 2"}) == 0
    assert md.read_bytes() == before


# --- CLI ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, available, expected",
    [
        (None, ["mic", "system"], []),
        ([], ["mic", "system"], []),
        (["mic"], ["mic", "system"], ["mic"]),
        (["microphone"], ["mic", "system"], ["mic"]),
        (["system"], ["mic", "system"], ["system"]),
        (["sys"], ["mic", "system"], ["system"]),
        (["all"], ["mic", "system"], ["mic", "system"]),
        (["both"], ["mic", "system"], ["mic", "system"]),
        (["ALL"], ["mic", "system"], ["mic", "system"]),
        (["mic,system"], ["mic", "system"], ["mic", "system"]),
        (["mic, sys"], ["mic", "system"], ["mic", "system"]),
        (["mic", "system"], ["mic", "system"], ["mic", "system"]),
        (["system", "sys", "system,all"], ["mic", "system"], ["system", "mic"]),
        (["mic,,"], ["mic", "system"], ["mic"]),
        ([""], ["mic", "system"], []),
    ],
)
def test_diarize_channels_parsing(values, available, expected):
    assert _diarize_channels(values, available) == expected


def test_diarize_channels_unknown_raises():
    with pytest.raises(typer.BadParameter) as ei:
        _diarize_channels(["mic,speakers"], ["mic"])
    assert "speakers" in str(ei.value)


def test_diarize_channels_unrecorded_dropped_with_warning(capsys):
    assert _diarize_channels(["all"], ["system"]) == ["system"]
    err = capsys.readouterr().err
    assert "mic" in err and "warning" in err
    assert _diarize_channels(["mic"], []) == []


def test_diarize_channels_no_warning_when_available(capsys):
    _diarize_channels(["all"], ["mic", "system"])
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("cmd", ["record", "replay"])
def test_help_mentions_diarize_options(cmd):
    r = CliRunner().invoke(app, [cmd, "--help"])
    assert r.exit_code == 0
    assert "--diarize" in r.output
    assert "--speaker-threshold" in r.output
