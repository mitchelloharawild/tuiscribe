import time

import numpy as np
import pytest

from tuiscribe.diarize import voices as V
from tuiscribe.diarize.voices import VoiceBank, centroid_of


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def around(base, rng, noise=0.05):
    return unit(base + rng.normal(0, noise, len(base)))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def people(rng):
    return {n: unit(rng.normal(size=32)) for n in ("Ann", "Bob", "Cat")}


def enrol(bank, name, base, rng, n=10, channel="mic", source="a.md", dur=3.0):
    return bank.add(name, [(around(base, rng), i * 5.0, i * 5.0 + dur) for i in range(n)], channel, source)


def test_add_save_reload_and_remove(tmp_path, rng, people):
    bank = VoiceBank(tmp_path)
    assert not bank and bank.names() == []
    assert enrol(bank, "Ann Lee", people["Ann"], rng) == 10
    assert (tmp_path / "ann-lee.npz").exists()
    again = VoiceBank(tmp_path)
    v = again.get("Ann Lee")
    assert again.names() == ["Ann Lee"] and len(v) == 10 and v.seconds == pytest.approx(30)
    assert set(v.channel) == {"mic"} and set(v.source) == {"a.md"}
    assert again.remove("Ann Lee") and not (tmp_path / "ann-lee.npz").exists()
    assert VoiceBank(tmp_path).names() == []
    assert not again.remove("nobody")


def test_add_skips_short_missing_and_duplicate_samples(tmp_path, rng, people):
    bank = VoiceBank(tmp_path)
    e = around(people["Ann"], rng)
    assert bank.add("Ann", [(e, 0, 0.5), (None, 1, 5), (e, 10, 13)], "mic", "a.md") == 1
    assert bank.add("Ann", [(e, 10, 13)], "mic", "a.md") == 0  # same source/channel/time
    assert bank.add("Ann", [(e, 10, 13)], "mic", "b.md") == 1
    assert bank.add("Nobody", [(e, 0, 0.2)], "mic", "a.md") == 0
    assert bank.get("Nobody") is None


def test_same_slug_gets_its_own_file(tmp_path, rng, people):
    bank = VoiceBank(tmp_path)
    enrol(bank, "Ann", people["Ann"], rng)
    enrol(bank, "ann", people["Bob"], rng)
    assert len(list(tmp_path.glob("*.npz"))) == 2
    assert VoiceBank(tmp_path).names() == ["Ann", "ann"]


def test_cap_drops_oldest(tmp_path, rng, people, monkeypatch):
    monkeypatch.setattr(V, "MAX_SAMPLES", 15)
    bank = VoiceBank(tmp_path)
    enrol(bank, "Ann", people["Ann"], rng, n=10, source="old.md")
    time.sleep(0.01)
    enrol(bank, "Ann", people["Ann"], rng, n=10, source="new.md")
    v = VoiceBank(tmp_path).get("Ann")
    assert len(v) == 15 and list(v.source).count("new.md") == 10


def test_other_model_is_not_matched(tmp_path, rng, people):
    enrol(VoiceBank(tmp_path, model="other"), "Ann", people["Ann"], rng)
    bank = VoiceBank(tmp_path)
    assert bank.names() == [] and bank.get("Ann").model == "other"
    with pytest.raises(ValueError):
        enrol(bank, "Ann", people["Ann"], rng)


def test_match_names_clear_groups_one_to_one(tmp_path, rng, people):
    bank = VoiceBank(tmp_path)
    for n in ("Ann", "Bob"):
        enrol(bank, n, people[n], rng)
    groups = {
        0: around(people["Bob"], rng),
        1: around(people["Ann"], rng),
        2: around(people["Ann"], rng, noise=0.08),  # Ann twice: only the closer gets her
        3: people["Cat"],  # unknown
    }
    m = bank.match(groups, "mic")
    assert m[0][0] == "Bob" and m[0][1] > 0.9
    assert sum(1 for name, _ in m.values() if name == "Ann") == 1
    assert 3 not in m
    assert bank.match(groups, "mic", exclude={"Bob"}).get(0) is None


def test_match_needs_margin_over_next_best(tmp_path, rng, people):
    bank = VoiceBank(tmp_path)
    enrol(bank, "Ann", people["Ann"], rng)
    enrol(bank, "Twin", people["Ann"], rng)  # sounds the same as Ann
    assert bank.match({0: around(people["Ann"], rng)}, "mic") == {}


def test_centroid_prefers_same_input_when_enough(tmp_path, rng, people):
    bank = VoiceBank(tmp_path)
    enrol(bank, "Ann", people["Ann"], rng, n=10, channel="mic")
    enrol(bank, "Ann", people["Bob"], rng, n=2, channel="system", source="b.md")  # 6 s: too little alone
    v = bank.get("Ann")
    assert v.centroid("system") @ people["Ann"] > 0.5  # pooled with the mic samples
    enrol(bank, "Ann", people["Bob"], rng, n=4, channel="system", source="c.md")  # now 18 s
    assert bank.get("Ann").centroid("system") @ people["Bob"] > 0.9


def test_newer_samples_weigh_more(tmp_path, rng, people):
    bank = VoiceBank(tmp_path)
    enrol(bank, "Ann", people["Ann"], rng, n=10)
    enrol(bank, "Ann", people["Bob"], rng, n=10, source="b.md")
    v = bank.get("Ann")
    v.added[:10] -= 5 * 365 * 86400  # the Ann-like samples are five years old
    c = v.centroid("mic")
    assert c @ people["Bob"] > c @ people["Ann"]


def test_centroid_of_prefers_reliable():
    a, b = unit([1, 0]), unit([0, 1])
    assert centroid_of([a, b], [3.0, 0.5]) @ a == pytest.approx(1)
    assert centroid_of([b], [0.5]) @ b == pytest.approx(1)
    assert centroid_of([None], [3.0]) is None


def test_voices_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TUISCRIBE_VOICES_DIR", str(tmp_path / "v"))
    assert V.voices_dir() == tmp_path / "v"
    monkeypatch.delenv("TUISCRIBE_VOICES_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert V.voices_dir() == tmp_path / "tuiscribe" / "voices"
