import wave

import numpy as np
import pytest

from tuiscribe.audio.wav import AlignedWavWriter
from tuiscribe.capture.devices import Device
from tuiscribe.config import SAMPLE_RATE
from tuiscribe.controller import Controller
from tuiscribe.events import EventBus, SetMute, StatusEvent


def dev(id, name, kind):
    return Device(id, name, name, kind, "alsa", "running", True)


@pytest.fixture
def ctl():
    return Controller({"mic": dev(1, "mic", "source"), "system": dev(2, "sys", "sink")}, bus=EventBus())


def sig(v, n=1600):
    return np.full(n, v, dtype=np.float32)


def test_levels_reflect_audio(ctl):
    ctl._on_chunk("mic", 0.0, sig(0.5))
    lv = ctl.poll_levels()
    assert lv["system"] is None
    m = lv["mic"]
    assert m.channel == "mic"
    assert m.rms_db == pytest.approx(-6.02, abs=0.01)
    assert m.peak_db == pytest.approx(-6.02, abs=0.01)
    assert not m.clipped and not m.digital_silence
    assert ctl.poll_levels()["mic"] is None


def test_digital_silence(ctl):
    ctl._on_chunk("mic", 0.0, sig(0.0))
    assert ctl.poll_levels()["mic"].digital_silence


def test_clipped(ctl):
    ctl._on_chunk("mic", 0.0, sig(0.999))
    assert ctl.poll_levels()["mic"].clipped
    ctl._on_chunk("mic", 0.0, sig(0.998))
    assert not ctl.poll_levels()["mic"].clipped


def test_mute_toggle_levels_but_no_write(ctl, tmp_path):
    p = tmp_path / "mic.wav"
    ctl._writers["mic"] = AlignedWavWriter(p, 0.0)
    ctl._on_chunk("mic", 0.0, sig(0.5))
    assert ctl.saved_seconds()["mic"] == 1600 / SAMPLE_RATE

    ctl.apply(SetMute("mic"))
    assert ctl.muted["mic"] is True
    ctl._on_chunk("mic", 0.1, sig(0.5))
    assert ctl.poll_levels()["mic"] is not None
    assert ctl.saved_seconds()["mic"] == 1600 / SAMPLE_RATE

    ctl.apply(SetMute("mic"))
    assert ctl.muted["mic"] is False
    ctl._on_chunk("mic", 0.3, sig(0.5))
    ctl._writers["mic"].close()
    with wave.open(str(p), "rb") as w:
        # padded up to t=0.3s (4800 frames), then 1600 more
        assert w.getnframes() == 4800 + 1600


def test_explicit_mute_values(ctl):
    ctl.apply(SetMute("mic", True))
    ctl.apply(SetMute("mic", True))
    assert ctl.muted["mic"] is True
    ctl.apply(SetMute("mic", False))
    assert ctl.muted["mic"] is False


def test_mute_publishes_status(ctl):
    ctl.apply(SetMute("system"))
    ev = ctl.bus.drain()
    assert ev == [StatusEvent("info", "SYSTEM muted", "system")]
    ctl.apply(SetMute("system"))
    assert ctl.bus.drain() == [StatusEvent("info", "SYSTEM unmuted", "system")]
