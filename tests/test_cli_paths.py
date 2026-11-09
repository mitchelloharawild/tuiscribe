from datetime import datetime
from pathlib import Path

from tuiscribe.cli import _audio_paths, _default_output
from tuiscribe.controller import Controller
from tuiscribe.capture.devices import Device

STARTED = datetime(2026, 11, 10, 14, 32, 5)


def test_default_output_uses_start_time():
    assert _default_output(STARTED) == Path("meeting-2026-11-10-143205.md")


def test_audio_next_to_transcript(tmp_path):
    out = tmp_path / "standup.md"
    assert _audio_paths(out, ["mic", "system"], STARTED) == {
        "mic": tmp_path / "standup.mic.wav",
        "system": tmp_path / "standup.system.wav",
    }


def test_audio_never_overwrites_previous_session(tmp_path):
    out = tmp_path / "standup.md"
    (tmp_path / "standup.system.wav").write_bytes(b"old")
    paths = _audio_paths(out, ["mic", "system"], STARTED)
    assert paths == {
        "mic": tmp_path / "standup-2026-11-10-143205.mic.wav",
        "system": tmp_path / "standup-2026-11-10-143205.system.wav",
    }


def test_controller_only_keeps_paths_for_recorded_channels(tmp_path):
    dev = Device(1, "n1", "d1", "source", None, "running")
    c = Controller({"mic": dev}, audio_paths={"mic": tmp_path / "a.mic.wav", "system": tmp_path / "a.system.wav"})
    assert c.audio_paths == {"mic": tmp_path / "a.mic.wav"}
