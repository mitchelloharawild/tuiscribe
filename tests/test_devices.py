import json

import pytest

from tuiscribe.capture.devices import Device, DeviceError, parse_devices, resolve


def node(id, name, media_class, desc=None, nick=None, state="idle"):
    props = {"node.name": name, "media.class": media_class, "device.api": "alsa"}
    if desc:
        props["node.description"] = desc
    if nick:
        props["node.nick"] = nick
    return {"id": id, "type": "PipeWire:Interface:Node", "info": {"state": state, "props": props}}


def metadata(as_string=False):
    def val(n):
        return json.dumps({"name": n}) if as_string else {"name": n}

    return {
        "id": 99,
        "type": "PipeWire:Interface:Metadata",
        "props": {"metadata.name": "default"},
        "metadata": [
            {"subject": 0, "key": "default.audio.source", "value": val("mic_b")},
            {"subject": 0, "key": "default.audio.sink", "value": val("spk")},
            {"subject": 0, "key": "default.audio.bogus", "value": "not json"},
        ],
    }


def dump(as_string=False):
    return [
        node(1, "mic_a", "Audio/Source", desc="Alpha Mic"),
        node(2, "mic_b", "Audio/Source", desc="Zed Mic"),
        node(3, "spk", "Audio/Sink", desc="Built-in Speakers"),
        node(4, "hdmi_out", "Audio/Sink", nick="HDMI Nick"),
        node(5, "bare_sink", "Audio/Sink"),
        node(6, "stream", "Stream/Output/Audio", desc="Firefox"),
        {"id": 7, "type": "PipeWire:Interface:Client", "info": {}},
        metadata(as_string),
    ]


@pytest.mark.parametrize("as_string", [False, True])
def test_parse(as_string):
    devs = parse_devices(dump(as_string))
    assert {d.id for d in devs} == {1, 2, 3, 4, 5}
    by = {d.id: d for d in devs}
    assert by[2].is_default and by[3].is_default
    assert not by[1].is_default and not by[4].is_default
    assert by[1].kind == "source" and by[3].kind == "sink"
    assert by[1].api == "alsa" and by[1].state == "idle"


def test_description_fallback():
    by = {d.id: d for d in parse_devices(dump())}
    assert by[1].description == "Alpha Mic"
    assert by[4].description == "HDMI Nick"
    assert by[5].description == "bare_sink"


def test_sort_order():
    devs = parse_devices(dump())
    # kind sorts alphabetically (sink < source); default first, then description
    assert [d.id for d in devs] == [3, 4, 5, 2, 1]


def test_no_metadata_no_defaults():
    devs = parse_devices([n for n in dump() if n["id"] != 99])
    assert not any(d.is_default for d in devs)


def test_looks_like_speakers():
    by = {d.id: d for d in parse_devices(dump())}
    assert by[3].looks_like_speakers
    assert by[4].looks_like_speakers  # hdmi in name
    assert not by[5].looks_like_speakers
    assert not by[2].looks_like_speakers
    spk_source = Device(9, "speaker_mic", "Speaker Mic", "source", None, None)
    assert not spk_source.looks_like_speakers


@pytest.fixture
def devs():
    return parse_devices(dump())


def test_resolve_default(devs):
    assert resolve(None, "source", devs).id == 2
    assert resolve(None, "sink", devs).id == 3


def test_resolve_id_and_name(devs):
    assert resolve("1", "source", devs).name == "mic_a"
    assert resolve("mic_a", "source", devs).id == 1


def test_resolve_id_wrong_kind(devs):
    with pytest.raises(DeviceError):
        resolve("3", "source", devs)


def test_resolve_description(devs):
    assert resolve("alpha", "source", devs).id == 1
    assert resolve("BUILT-IN", "sink", devs).id == 3


def test_resolve_ambiguous(devs):
    with pytest.raises(DeviceError, match="several"):
        resolve("mic", "source", devs)


def test_resolve_no_match(devs):
    with pytest.raises(DeviceError, match="no audio"):
        resolve("nothing", "source", devs)


def test_resolve_no_default():
    d = [x for x in parse_devices(dump()) if x.id != 2]
    d = [x for x in d if not x.is_default]
    with pytest.raises(DeviceError, match="default"):
        resolve(None, "source", d)


def test_resolve_no_devices():
    with pytest.raises(DeviceError):
        resolve(None, "source", [])
