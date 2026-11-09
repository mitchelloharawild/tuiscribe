"""Discover PipeWire audio nodes via `pw-dump`.

Mic candidates are `Audio/Source` nodes; system-audio candidates are
`Audio/Sink` nodes (we capture their monitor). Defaults come from the
`default` metadata object that WirePlumber maintains.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Literal

Kind = Literal["source", "sink"]


@dataclass(frozen=True)
class Device:
    id: int
    name: str  # node.name, stable across restarts; used as pw-record --target
    description: str
    kind: Kind
    api: str | None  # "alsa", "bluez5", ...
    state: str | None  # "running", "suspended", "idle"
    is_default: bool = False

    @property
    def looks_like_speakers(self) -> bool:
        """Heuristic for the headphones warning: built-in speakers or HDMI outputs."""
        if self.kind != "sink":
            return False
        n = self.name.lower()
        d = self.description.lower()
        return "speaker" in n or "speaker" in d or "hdmi" in n


class DeviceError(RuntimeError):
    pass


def _pw_dump() -> list[dict]:
    try:
        out = subprocess.run(
            ["pw-dump"], check=True, capture_output=True, text=True, timeout=5
        ).stdout
    except FileNotFoundError as e:
        raise DeviceError("pw-dump not found; is PipeWire installed?") from e
    except subprocess.SubprocessError as e:
        raise DeviceError(f"pw-dump failed: {e}") from e
    return json.loads(out)


def parse_devices(dump: list[dict]) -> list[Device]:
    """Pure parser over `pw-dump` output (separate for testing)."""
    defaults: dict[str, str] = {}
    for obj in dump:
        if obj.get("type", "").endswith(":Metadata") and (
            obj.get("props", {}).get("metadata.name") == "default"
        ):
            for m in obj.get("metadata") or []:
                v = m.get("value")
                if isinstance(v, dict) and "name" in v:
                    defaults[m["key"]] = v["name"]
                elif isinstance(v, str):  # older versions serialise as a JSON string
                    try:
                        defaults[m["key"]] = json.loads(v)["name"]
                    except (ValueError, KeyError, TypeError):
                        pass

    default_source = defaults.get("default.audio.source")
    default_sink = defaults.get("default.audio.sink")

    devices: list[Device] = []
    for obj in dump:
        if not obj.get("type", "").endswith(":Node"):
            continue
        info = obj.get("info") or {}
        props = info.get("props") or {}
        media_class = props.get("media.class")
        if media_class == "Audio/Source":
            kind: Kind = "source"
            default = default_source
        elif media_class == "Audio/Sink":
            kind = "sink"
            default = default_sink
        else:
            continue
        name = props.get("node.name", str(obj["id"]))
        devices.append(
            Device(
                id=obj["id"],
                name=name,
                description=props.get("node.description") or props.get("node.nick") or name,
                kind=kind,
                api=props.get("device.api"),
                state=info.get("state"),
                is_default=(name == default),
            )
        )
    devices.sort(key=lambda d: (d.kind, not d.is_default, d.description))
    return devices


def list_devices() -> list[Device]:
    return parse_devices(_pw_dump())


def resolve(spec: str | None, kind: Kind, devices: list[Device] | None = None) -> Device:
    """Find a device by id, exact node.name, or case-insensitive description substring.

    `spec=None` returns the current default of that kind.
    """
    devices = [d for d in (devices if devices is not None else list_devices()) if d.kind == kind]
    if not devices:
        raise DeviceError(f"no PipeWire audio {kind}s found")
    if spec is None:
        for d in devices:
            if d.is_default:
                return d
        raise DeviceError(f"no default audio {kind} set in PipeWire")
    if spec.isdigit():
        for d in devices:
            if d.id == int(spec):
                return d
    for d in devices:
        if d.name == spec:
            return d
    matches = [d for d in devices if spec.lower() in d.description.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(f"{d.id} ({d.description})" for d in matches)
        raise DeviceError(f"'{spec}' matches several {kind}s: {names}")
    raise DeviceError(f"no audio {kind} matches '{spec}' (see `tuiscribe devices`)")
