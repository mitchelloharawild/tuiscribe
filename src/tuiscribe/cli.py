"""Command-line entry points."""

from __future__ import annotations

import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from .capture.devices import Device, DeviceError, list_devices, resolve
from .config import CHANNEL_LABELS
from .controller import Controller
from .events import SourceEvent, StatusEvent

app = typer.Typer(
    help="Capture mic + system audio from PipeWire and transcribe it to Markdown.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def devices(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List mic sources and system-audio sinks (their monitors are captured)."""
    try:
        devs = list_devices()
    except DeviceError as e:
        raise typer.Exit(_fail(str(e)))
    if as_json:
        typer.echo(json.dumps([asdict(d) for d in devs], indent=2))
        return
    for kind, title in (("source", "Mic sources (--mic)"), ("sink", "System audio sinks (--system)")):
        typer.secho(title, bold=True)
        for d in (d for d in devs if d.kind == kind):
            mark = "*" if d.is_default else " "
            warn = "  (speakers)" if d.looks_like_speakers else ""
            typer.echo(f"  {mark} {d.id:>4}  {d.description:<40} {d.api or '':<7} {d.state or ''}{warn}")
            typer.secho(f"          {d.name}", dim=True)
        typer.echo()
    typer.secho("* = current PipeWire default. --mic/--system take an id, node name or part of the description.", dim=True)


@app.command()
def record(
    mic: Annotated[str | None, typer.Option(help="Mic source: id, node name or description. Default: PipeWire default.")] = None,
    system: Annotated[str | None, typer.Option(help="Sink whose monitor is captured. Default: PipeWire default.")] = None,
    no_mic: Annotated[bool, typer.Option("--no-mic", help="Capture system audio only.")] = False,
    no_system: Annotated[bool, typer.Option("--no-system", help="Capture the mic only.")] = False,
    save_audio: Annotated[Path | None, typer.Option(help="Also write time-aligned mic.wav / system.wav into this directory.")] = None,
    duration: Annotated[float | None, typer.Option(help="Stop after this many seconds.")] = None,
    no_tui: Annotated[bool, typer.Option("--no-tui", help="Plain log output instead of the interactive display.")] = False,
) -> None:
    """Capture both channels with live level meters.

    Transcription to Markdown is not wired in yet (build steps 3-5).
    """
    if no_mic and no_system:
        raise typer.Exit(_fail("--no-mic and --no-system leave nothing to record"))
    try:
        devs = list_devices()
        chosen: dict[str, Device] = {}
        if not no_mic:
            chosen["mic"] = resolve(mic, "source", devs)
        if not no_system:
            chosen["system"] = resolve(system, "sink", devs)
    except DeviceError as e:
        raise typer.Exit(_fail(str(e)))

    controller = Controller(chosen, save_audio=save_audio)
    if no_tui or not sys.stdout.isatty():
        _run_plain(controller, duration)
    else:
        from .tui.app import TuiscribeApp

        TuiscribeApp(controller, duration=duration).run()
    _print_summary(controller)


def _run_plain(controller: Controller, duration: float | None) -> None:
    stop = False

    def _sigint(*_: object) -> None:
        nonlocal stop
        stop = True

    old = signal.signal(signal.SIGINT, _sigint)
    old_term = signal.signal(signal.SIGTERM, _sigint)
    controller.start()
    for ch, dev in controller.devices.items():
        typer.echo(f"{CHANNEL_LABELS[ch]:<7} {dev.description}  [{dev.name}]")
    try:
        next_line = time.monotonic() + 1.0
        while not stop and (duration is None or controller.elapsed() < duration):
            _print_events(controller)
            time.sleep(0.05)
            if time.monotonic() >= next_line:
                next_line += 1.0
                parts = []
                for ch, lvl in controller.poll_levels().items():
                    label = CHANNEL_LABELS[ch]
                    if lvl is None:
                        parts.append(f"{label} no data")
                    else:
                        flag = " MUTED" if controller.muted[ch] else ""
                        flag += " ZERO" if lvl.digital_silence else ""
                        flag += " CLIP" if lvl.clipped else ""
                        parts.append(f"{label} {lvl.rms_db:6.1f} dB (peak {lvl.peak_db:6.1f}){flag}")
                typer.echo(f"{controller.elapsed():7.1f}s  " + "  ".join(parts))
    finally:
        controller.stop()
        _print_events(controller)
        signal.signal(signal.SIGINT, old)
        signal.signal(signal.SIGTERM, old_term)


def _print_events(controller: Controller) -> None:
    for ev in controller.bus.drain():
        match ev:
            case SourceEvent(channel, device, "lost", detail):
                typer.secho(f"{CHANNEL_LABELS[channel]} LOST ({device.description}): {detail}", fg="red", err=True)
            case StatusEvent("warning" | "error" as sev, message, _):
                typer.secho(f"{sev}: {message}", fg="yellow", err=True)


def _print_summary(controller: Controller) -> None:
    for ch, secs in controller.saved_seconds().items():
        if controller.save_audio is not None:
            typer.echo(f"saved {controller.save_audio / f'{ch}.wav'} ({secs:.1f} s)")


def _fail(msg: str) -> int:
    typer.secho(f"error: {msg}", fg="red", err=True)
    return 1


def main() -> None:
    app()
