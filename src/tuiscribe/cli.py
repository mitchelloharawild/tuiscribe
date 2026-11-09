"""Command-line entry points."""

from __future__ import annotations

import json
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from .capture.devices import Device, DeviceError, list_devices, resolve
from .config import CHANNEL_LABELS
from .controller import Controller
from .events import FinalEvent, PartialEvent, SourceEvent, StatusEvent
from .records import Header, Note, Segment, from_json
from .render import RenderOptions, fmt_clock, render_records

app = typer.Typer(
    help="Capture mic + system audio from PipeWire and transcribe it to Markdown.",
    no_args_is_help=True,
    add_completion=False,
)

ModelOpt = Annotated[str, typer.Option("--model", help="faster-whisper model: tiny.en, base.en, small.en, medium.en, distil-large-v3, ...")]
GapShortOpt = Annotated[float, typer.Option(help="Pause (s) that starts a new paragraph for the same speaker.")]
GapLongOpt = Annotated[float, typer.Option(help="Pause (s) that gets a '⋯ N s of silence ⋯' marker.")]


def _default_output() -> Path:
    return Path(f"meeting-{datetime.now():%Y-%m-%d-%H%M}.md")


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
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Transcript file (default: meeting-<date>-<time>.md). Appends if it exists.")] = None,
    mic: Annotated[str | None, typer.Option(help="Mic source: id, node name or description. Default: PipeWire default.")] = None,
    system: Annotated[str | None, typer.Option(help="Sink whose monitor is captured. Default: PipeWire default.")] = None,
    no_mic: Annotated[bool, typer.Option("--no-mic", help="Capture system audio only.")] = False,
    no_system: Annotated[bool, typer.Option("--no-system", help="Capture the mic only.")] = False,
    model: ModelOpt = "small.en",
    no_transcribe: Annotated[bool, typer.Option("--no-transcribe", help="Meters (and --save-audio) only; no model, no transcript.")] = False,
    no_partials: Annotated[bool, typer.Option("--no-partials", help="Skip live preview passes (saves CPU).")] = False,
    gap_short: GapShortOpt = 2.0,
    gap_long: GapLongOpt = 10.0,
    save_audio: Annotated[Path | None, typer.Option(help="Also write time-aligned mic.wav / system.wav into this directory.")] = None,
    duration: Annotated[float | None, typer.Option(help="Stop after this many seconds.")] = None,
    no_tui: Annotated[bool, typer.Option("--no-tui", help="Plain log output instead of the interactive display.")] = False,
) -> None:
    """Capture mic + system audio and transcribe it live to Markdown."""
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

    transcriber = None
    if not no_transcribe:
        from .stt.faster_whisper import FasterWhisperTranscriber

        transcriber = FasterWhisperTranscriber(model)
        output = output or _default_output()

    controller = Controller(
        chosen,
        output=None if no_transcribe else output,
        transcriber=transcriber,
        render_opts=RenderOptions(gap_short=gap_short, gap_long=gap_long),
        save_audio=save_audio,
        partials=not no_partials,
    )
    if no_tui or not sys.stdout.isatty():
        _run_plain(controller, duration)
    else:
        from .tui.app import TuiscribeApp

        TuiscribeApp(controller, duration=duration).run()
    _print_summary(controller)


@app.command()
def replay(
    files: Annotated[list[Path], typer.Argument(help="Audio file(s). Prefix with 'mic=' or 'system=' to pick the channel (default: system).", show_default=False)],
    output: Annotated[Path, typer.Option("-o", "--output", help="Transcript file to write (overwritten).")],
    model: ModelOpt = "small.en",
    gap_short: GapShortOpt = 2.0,
    gap_long: GapLongOpt = 10.0,
) -> None:
    """Run the transcription pipeline on recorded audio (e.g. from --save-audio)."""
    from .audio.decode import load_audio
    from .output import TranscriptWriter, sidecar_path
    from .pipeline import transcribe_offline
    from .stt.faster_whisper import FasterWhisperTranscriber

    audio: dict = {}
    names: dict[str, str] = {}
    for spec in files:
        ch, _, path = str(spec).rpartition("=") if "=" in str(spec) else ("system", "", str(spec))
        if ch not in CHANNEL_LABELS:
            raise typer.Exit(_fail(f"unknown channel '{ch}' (use mic= or system=)"))
        if ch in audio:
            raise typer.Exit(_fail(f"two files for channel '{ch}'"))
        try:
            audio[ch] = load_audio(Path(path))
        except RuntimeError as e:
            raise typer.Exit(_fail(str(e)))
        names[ch] = Path(path).name
        typer.echo(f"{CHANNEL_LABELS[ch]:<7} {path} ({len(audio[ch]) / 16000:.1f} s)")

    for p in (output, sidecar_path(output)):
        p.unlink(missing_ok=True)
    tr = FasterWhisperTranscriber(model)
    typer.echo(f"loading {tr.name} ...")
    t0 = time.monotonic()
    tr.load()
    writer = TranscriptWriter(output, RenderOptions(gap_short=gap_short, gap_long=gap_long))
    started_at = datetime.now().astimezone()
    writer.header(Header(started_at=started_at, sources=names, model=tr.name))

    def progress(i: int, n: int, seg: Segment | None) -> None:
        if seg is not None:
            typer.echo(f"[{i}/{n}] {fmt_clock(started_at, seg.start)} {seg.label}: {seg.text}")

    t1 = time.monotonic()
    transcribe_offline(audio, tr, writer, progress)
    audio_s = max(len(a) for a in audio.values()) / 16000
    writer.footer(started_at + timedelta(seconds=audio_s), audio_s)
    typer.echo(f"wrote {output} ({writer.segments} segments; STT {time.monotonic() - t1:.1f} s for {audio_s:.1f} s audio, load {t1 - t0:.1f} s)")


@app.command()
def render(
    sidecar: Annotated[Path, typer.Argument(help="The .jsonl written next to a transcript.")],
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Markdown file to write (default: stdout).")] = None,
    gap_short: GapShortOpt = 2.0,
    gap_long: GapLongOpt = 10.0,
) -> None:
    """Regenerate Markdown from a .jsonl sidecar (e.g. with different gap settings)."""
    try:
        records = [from_json(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, TypeError) as e:
        raise typer.Exit(_fail(f"cannot read {sidecar}: {e}"))
    md = render_records(records, RenderOptions(gap_short=gap_short, gap_long=gap_long))
    if output is None:
        typer.echo(md, nl=False)
    else:
        output.write_text(md, encoding="utf-8")
        typer.echo(f"wrote {output}")


# --- plain (--no-tui) mode ---------------------------------------------------


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
    if controller.writer is not None:
        typer.echo(f"writing {controller.writer.md_path}")
    try:
        next_line = time.monotonic() + 5.0
        while not stop and (duration is None or controller.elapsed() < duration):
            _print_events(controller)
            time.sleep(0.05)
            if time.monotonic() >= next_line:
                next_line += 5.0
                typer.secho(f"{controller.elapsed():7.1f}s  {_levels_line(controller)}", dim=True, err=True)
    finally:
        typer.secho("stopping: transcribing what is left ...", dim=True, err=True)
        controller.stop()
        _print_events(controller)
        signal.signal(signal.SIGINT, old)
        signal.signal(signal.SIGTERM, old_term)


def _levels_line(controller: Controller) -> str:
    parts = []
    for ch, lvl in controller.poll_levels().items():
        label = CHANNEL_LABELS[ch]
        if lvl is None:
            parts.append(f"{label} no data")
            continue
        flag = " MUTED" if controller.muted[ch] else ""
        flag += " ZERO" if lvl.digital_silence else ""
        flag += " CLIP" if lvl.clipped else ""
        parts.append(f"{label} {lvl.rms_db:6.1f} dB{flag}")
    st = controller.stats()
    if st is not None:
        parts.append(f"STT {st.model_state} q={st.queue}" + (f" lat={st.latency:.1f}s" if st.latency is not None else ""))
    return "  ".join(parts)


def _print_events(controller: Controller) -> None:
    for ev in controller.bus.drain():
        match ev:
            case SourceEvent(channel, device, "lost", detail):
                typer.secho(f"{CHANNEL_LABELS[channel]} LOST ({device.description}): {detail}", fg="red", err=True)
            case StatusEvent(sev, message, _):
                typer.secho(f"{sev}: {message}", fg={"info": None, "warning": "yellow", "error": "red"}[sev], err=True)
            case FinalEvent(Segment() as seg, _) if controller.started_at is not None:
                typer.echo(f"{fmt_clock(controller.started_at, seg.start)} {seg.label}: {seg.text}")
            case FinalEvent(Note() as note, _):
                typer.echo(f"note: {note.text}")
            case PartialEvent():
                pass


def _print_summary(controller: Controller) -> None:
    if controller.writer is not None:
        typer.echo(f"transcript: {controller.writer.md_path} ({controller.writer.segments} segments), sidecar: {controller.writer.jsonl_path}")
    for ch, secs in controller.saved_seconds().items():
        if controller.save_audio is not None:
            typer.echo(f"audio: {controller.save_audio / f'{ch}.wav'} ({secs:.1f} s)")


def _fail(msg: str) -> int:
    typer.secho(f"error: {msg}", fg="red", err=True)
    return 1


def main() -> None:
    app()
