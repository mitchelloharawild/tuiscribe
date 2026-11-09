"""Command-line entry points."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from .capture.devices import Device, DeviceError, list_devices, resolve
from .config import CHANNEL_LABELS, CHANNELS
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
DiarizeOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--diarize",
        metavar="INPUTS",
        help="Label separate speakers within these inputs: mic, system or all "
        "(comma-separated or repeated). E.g. --diarize system -> Remote 1, Remote 2, ...",
    ),
]
SpeakerThresholdOpt = Annotated[
    float,
    typer.Option(help="Voice similarity (0-1) needed to count as the same speaker. Lower merges more voices."),
]
CHANNEL_ALIASES = {"mic": "mic", "microphone": "mic", "system": "system", "sys": "system"}


def _diarize_channels(values: list[str] | None, available: list[str]) -> list[str]:
    """Parse --diarize values into channels; warns about inputs that are not being recorded."""
    wanted: list[str] = []
    for value in values or []:
        for part in value.split(","):
            name = part.strip().lower()
            if not name:
                continue
            if name in ("all", "both"):
                chans = list(CHANNELS)
            elif name in CHANNEL_ALIASES:
                chans = [CHANNEL_ALIASES[name]]
            else:
                raise typer.BadParameter(f"unknown input '{part.strip()}' (use mic, system or all)", param_hint="--diarize")
            wanted += [c for c in chans if c not in wanted]
    for ch in wanted:
        if ch not in available:
            typer.secho(f"warning: --diarize {ch}: that input is not being transcribed", fg="yellow", err=True)
    return [ch for ch in wanted if ch in available]


def _make_diarizer(channels: list[str], threshold: float):
    if not channels:
        return None
    from .diarize import Diarizer

    return Diarizer(channels, threshold=threshold)


def _default_output(started: datetime) -> Path:
    return Path(f"meeting-{started:%Y-%m-%d-%H%M%S}.md")


def _audio_paths(stem_path: Path, channels: list[str], started: datetime) -> dict[str, Path]:
    """`<stem>.mic.wav` etc. next to the transcript; never overwrite an earlier session's audio."""
    def paths(stem: str) -> dict[str, Path]:
        return {ch: stem_path.with_name(f"{stem}.{ch}.wav") for ch in channels}

    out = paths(stem_path.stem)
    if any(p.exists() for p in out.values()):  # e.g. -o names an existing transcript we append to
        out = paths(f"{stem_path.stem}-{started:%Y-%m-%d-%H%M%S}")
    return out


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
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Transcript file (default: meeting-<date>-<time>.md, from when recording starts). Appends if it exists.")] = None,
    mic: Annotated[str | None, typer.Option(help="Mic source: id, node name or description. Default: PipeWire default.")] = None,
    system: Annotated[str | None, typer.Option(help="Sink whose monitor is captured. Default: PipeWire default.")] = None,
    no_mic: Annotated[bool, typer.Option("--no-mic", help="Capture system audio only.")] = False,
    no_system: Annotated[bool, typer.Option("--no-system", help="Capture the mic only.")] = False,
    model: ModelOpt = "small.en",
    no_transcribe: Annotated[bool, typer.Option("--no-transcribe", help="Meters (and saved audio) only; no model, no transcript.")] = False,
    no_partials: Annotated[bool, typer.Option("--no-partials", help="Skip live preview passes (saves CPU).")] = False,
    gap_short: GapShortOpt = 2.0,
    gap_long: GapLongOpt = 10.0,
    diarize: DiarizeOpt = None,
    speaker_threshold: SpeakerThresholdOpt = 0.5,
    save_audio: Annotated[bool, typer.Option("--save-audio/--no-save-audio", help="Keep time-aligned <transcript>.mic.wav / .system.wav next to the transcript.")] = True,
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

    diarize_chans = _diarize_channels(diarize, [] if no_transcribe else list(chosen))
    transcriber = None
    if not no_transcribe:
        from .stt.faster_whisper import FasterWhisperTranscriber

        transcriber = FasterWhisperTranscriber(model)

    started = datetime.now()
    output = output or _default_output(started)
    controller = Controller(
        chosen,
        output=None if no_transcribe else output,
        transcriber=transcriber,
        render_opts=RenderOptions(gap_short=gap_short, gap_long=gap_long),
        audio_paths=_audio_paths(output, list(chosen), started) if save_audio else None,
        partials=not no_partials,
        diarizer=_make_diarizer(diarize_chans, speaker_threshold),
    )
    if no_tui or not sys.stdout.isatty():
        _run_plain(controller, duration)
    else:
        from .tui.app import TuiscribeApp

        tui = TuiscribeApp(controller, duration=duration)
        tui.run()
        if tui.abandoned:
            typer.secho("abandoned: speech still being transcribed was not written; no footer.", fg="yellow", err=True)
            os._exit(130)
    _print_summary(controller)


@app.command()
def replay(
    files: Annotated[list[Path], typer.Argument(help="Audio file(s). Prefix with 'mic=' or 'system=' to pick the channel (default: system).", show_default=False)],
    output: Annotated[Path, typer.Option("-o", "--output", help="Transcript file to write (overwritten).")],
    model: ModelOpt = "small.en",
    gap_short: GapShortOpt = 2.0,
    gap_long: GapLongOpt = 10.0,
    diarize: DiarizeOpt = None,
    speaker_threshold: SpeakerThresholdOpt = 0.5,
) -> None:
    """Run the transcription pipeline on recorded audio (e.g. the saved .wav files)."""
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

    diarizer = _make_diarizer(_diarize_channels(diarize, list(audio)), speaker_threshold)
    for p in (output, sidecar_path(output)):
        p.unlink(missing_ok=True)
    tr = FasterWhisperTranscriber(model)
    typer.echo(f"loading {tr.name} ...")
    t0 = time.monotonic()
    tr.load()
    if diarizer is not None:
        diarizer.load()
    writer = TranscriptWriter(output, RenderOptions(gap_short=gap_short, gap_long=gap_long))
    started_at = datetime.now().astimezone()
    writer.header(Header(started_at=started_at, sources=names, model=tr.name))

    def progress(i: int, n: int, seg: Segment | None) -> None:
        if seg is not None:
            typer.echo(f"[{i}/{n}] {fmt_clock(started_at, seg.start)} {seg.label}: {seg.text}")

    t1 = time.monotonic()
    segs = transcribe_offline(audio, tr, writer, progress, diarizer=diarizer)
    audio_s = max(len(a) for a in audio.values()) / 16000
    writer.footer(started_at + timedelta(seconds=audio_s), audio_s)
    if diarizer is not None:
        for ch in sorted(diarizer.channels):
            n = len({s.speaker for s in segs if s.channel == ch})
            typer.echo(f"{CHANNEL_LABELS[ch]}: {n} speaker{'s' if n != 1 else ''}")
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
        signal.signal(signal.SIGINT, signal.default_int_handler)
        typer.secho("stopping: transcribing what is left ... (Ctrl-C again to abandon)", dim=True, err=True)
        try:
            controller.stop()
        except KeyboardInterrupt:
            typer.secho("abandoned: pending speech was not written; no footer.", fg="yellow", err=True)
            os._exit(130)
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
    if controller.relabelled:
        typer.echo(f"speaker labels revised for {controller.relabelled} segments after the session")
    for ch, secs in controller.saved_seconds().items():
        typer.echo(f"audio: {controller.audio_paths[ch]} ({secs:.1f} s)")


def _fail(msg: str) -> int:
    typer.secho(f"error: {msg}", fg="red", err=True)
    return 1


def main() -> None:
    app()
