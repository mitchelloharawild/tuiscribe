# tuiscribe

Live meeting transcription on Linux. It captures your **microphone** and **system
audio** (everything you hear) through PipeWire, transcribes both locally with
Whisper, and appends the result to a Markdown file. Mic speech is labelled
**You**, system audio is labelled **Remote**, and pauses and silences are marked.

Everything runs on the CPU and offline after the first model download.
**Use headphones.** With speakers, the mic also picks up remote voices and they
get transcribed twice.

## Setup

Needs PipeWire (`pw-record`, `pw-dump`), Python 3.12 and [uv](https://docs.astral.sh/uv/);
`ffmpeg` is only needed for `replay`.

```sh
cd tuiscribe
uv sync
uv run tuiscribe devices        # check what will be captured
```

The first `record` or `replay` downloads the Whisper model (`small.en`, about 460 MB)
into `~/.cache/huggingface`.

## Usage

```sh
uv run tuiscribe record -o meeting.md
```

This opens a live display with input meters for both channels, the transcript
as it is written (dimmed lines are live guesses that get replaced), and status:

| Key | Action |
|---|---|
| `m` / `s` | switch the mic / system-audio source |
| `M` / `S` | mute or unmute mic / system (muted audio is not transcribed) |
| `space` | pause or resume transcription |
| `n` | insert a note or heading (e.g. `## Action items`) |
| `?` | help |
| `q` / `Ctrl-C` | stop: finishes the last utterances and writes a footer |

Useful options (see `tuiscribe record --help`):

- `--mic`, `--system`: pick a device by id, node name or part of its name (see `devices`)
- `--no-mic`, `--no-system`: record one channel only
- `--model`: e.g. `base.en` (faster), `medium.en` or `distil-large-v3` (more accurate, slower)
- `--no-partials`: skip the live preview passes to save CPU
- `--gap-short 2 --gap-long 10`: pause lengths for a new paragraph or a silence marker
- `--save-audio DIR`: also keep `mic.wav` and `system.wav`
- `--no-tui`: plain line output (also used automatically when not on a terminal)
- `--no-transcribe`: meters only

Next to `meeting.md`, a `meeting.jsonl` sidecar keeps every segment with
timings. You can regenerate the Markdown from it, or transcribe saved audio again:

```sh
uv run tuiscribe render meeting.jsonl -o meeting.md --gap-long 30
uv run tuiscribe replay mic=rec/mic.wav system=rec/system.wav -o meeting.md
```

## Output

```markdown
# Meeting: 2026-11-10 14:02

**Sources:** mic = "Headset", system = "Headset" · **Model:** faster-whisper small.en

**You** · 14:02:11
Okay, can everyone see my screen?

**Remote** · 14:02:15
Yes, that's fine. Go ahead.

*⋯ 42 s of silence ⋯*

**You** · 14:02:58
So the first item is the forecast comparison.
```

## Development

```sh
uv run pytest
```
