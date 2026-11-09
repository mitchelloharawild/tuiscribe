# tuiscribe

Live meeting transcription in the terminal, for Linux with PipeWire.

tuiscribe records your **microphone** and your **system audio** (everything you
hear) as two separate inputs and transcribes both as you talk. It uses
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the CPU, so
nothing leaves your machine. The transcript is written to a Markdown file as the
meeting goes. Your speech is labelled **You**, the other side **Remote**, and
pauses are marked.

- A terminal UI shows input meters, the transcript as it is written and status,
  with keys to switch or mute inputs, pause, and add notes.
- The audio is saved next to the transcript, so you can transcribe it again later
  with a different model.
- Optionally, it tells apart voices within an input (`--diarize`) as Remote 1,
  Remote 2, ...

Use headphones. With speakers, the mic also picks up the other side and they get
transcribed twice.

## Install

You need Linux with PipeWire (`pw-record`, `pw-dump`) and
[uv](https://docs.astral.sh/uv/). `replay` also needs `ffmpeg`.

```sh
uv tool install git+https://github.com/mitchelloharawild/tuiscribe
```

The first run downloads the Whisper model (`small.en`, about 460 MB).

## Usage

```sh
tuiscribe devices                 # list the mics and outputs it can capture
tuiscribe record                  # -> meeting-2026-11-10-143205.md, .jsonl, .mic.wav, .system.wav
tuiscribe record -o standup.md --diarize system
```

| Key | Action |
|---|---|
| `m` / `s` | pick the mic / system-audio source |
| `M` / `S` | mute or unmute it |
| `space` | pause or resume |
| `n` | add a note or heading |
| `q` | stop: finishes the last utterances, then exits |

Other commands:

```sh
tuiscribe replay mic=standup.mic.wav system=standup.system.wav -o redo.md --model medium.en
tuiscribe render standup.jsonl -o standup.md     # rebuild the Markdown from the .jsonl sidecar
```

See `tuiscribe record --help` for all options: the model, the inputs to record,
pause lengths, `--no-tui`, `--no-save-audio` and more.

## Output

```markdown
**You** · 14:02:11
Okay, can everyone see my screen?

**Remote** · 14:02:15
Yes, that's fine. Go ahead.

*⋯ 42 s of silence ⋯*
```

## Speaker labels (prototype)

`--diarize system` (or `mic`, or `all`) gives each utterance a voiceprint from a
small speaker model (3D-Speaker CAM++, 28 MB, cached in `~/.cache/tuiscribe/models`).
Speakers are labelled live. When you stop, the whole session is grouped again and
the transcript is rewritten with the corrected labels.

The labels are numbers, not names. A very short utterance ("yeah") takes the
nearest or the previous speaker. If one person is split into two speakers, lower
`--speaker-threshold`; if two people are merged into one, raise it.

## Development

```sh
uv sync
uv run pytest
```
