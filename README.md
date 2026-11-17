<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/tuiscribe-lockup-dark.svg">
    <img src="assets/logo/tuiscribe-lockup.svg" alt="tuiscribe" width="356">
  </picture>
</h1>

Live meeting transcription in the terminal, for Linux with PipeWire.

<p align="center">
  <img src="assets/demo/tuiscribe.gif" alt="tuiscribe recording a two-person call: input meters, live partial transcripts that settle into You and Remote lines, a silence marker, an added note, then a clean stop" width="800">
</p>

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
| `d` | speaker labels on or off for an input; name speakers |
| `q` | stop: finishes the last utterances, then exits |

Other commands:

```sh
tuiscribe replay mic=standup.mic.wav system=standup.system.wav -o redo.md --model medium.en
tuiscribe render standup.jsonl -o standup.md     # rebuild the Markdown from the .jsonl sidecar
tuiscribe diarize standup.md --diarize mic       # add speaker labels afterwards, from the saved audio
tuiscribe voices add Simon standup.md -l "Local 2"  # remember a voice, so later sessions name Simon
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

Speaker labels tell apart voices *within* one input: several people on a call
(`system`: Remote 1, Remote 2, ...) or in the room with you (`mic`: Local 1,
Local 2, ...). Each utterance gets a voiceprint from a small speaker model
(3D-Speaker CAM++, 28 MB, cached in `~/.cache/tuiscribe/models`).

- `--diarize system` (or `mic`, or `all`) switches them on from the start.
- `d` in the TUI switches them on or off for an input during the session.
- `tuiscribe diarize` adds them to a finished transcript, from its saved audio,
  without transcribing again. It rewrites the transcript in place, or writes to
  `-o` (required if you have edited the `.md` by hand).

Speakers are labelled live. When you stop, the whole session is grouped again and
the transcript is rewritten with the corrected labels. An input that is on at that
point is labelled for the whole session, including speech from before you
switched it on (this uses the saved audio, so not with `--no-save-audio`). An
input switched off again gets no labels.

A very short utterance ("yeah") takes the nearest or the previous speaker. If one
person is split into two speakers, lower `--speaker-threshold`; if two people are
merged into one, raise it.

### Known speakers

Name someone during a session (`d`, then *Name speakers…*) and their voice is
saved when you stop, so later sessions name them instead of numbering them. The
same list lets you confirm a saved voice, which adds this session's speech to it.
Or save a voice from a finished transcript where their label is right:

```sh
tuiscribe voices add Simon standup.md --label "Local 2"   # also relabels Local 2 as Simon there
tuiscribe voices list
tuiscribe voices remove Simon
```

Saved people are named live from their first clear utterance, and newcomers are
numbered. When you stop, a speaker group is named when it clearly sounds like one
saved person (and more like them than anyone else saved); each name is used for
at most one group, and the rest stay numbered. Voices are only learned from labels you confirm, never
from automatic matches. Adding more transcripts of the same person makes the match
more reliable, especially across rooms and microphones: recordings from the same
input (mic or system) are compared first, and newer ones count for more. Each
person keeps up to 300 samples (the oldest are dropped).

Voices are speaker embeddings (biometric data) and stay on this machine, in
`~/.local/share/tuiscribe/voices` (or `$TUISCRIBE_VOICES_DIR`), one file per person.
Ask people before saving theirs. `--no-voices` turns name matching off.

## Development

```sh
uv sync
uv run pytest
```
