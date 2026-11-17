"""Synthesise the demo dialogue into two time-aligned tracks: mic.wav and system.wav.

The text is "The North Wind and the Sun", the standard passage used in
phonetics, split between "You" (mic) and "Remote" (system audio). Each line is
spoken with piper TTS in its own voice and placed on its channel's track, with
silence on the other track, so playing both files together sounds like a call.

Usage: python3 build_audio.py OUT_DIR VOICE_DIR
"""

from __future__ import annotations

import array
import random
import subprocess
import sys
import wave
from pathlib import Path

VOICES = {"mic": "en_US-lessac-medium", "system": "en_GB-alan-medium"}

# (channel, text, seconds of silence after it)
DIALOGUE = [
    ("mic", "The North Wind and the Sun were disputing which was the stronger, "
            "when a traveler came along wrapped in a warm cloak.", 0.8),
    ("system", "They agreed that the one who first succeeded in making the traveler "
               "take his cloak off should be considered stronger than the other.", 0.8),
    ("mic", "Then the North Wind blew as hard as he could, but the more he blew the "
            "more closely did the traveler fold his cloak around him; and at last "
            "the North Wind gave up the attempt.", 4.5),
    ("system", "Then the Sun shined out warmly, and immediately the traveler took off his cloak.", 0.8),
    ("mic", "And so the North Wind was obliged to confess that the Sun was the stronger of the two.", 0.5),
]
LEAD_IN = 0.5  # seconds of silence before the first line
# A faint noise floor (about -70 dBFS), like a real room or call. A virtual sink
# otherwise outputs exact zeros, which tuiscribe flags as a muted device.
NOISE_SD = 10.0
TAIL = 15.0  # seconds of noise floor after the last line, while the note is typed and tuiscribe stops


def synth(text: str, voice: str, voice_dir: Path, out: Path) -> None:
    subprocess.run(
        ["uvx", "--from", "piper-tts", "piper", "-m", str(voice_dir / f"{voice}.onnx"),
         "-f", str(out), "--length-scale", "0.95"],
        input=text.encode(), check=True, capture_output=True,
    )


def main() -> None:
    out_dir, voice_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = {ch: bytearray() for ch in VOICES}
    rate = None
    for i, (channel, text, pause) in enumerate(DIALOGUE):
        clip = out_dir / f"line{i}.wav"
        synth(text, VOICES[channel], voice_dir, clip)
        with wave.open(str(clip), "rb") as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2
            if rate is None:
                rate = w.getframerate()
                for t in tracks.values():
                    t.extend(bytes(2 * int(LEAD_IN * rate)))
            assert w.getframerate() == rate, "both voices must share a sample rate"
            speech = w.readframes(w.getnframes())
        clip.unlink()
        gap = bytes(2 * int(pause * rate))
        for ch, t in tracks.items():
            t.extend(speech if ch == channel else bytes(len(speech)))
            t.extend(gap)
    for t in tracks.values():
        t.extend(bytes(2 * int(TAIL * rate)))
    rng = random.Random(0)
    for ch, data in tracks.items():
        samples = array.array("h", bytes(data))
        for i, v in enumerate(samples):
            samples[i] = max(-32768, min(32767, v + round(rng.gauss(0.0, NOISE_SD))))
        data = samples.tobytes()
        with wave.open(str(out_dir / f"{ch}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(data))
    print(f"{len(tracks['mic']) / 2 / rate - TAIL:.1f} s of dialogue (+{TAIL:.0f} s tail) in {out_dir}")


if __name__ == "__main__":
    main()
