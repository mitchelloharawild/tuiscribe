#!/usr/bin/env bash
# Re-record assets/demo/tuiscribe.gif, the README demo.
#
# It synthesises a two-voice dialogue with piper TTS, plays it into virtual
# PipeWire devices (a "mic" source and a "system audio" sink) and records
# `tuiscribe record` on them with VHS. Nothing is played on your speakers and
# your default devices are left alone.
#
# Needs: vhs and ttyd on PATH (https://github.com/charmbracelet/vhs), ffmpeg,
# uv, pactl and pw-play. gifsicle is used to shrink the GIF if installed.
#
# Usage: scripts/demo/make-demo.sh [--rebuild-audio]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CACHE="$HERE/.cache"
GIF="$REPO/assets/demo/tuiscribe.gif"
PLAY_DELAY=4.5   # seconds after setup before the dialogue starts (covers typing + model load)

for tool in vhs ttyd ffmpeg uv uvx pactl pw-play; do
    command -v "$tool" >/dev/null || { echo "make-demo: '$tool' not found on PATH" >&2; exit 1; }
done

mkdir -p "$CACHE/voices" "$(dirname "$GIF")"

# 1. TTS voices and the dialogue audio (cached; --rebuild-audio to redo).
if [[ "${1:-}" == "--rebuild-audio" ]]; then rm -f "$CACHE/audio/"*.wav; fi
if [[ ! -f "$CACHE/voices/en_US-lessac-medium.onnx" || ! -f "$CACHE/voices/en_GB-alan-medium.onnx" ]]; then
    (cd "$CACHE/voices" && uvx --from piper-tts python -m piper.download_voices en_US-lessac-medium en_GB-alan-medium)
fi
if [[ ! -f "$CACHE/audio/mic.wav" || "$HERE/build_audio.py" -nt "$CACHE/audio/mic.wav" ]]; then
    python3 "$HERE/build_audio.py" "$CACHE/audio" "$CACHE/voices"
fi

# 2. The working tree's tuiscribe, with the model already downloaded.
uv sync --project "$REPO" --quiet
"$REPO/.venv/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

# 3. Virtual devices: a sink whose monitor is remapped into a source (the mic),
#    and a plain sink (system audio). Unloaded again on exit.
MODULES=()
cleanup() {
    for ((i = ${#MODULES[@]} - 1; i >= 0; i--)); do pactl unload-module "${MODULES[i]}" 2>/dev/null || true; done
    rm -rf "${SESSION:-}"
}
trap cleanup EXIT
MODULES+=("$(pactl load-module module-null-sink sink_name=tuiscribe-demo-mic-in sink_properties=device.description=Demo-mic-in)")
MODULES+=("$(pactl load-module module-remap-source master=tuiscribe-demo-mic-in.monitor source_name=tuiscribe-demo-mic source_properties=device.description=Demo-mic)")
MODULES+=("$(pactl load-module module-null-sink sink_name=tuiscribe-demo-sys sink_properties=device.description=Demo-system)")

# 4. Record. The tape types `tuiscribe record -o standup.md` in a throwaway dir.
SESSION="$(mktemp -d)"
PLAY="$SESSION/.play.sh"
cat >"$PLAY" <<EOF
#!/usr/bin/env bash
sleep "\$1"
pw-play --target tuiscribe-demo-mic-in "$CACHE/audio/mic.wav" &
pw-play --target tuiscribe-demo-sys "$CACHE/audio/system.wav" &
wait
EOF
chmod +x "$PLAY"

export DEMO_SESSION="$SESSION" DEMO_TUISCRIBE="$REPO/.venv/bin/tuiscribe" DEMO_PLAY="$PLAY" DEMO_PLAY_DELAY="$PLAY_DELAY"
(cd "$REPO" && vhs "$HERE/demo.tape")

# 5. Shrink.
if command -v gifsicle >/dev/null; then
    gifsicle -O3 --lossy=60 --batch "$GIF"
fi
echo "wrote $GIF ($(du -h "$GIF" | cut -f1))"
