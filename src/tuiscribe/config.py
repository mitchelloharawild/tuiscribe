"""Shared constants. Everything downstream of capture assumes this format."""

SAMPLE_RATE = 16_000  # Hz, mono float32 (what Whisper and Silero VAD expect)
CHUNK_MS = 20
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000  # 320 samples per capture chunk
BYTES_PER_SAMPLE = 4  # float32

CHANNELS = ("mic", "system")
CHANNEL_LABELS = {"mic": "MIC", "system": "SYSTEM"}

# Level meters
METER_FPS = 20
SILENCE_DB = -60.0  # below this a channel counts as silent for "silent for N s"
DIGITAL_SILENCE = 1e-7  # peak below this is exact zeros -> likely wrong/muted device
