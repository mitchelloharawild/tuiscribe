import wave

import numpy as np

from tuiscribe.audio.wav import PAD_THRESHOLD, AlignedWavWriter
from tuiscribe.config import SAMPLE_RATE


def chunk(n, v=0.5):
    return np.full(n, v, dtype=np.float32)


def read(path):
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), data


def test_format_and_contiguous(tmp_path):
    p = tmp_path / "a.wav"
    w = AlignedWavWriter(p, 10.0)
    n = 320
    for i in range(5):
        w.write(10.0 + i * n / SAMPLE_RATE, chunk(n))
    w.close()
    ch, sw, rate, data = read(p)
    assert (ch, sw, rate) == (1, 2, 16000)
    assert len(data) == 5 * n
    assert np.all(data == int(0.5 * 32767))
    assert w.seconds == 5 * n / SAMPLE_RATE


def test_gap_is_padded(tmp_path):
    p = tmp_path / "a.wav"
    w = AlignedWavWriter(p, 0.0)
    w.write(0.0, chunk(1600))
    w.write(1.0, chunk(1600))  # should start at frame 16000
    w.close()
    _, _, _, data = read(p)
    assert len(data) == 16000 + 1600
    assert np.all(data[1600:16000] == 0)
    assert np.all(data[16000:] != 0)


def test_initial_offset_padded(tmp_path):
    p = tmp_path / "a.wav"
    w = AlignedWavWriter(p, 5.0)
    w.write(5.5, chunk(100))
    w.close()
    _, _, _, data = read(p)
    assert len(data) == 8000 + 100


def test_jitter_not_padded(tmp_path):
    p = tmp_path / "a.wav"
    w = AlignedWavWriter(p, 0.0)
    w.write(0.0, chunk(1600))
    # 50 ms late: below the 100 ms threshold
    w.write(1600 / SAMPLE_RATE + 0.05, chunk(1600))
    w.close()
    _, _, _, data = read(p)
    assert len(data) == 3200


def test_exactly_threshold_not_padded(tmp_path):
    p = tmp_path / "b.wav"
    w = AlignedWavWriter(p, 0.0)
    w.write(0.0, chunk(1600))
    w.write((1600 + PAD_THRESHOLD) / SAMPLE_RATE, chunk(1600))
    w.close()
    _, _, _, data = read(p)
    assert len(data) == 3200


def test_close_idempotent_and_write_after_close(tmp_path):
    p = tmp_path / "a.wav"
    w = AlignedWavWriter(p, 0.0)
    w.write(0.0, chunk(100))
    w.close()
    w.close()
    w.write(1.0, chunk(100))  # ignored
    _, _, _, data = read(p)
    assert len(data) == 100


def test_clipping(tmp_path):
    p = tmp_path / "a.wav"
    w = AlignedWavWriter(p, 0.0)
    w.write(0.0, np.array([2.0, -2.0], dtype=np.float32))
    w.close()
    _, _, _, data = read(p)
    assert list(data) == [32767, -32767]
