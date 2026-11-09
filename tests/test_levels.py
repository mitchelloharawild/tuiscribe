import math

import numpy as np

from tuiscribe.capture.levels import FLOOR_DB, LevelAccumulator, to_db


def test_to_db():
    assert to_db(1.0) == 0.0
    assert to_db(0.0) == FLOOR_DB
    assert math.isclose(to_db(0.5), -6.0206, abs_tol=1e-3)


def test_empty_returns_none():
    acc = LevelAccumulator()
    assert acc.take() is None
    acc.add(np.zeros(0, dtype=np.float32))
    assert acc.take() is None


def test_constant_signal_across_adds():
    acc = LevelAccumulator()
    for _ in range(3):
        acc.add(np.full(100, 0.5, dtype=np.float32))
    lvl = acc.take()
    assert lvl.samples == 300
    assert lvl.peak == 0.5
    assert math.isclose(lvl.rms_db, to_db(0.5), abs_tol=1e-4)
    assert math.isclose(lvl.peak_db, to_db(0.5), abs_tol=1e-4)


def test_sine_rms_and_peak():
    t = np.arange(16000) / 16000
    s = (0.8 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    acc = LevelAccumulator()
    for i in range(0, len(s), 1600):
        acc.add(s[i : i + 1600])
    lvl = acc.take()
    assert math.isclose(lvl.rms_db, to_db(0.8 / math.sqrt(2)), abs_tol=0.01)
    assert math.isclose(lvl.peak_db, to_db(0.8), abs_tol=0.01)


def test_peak_uses_abs_and_max_across_adds():
    acc = LevelAccumulator()
    acc.add(np.array([0.1, -0.9], dtype=np.float32))
    acc.add(np.array([0.3], dtype=np.float32))
    assert acc.take().peak == np.float32(0.9)


def test_resets_after_take():
    acc = LevelAccumulator()
    acc.add(np.full(10, 1.0, dtype=np.float32))
    assert acc.take() is not None
    assert acc.take() is None
    acc.add(np.full(10, 0.1, dtype=np.float32))
    lvl = acc.take()
    assert lvl.samples == 10 and lvl.peak < 0.2
