"""Capture one channel by running `pw-record` and reading raw f32 PCM from stdout.

PipeWire does the downmix and resampling to 16 kHz mono, so chunks arrive in
the pipeline's native format. Each chunk is passed to a callback together with
its capture time on the monotonic clock:

    t = stream_start + samples_read / SAMPLE_RATE

`stream_start` is taken when the first bytes arrive, minus the duration of that
first chunk, so both channels share one clock to within a chunk or two.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from ..config import BYTES_PER_SAMPLE, CHUNK_SAMPLES, SAMPLE_RATE
from .devices import Device

# (channel, t0_monotonic, samples) -> None. Runs on the reader thread: keep it cheap.
ChunkCallback = Callable[[str, float, np.ndarray], None]
# (channel, message) -> None, called if pw-record exits unexpectedly.
ErrorCallback = Callable[[str, str], None]


def pw_record_command(device: Device, channel: str) -> list[str]:
    props = f"{{ node.name=tuiscribe-{channel} node.description=\"tuiscribe ({channel})\""
    if device.kind == "sink":
        props += " stream.capture.sink=true"
    props += " }"
    return [
        "pw-record",
        "--raw",
        "--format", "f32",
        "--rate", str(SAMPLE_RATE),
        "--channels", "1",
        "--latency", "20ms",
        "--media-category", "Capture",
        "--media-role", "Communication",
        "--target", device.name,
        "-P", props,
        "-",
    ]  # fmt: skip


class PwRecordSource:
    """One `pw-record` process feeding one channel. Restartable with a new device."""

    def __init__(
        self,
        channel: str,
        device: Device,
        on_chunk: ChunkCallback,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self.channel = channel
        self.device = device
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.samples_read = 0
        self.stream_start: float | None = None
        self._stderr_tail: deque[str] = deque(maxlen=5)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.running:
            return
        self._stopping.clear()
        self.samples_read = 0
        self.stream_start = None
        self._proc = subprocess.Popen(
            pw_record_command(self.device, self.channel),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        self._thread = threading.Thread(
            target=self._read_loop, name=f"capture-{self.channel}", daemon=True
        )
        self._stderr_tail = deque(maxlen=5)
        self._thread.start()
        threading.Thread(
            target=self._drain_stderr, args=(self._proc, self._stderr_tail),
            name=f"capture-{self.channel}-stderr", daemon=True,
        ).start()

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen[bytes], tail: deque[str]) -> None:
        # Read continuously so a chatty pw-record can never block on a full pipe.
        assert proc.stderr is not None
        for line in iter(proc.stderr.readline, b""):
            if text := line.decode(errors="replace").strip():
                tail.append(text)

    def stop(self, timeout: float = 2.0) -> None:
        self._stopping.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
        self._proc = None
        self._thread = None

    def switch(self, device: Device) -> None:
        """Restart on a different device; the channel identity is unchanged."""
        self.stop()
        self.device = device
        self.start()

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        chunk_bytes = CHUNK_SAMPLES * BYTES_PER_SAMPLE
        buf = bytearray()
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            now = time.monotonic()
            buf += data
            usable = len(buf) - len(buf) % BYTES_PER_SAMPLE
            if usable == 0:
                continue
            samples = np.frombuffer(bytes(buf[:usable]), dtype=np.float32).copy()
            del buf[:usable]
            if self.stream_start is None:
                self.stream_start = now - len(samples) / SAMPLE_RATE
            t0 = self.stream_start + self.samples_read / SAMPLE_RATE
            self.samples_read += len(samples)
            self._on_chunk(self.channel, t0, samples)

        if not self._stopping.is_set() and self._on_error is not None:
            proc.wait()
            time.sleep(0.05)  # let the stderr drain thread catch the last line
            err = self._stderr_tail[-1] if self._stderr_tail else ""
            self._on_error(self.channel, f"pw-record exited ({proc.returncode})" + (f": {err}" if err else ""))
