"""Known voices: speaker embeddings saved per person, matched against new sessions.

Each person is one `.npz` file in `voices_dir()` (default
`~/.local/share/tuiscribe/voices`, or `$TUISCRIBE_VOICES_DIR`). A file holds
that person's samples: one embedding per utterance, with its duration, input
channel, where it came from (transcript path and times) and when it was added.
Samples are only ever added from labels a person confirmed, so a wrong
automatic match is never learned.

Matching compares a speaker group's centroid (the duration-weighted mean of
its embeddings) with each person's centroid. A person's centroid is built
from their samples on the same input when there is enough of them (the mic
and call audio sound different), weighting newer samples more. A group is
named only if the best person reaches `MATCH_THRESHOLD` and beats the next
best by `MATCH_MARGIN`; each person names at most one group.

Embeddings from different models are not comparable, so a voice saved with
another model is ignored (and reported by `tuiscribe voices list`).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cluster import MIN_RELIABLE_S
from .embed import DEFAULT_MODEL

MATCH_THRESHOLD = 0.7  # centroid cosine similarity needed to name a group
MATCH_MARGIN = 0.1  # ... and how far ahead of the next-best person it must be
MAX_SAMPLES = 300  # per person; the oldest are dropped first
SAME_INPUT_MIN_S = 10.0  # speech on an input before that input's samples are used alone
HALF_LIFE_DAYS = 180.0  # a sample's weight halves over this age


def voices_dir() -> Path:
    if env := os.environ.get("TUISCRIBE_VOICES_DIR"):
        return Path(env)
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "tuiscribe" / "voices"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "voice"


@dataclass
class Voice:
    name: str
    model: str
    emb: np.ndarray  # (n, dim) float32, L2-normalised
    duration: np.ndarray  # (n,) seconds
    channel: np.ndarray  # (n,) str
    source: np.ndarray  # (n,) str: transcript the sample came from
    start: np.ndarray  # (n,) session seconds
    end: np.ndarray
    added: np.ndarray  # (n,) unix time

    @classmethod
    def empty(cls, name: str, model: str, dim: int) -> Voice:
        return cls(
            name, model, np.zeros((0, dim), np.float32), np.zeros(0), np.zeros(0, "U16"),
            np.zeros(0, "U1"), np.zeros(0), np.zeros(0), np.zeros(0),
        )

    def __len__(self) -> int:
        return len(self.duration)

    @property
    def seconds(self) -> float:
        return float(self.duration.sum())

    def centroid(self, channel: str | None = None, now: float | None = None) -> np.ndarray | None:
        """Weighted mean embedding: this input's samples if there is enough of
        them, else all. Newer and longer samples count more."""
        if not len(self):
            return None
        mask = np.ones(len(self), bool)
        if channel is not None:
            same = self.channel == channel
            if self.duration[same].sum() >= SAME_INPUT_MIN_S:
                mask = same
        age_days = ((now or time.time()) - self.added[mask]) / 86400.0
        w = self.duration[mask] * 0.5 ** (np.maximum(age_days, 0) / HALF_LIFE_DAYS)
        c = (self.emb[mask] * w[:, None]).sum(0)
        norm = float(np.linalg.norm(c))
        return c / norm if norm > 0 else None


class VoiceBank:
    """The saved voices in a directory. Reads every file on construction."""

    def __init__(self, directory: Path | None = None, model: str = DEFAULT_MODEL) -> None:
        self.dir = directory or voices_dir()
        self.model = model
        self.voices: dict[str, Voice] = {}  # every voice file, whatever its model
        self.unreadable: list[Path] = []
        self._paths: dict[str, Path] = {}
        self._centroids: dict[tuple[str, str | None], np.ndarray | None] = {}  # (name, channel)
        if self.dir.is_dir():
            for path in sorted(p for p in self.dir.glob("*.npz") if not p.name.endswith(".part.npz")):
                try:
                    v = _read(path)
                except (OSError, ValueError, KeyError):
                    self.unreadable.append(path)
                    continue
                self.voices[v.name] = v
                self._paths[v.name] = path

    def names(self) -> list[str]:
        """People that can be matched (saved with this model)."""
        return sorted(n for n, v in self.voices.items() if v.model == self.model and len(v))

    def __bool__(self) -> bool:
        return bool(self.names())

    def get(self, name: str) -> Voice | None:
        return self.voices.get(name)

    def add(
        self,
        name: str,
        samples: Iterable[tuple[np.ndarray | None, float, float]],  # (embedding, start, end)
        channel: str,
        source: str,
    ) -> int:
        """Save samples for `name` and return how many were new. Samples shorter
        than `MIN_RELIABLE_S`, without an embedding, or already saved from the
        same source, channel and time are skipped."""
        rows = [(e, s, t) for e, s, t in samples if e is not None and t - s >= MIN_RELIABLE_S]
        v = self.voices.get(name)
        if v is not None and v.model != self.model:
            raise ValueError(f"{name} was saved with the {v.model} speaker model; remove it first")
        if not rows and v is None:
            return 0
        dim = len(rows[0][0]) if rows else v.emb.shape[1]
        v = v or Voice.empty(name, self.model, dim)
        have = {(src, ch, round(float(st), 3)) for src, ch, st in zip(v.source, v.channel, v.start)}
        new = [(e, s, t) for e, s, t in rows if (source, channel, round(s, 3)) not in have]
        if not new:
            return 0
        now = time.time()
        v = Voice(
            name,
            v.model,
            np.concatenate([v.emb, np.stack([e for e, _, _ in new]).astype(np.float32)]),
            np.concatenate([v.duration, [t - s for _, s, t in new]]),
            np.concatenate([v.channel, [channel] * len(new)]),
            np.concatenate([v.source, [source] * len(new)]),
            np.concatenate([v.start, [s for _, s, _ in new]]),
            np.concatenate([v.end, [t for _, _, t in new]]),
            np.concatenate([v.added, [now] * len(new)]),
        )
        if len(v) > MAX_SAMPLES:  # drop the oldest (stable: earlier rows of equal age go first)
            keep = np.sort(np.argsort(-v.added, kind="stable")[:MAX_SAMPLES])
            v = Voice(v.name, v.model, *(a[keep] for a in (v.emb, v.duration, v.channel, v.source, v.start, v.end, v.added)))
        self._write(v)
        self.voices[name] = v
        self._centroids.clear()
        return len(new)

    def remove(self, name: str) -> bool:
        v = self.voices.pop(name, None)
        if v is None:
            return False
        self._centroids.clear()
        path = self._paths.pop(name, None)
        if path is not None:
            path.unlink(missing_ok=True)
        return True

    def scores(self, centroid: np.ndarray, channel: str | None = None) -> dict[str, float]:
        """Cosine similarity of a group centroid with each matchable person."""
        out = {}
        for n in self.names():
            if (n, channel) not in self._centroids:
                self._centroids[(n, channel)] = self.voices[n].centroid(channel)
            c = self._centroids[(n, channel)]
            if c is not None and c.shape == centroid.shape:
                out[n] = float(c @ centroid)
        return out

    def match(
        self,
        centroids: Mapping[Hashable, np.ndarray],
        channel: str | None = None,
        exclude: Iterable[str] = (),
        threshold: float = MATCH_THRESHOLD,
        margin: float = MATCH_MARGIN,
    ) -> dict[Hashable, tuple[str, float]]:
        """Name the groups that clearly sound like one saved person: {group: (name, score)}.
        People in `exclude` (already named in this session) are not candidates."""
        excluded = set(exclude)
        wants = []
        for k, c in centroids.items():
            s = {n: v for n, v in self.scores(c, channel).items() if n not in excluded}
            if not s:
                continue
            ranked = sorted(s.values(), reverse=True)
            best = max(s, key=s.get)
            second = ranked[1] if len(ranked) > 1 else -1.0
            if s[best] >= threshold and s[best] - second >= margin:
                wants.append((s[best], k, best))
        out: dict[Hashable, tuple[str, float]] = {}
        taken: set[str] = set()
        for score, k, name in sorted(wants, key=lambda w: -w[0]):  # each person names one group
            if name not in taken:
                out[k] = (name, score)
                taken.add(name)
        return out

    def _path(self, name: str) -> Path:
        """The voice's file: the one it was read from, else a new one named after it."""
        if name in self._paths:
            return self._paths[name]
        taken = set(self._paths.values())
        path, n = self.dir / f"{_slug(name)}.npz", 1
        while path in taken or path.exists():
            n += 1
            path = self.dir / f"{_slug(name)}-{n}.npz"
        self._paths[name] = path
        return path

    def _write(self, v: Voice) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(v.name)
        tmp = path.with_name(path.stem + ".part.npz")
        np.savez(
            tmp, name=np.array(v.name), model=np.array(v.model), emb=v.emb, duration=v.duration,
            channel=v.channel.astype(str), source=v.source.astype(str), start=v.start, end=v.end, added=v.added,
        )
        tmp.replace(path)


def _read(path: Path) -> Voice:
    with np.load(path, allow_pickle=False) as f:
        return Voice(
            str(f["name"]), str(f["model"]), f["emb"].astype(np.float32), f["duration"], f["channel"],
            f["source"], f["start"], f["end"], f["added"],
        )


def centroid_of(embs: Iterable[np.ndarray | None], durations: Iterable[float]) -> np.ndarray | None:
    """Duration-weighted mean of reliable embeddings (falling back to all), normalised."""
    pairs = [(e, d) for e, d in zip(embs, durations) if e is not None]
    if not pairs:
        return None
    reliable = [(e, d) for e, d in pairs if d >= MIN_RELIABLE_S] or pairs
    c = sum(e * d for e, d in reliable)
    norm = float(np.linalg.norm(c))
    return c / norm if norm > 0 else None
