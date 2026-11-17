"""Who is speaking within a channel: "Remote 1", "Remote 2", ...

Channels are already split by source (mic vs system audio), so diarization
only separates voices *within* a channel. While a channel is diarized, each
finished utterance gets a speaker embedding and a live label from online
clustering. Channels can be switched on and off during a session. When the
session ends, `final_labels()` re-clusters the whole session at once. Segments
that were never embedded live (said before the channel was switched on) are
embedded then from the saved audio, so the labels cover the whole session.

With a `VoiceBank` of known voices, a group that clearly sounds like a saved
person is labelled with their name instead of "Local N" (see `voices`).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

import numpy as np

from ..records import Segment
from .cluster import DEFAULT_THRESHOLD, OnlineClusterer, cluster_all
from .embed import DEFAULT_MODEL, SpeakerEmbedder
from .voices import VoiceBank, centroid_of

# Label stem per channel when it is diarized. The mic may hear several people
# in the room, so its speakers are "Local N" rather than "You".
DIARIZED_LABEL = {"mic": "Local", "system": "Remote"}

Key = tuple[str, float]  # (channel, segment start) identifies a segment
# (channel, start, end) -> that stretch of 16 kHz mono audio, or None if unavailable
AudioSource = Callable[[str, float, float], "np.ndarray | None"]


def speaker_label(channel: str, index: int) -> str:
    return f"{DIARIZED_LABEL.get(channel, channel)} {index + 1}"


class Diarizer:
    def __init__(
        self,
        channels: Iterable[str] = (),
        embedder: SpeakerEmbedder | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        voices: VoiceBank | None = None,
    ) -> None:
        self.embedder = embedder or SpeakerEmbedder(DEFAULT_MODEL)
        self.threshold = threshold
        self.voices = voices
        # set by final_labels: channel -> label -> that group's (key, embedding, duration)
        self.groups: dict[str, dict[str, list[tuple[Key, np.ndarray | None, float]]]] = {}
        # channel -> label -> similarity, for labels that came from a known voice
        self.matched: dict[str, dict[str, float]] = {}
        self.channels: set[str] = set()
        self.ready = False  # model loaded
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._online: dict[str, OnlineClusterer] = {}
        self._history: dict[str, list[tuple[Key, np.ndarray | None, float]]] = {}
        for ch in channels:
            self.set_enabled(ch, True)

    def applies(self, channel: str) -> bool:
        return channel in self.channels

    def set_enabled(self, channel: str, on: bool) -> None:
        """Switch a channel on or off (any thread). Its live voiceprints are kept
        while it is off, so switching back on carries on with the same speakers."""
        with self._lock:
            if on:
                self._online.setdefault(channel, OnlineClusterer(self.threshold))
                self._history.setdefault(channel, [])
                self.channels = self.channels | {channel}
            else:
                self.channels = self.channels - {channel}

    def load(self) -> None:
        """Load the speaker model (downloads it the first time). Slow; idempotent."""
        with self._load_lock:
            if not self.ready:
                self.embedder.load()
                self.ready = True

    def label(self, key: Key, audio: np.ndarray, duration: float) -> str:
        """Live label for one finished utterance of `key[0]` (STT thread)."""
        ch = key[0]
        emb = self.embedder.embed(audio)
        with self._lock:
            index = self._online[ch].assign(emb, duration)
            self._history[ch].append((key, emb, duration))
        return speaker_label(ch, index)

    def speakers(self, channel: str) -> int:
        return self._online[channel].n_speakers if channel in self._online else 0

    def missing(self, segments: Iterable[Segment]) -> int:
        """How many of these segments on diarized channels have no live embedding."""
        with self._lock:
            known = {key for hist in self._history.values() for key, _, _ in hist}
            chans = self.channels
        return sum(1 for s in segments if s.channel in chans and (s.channel, s.start) not in known)

    def final_labels(
        self, segments: Iterable[Segment] | None = None, audio: AudioSource | None = None
    ) -> dict[Key, str]:
        """Labels from clustering the whole session at once.

        Without `segments`: every utterance labelled live on a diarized channel.
        With `segments`: those on diarized channels. The ones not embedded live are
        embedded from `audio`; any that still have no audio are left out.
        """
        with self._lock:
            chans = set(self.channels)
            history = {ch: list(self._history.get(ch, [])) for ch in chans}
        out: dict[Key, str] = {}
        for ch in sorted(chans):
            if segments is None:
                entries = history[ch]
            else:
                known = {key: (emb, dur) for key, emb, dur in history[ch]}
                entries = []
                for s in segments:
                    if s.channel != ch:
                        continue
                    key = (ch, s.start)
                    if key in known:
                        entries.append((key, *known[key]))
                    elif audio is not None:
                        clip = audio(ch, s.start, s.end)
                        if clip is not None:
                            entries.append((key, self.embedder.embed(clip), s.end - s.start))
            if not entries:
                continue
            labels = cluster_all([e for _, e, _ in entries], [d for _, _, d in entries], self.threshold)
            names, scores = self._names(ch, entries, labels)
            self.groups[ch] = {}
            self.matched[ch] = scores
            for entry, k in zip(entries, labels):
                out[entry[0]] = names[k]
                self.groups[ch].setdefault(names[k], []).append(entry)
        return out

    def _names(
        self, ch: str, entries: list[tuple[Key, np.ndarray | None, float]], labels: list[int]
    ) -> tuple[dict[int, str], dict[str, float]]:
        """Label per group: a known voice's name if it clearly matches, else
        "Local N" numbered by first appearance among the unnamed groups."""
        matched: dict = {}
        if self.voices:
            cents = {}
            for k in sorted(set(labels)):
                members = [(e, d) for (_, e, d), lab in zip(entries, labels) if lab == k]
                c = centroid_of([e for e, _ in members], [d for _, d in members])
                if c is not None:
                    cents[k] = c
            matched = self.voices.match(cents, ch)
        names: dict[int, str] = {}
        n = 0
        for k in labels:  # first-appearance order
            if k in names:
                continue
            if k in matched:
                names[k] = matched[k][0]
            else:
                names[k] = speaker_label(ch, n)
                n += 1
        return names, {name: score for name, score in matched.values()}
