"""Who is speaking within a channel: "Remote 1", "Remote 2", ...

Channels are already split by source (mic vs system audio), so diarization
only separates voices *within* a channel. While a channel is diarized, each
finished utterance gets a speaker embedding and a live label from online
clustering. Channels can be switched on and off during a session. When the
session ends, `final_labels()` re-clusters the whole session at once. Segments
that were never embedded live (said before the channel was switched on) are
embedded then from the saved audio, so the labels cover the whole session.

With a `VoiceBank` of known voices, a group that clearly sounds like a saved
person is labelled with their name instead of "Local N" (see `voices`). Live,
the known voices are the first speakers of each channel's online clustering,
so a known person is named from their first clear utterance.

A live speaker can also be named by hand (`name_speaker`). That name is
confirmed: it goes to whichever final group holds most of that live speaker's
speech, and `learn()` saves that group's utterances to the person's voice.
Only confirmed names are learned, never automatic matches.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from ..records import Segment
from .cluster import DEFAULT_THRESHOLD, OnlineClusterer, cluster_all
from .embed import DEFAULT_MODEL, SpeakerEmbedder
from .voices import VoiceBank, centroid_of

# Label stem per channel when it is diarized. The mic may hear several people
# in the room, so its speakers are "Local N" rather than "You".
DIARIZED_LABEL = {"mic": "Local", "system": "Remote"}

@dataclass(frozen=True)
class LiveSpeaker:
    channel: str
    index: int  # live speaker number, for `name_speaker`
    label: str  # as shown now
    seconds: float  # clear speech so far this session
    confirmed: bool  # named by hand
    suggestion: tuple[str, float] | None  # closest saved voice and its similarity


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
        self._seeds: dict[str, list[str]] = {}  # channel -> names of its known-voice live speakers
        self._live_index: dict[str, dict[Key, int]] = {}  # channel -> segment -> live speaker
        self._named: dict[str, dict[int, str]] = {}  # channel -> live speaker -> confirmed name
        self.confirmed: dict[str, set[str]] = {}  # set by final_labels: channel -> hand-given names used
        for ch in channels:
            self.set_enabled(ch, True)

    def applies(self, channel: str) -> bool:
        return channel in self.channels

    def set_enabled(self, channel: str, on: bool) -> None:
        """Switch a channel on or off (any thread). Its live voiceprints are kept
        while it is off, so switching back on carries on with the same speakers."""
        with self._lock:
            if on:
                if channel not in self._online:
                    names, cents = [], []
                    for name in self.voices.names() if self.voices else []:
                        c = self.voices.get(name).centroid(channel)
                        if c is not None:
                            names.append(name)
                            cents.append(c)
                    self._seeds[channel] = names
                    self._online[channel] = OnlineClusterer(self.threshold, seeds=cents)
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
            self._live_index.setdefault(ch, {})[key] = index
            return self._live_label(ch, index)

    def _live_label(self, ch: str, index: int) -> str:
        if named := self._named.get(ch, {}).get(index):
            return named
        seeds = self._seeds.get(ch, [])
        return seeds[index] if index < len(seeds) else speaker_label(ch, index - len(seeds))

    def live_speakers(self) -> list[LiveSpeaker]:
        """Speakers heard so far on the diarized channels, for naming them."""
        out = []
        with self._lock:
            for ch in sorted(self.channels):
                o = self._online.get(ch)
                heard = set(self._live_index.get(ch, {}).values())
                for k in range(o.n_speakers if o else 0):
                    if k not in heard:
                        continue
                    suggestion = None
                    if self.voices:
                        scores = self.voices.scores(o.centroid(k), ch)
                        if scores:
                            best = max(scores, key=scores.get)
                            suggestion = (best, scores[best])
                    out.append(LiveSpeaker(ch, k, self._live_label(ch, k), o.seconds(k), k in self._named.get(ch, {}), suggestion))
        return out

    def name_speaker(self, channel: str, index: int, name: str | None) -> None:
        """Confirm live speaker `index` on `channel` as `name` (None: forget the name)."""
        with self._lock:
            named = self._named.setdefault(channel, {})
            if name:
                named[index] = name
            else:
                named.pop(index, None)

    def speakers(self, channel: str) -> int:
        """Speakers heard live on `channel` (known voices count once heard)."""
        o = self._online.get(channel)
        return sum(1 for k in range(o.n_speakers) if k >= o.n_seeds or o.seconds(k) > 0) if o else 0

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
        confirmed = self._confirmed_groups(ch, entries, labels)
        matched: dict = {}
        if self.voices:
            cents = {}
            for k in sorted(set(labels) - set(confirmed)):
                members = [(e, d) for (_, e, d), lab in zip(entries, labels) if lab == k]
                c = centroid_of([e for e, _ in members], [d for _, d in members])
                if c is not None:
                    cents[k] = c
            matched = self.voices.match(cents, ch, exclude=confirmed.values())
        names: dict[int, str] = {}
        n = 0
        for k in labels:  # first-appearance order
            if k in names:
                continue
            if k in confirmed:
                names[k] = confirmed[k]
            elif k in matched:
                names[k] = matched[k][0]
            else:
                names[k] = speaker_label(ch, n)
                n += 1
        self.confirmed[ch] = set(confirmed.values())
        return names, {name: score for name, score in matched.values()}

    def _confirmed_groups(
        self, ch: str, entries: list[tuple[Key, np.ndarray | None, float]], labels: list[int]
    ) -> dict[int, str]:
        """Final group -> hand-given name: each named live speaker's name goes to the
        group holding most of its speech (a group named twice keeps the larger share)."""
        with self._lock:
            named = dict(self._named.get(ch, {}))
            live = dict(self._live_index.get(ch, {}))
        best: dict[int, tuple[float, str]] = {}
        for index, name in named.items():
            share: dict[int, float] = {}
            for (key, _, dur), k in zip(entries, labels):
                if live.get(key) == index:
                    share[k] = share.get(k, 0.0) + dur
            if not share:
                continue
            k = max(share, key=share.get)
            if share[k] > best.get(k, (0.0, ""))[0]:
                best[k] = (share[k], name)
        return {k: name for k, (_, name) in best.items()}

    def learn(self, source: str) -> dict[str, int]:
        """Save the utterances of each confirmed speaker (from the last `final_labels`)
        to their voice. Returns {name: samples added}."""
        if self.voices is None:
            return {}
        added: dict[str, int] = {}
        for ch, names in self.confirmed.items():
            for name in names:
                entries = self.groups.get(ch, {}).get(name, [])
                n = self.voices.add(name, [(e, key[1], key[1] + dur) for key, e, dur in entries], ch, source)
                added[name] = added.get(name, 0) + n
        return added
