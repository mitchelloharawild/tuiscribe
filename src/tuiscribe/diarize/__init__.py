"""Who is speaking within a channel: "Remote 1", "Remote 2", ...

Channels are already split by source (mic vs system audio), so diarization
only separates voices *within* a channel. Each finished utterance gets a
speaker embedding and a live label from online clustering; when the session
ends, `final_labels()` re-clusters everything at once and the transcript is
rewritten with the corrected labels.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .cluster import DEFAULT_THRESHOLD, OnlineClusterer, cluster_all
from .embed import DEFAULT_MODEL, SpeakerEmbedder

# Label stem per channel when it is diarized. The mic may hear several people
# in the room, so its speakers are "Local N" rather than "You".
DIARIZED_LABEL = {"mic": "Local", "system": "Remote"}

Key = tuple[str, float]  # (channel, segment start) identifies a segment


def speaker_label(channel: str, index: int) -> str:
    return f"{DIARIZED_LABEL.get(channel, channel)} {index + 1}"


class Diarizer:
    def __init__(
        self,
        channels: Iterable[str],
        embedder: SpeakerEmbedder | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.channels = set(channels)
        self.embedder = embedder or SpeakerEmbedder(DEFAULT_MODEL)
        self.threshold = threshold
        self._online = {ch: OnlineClusterer(threshold) for ch in self.channels}
        self._history: dict[str, list[tuple[Key, np.ndarray | None, float]]] = {ch: [] for ch in self.channels}

    def applies(self, channel: str) -> bool:
        return channel in self.channels

    def load(self) -> None:
        self.embedder.load()

    def label(self, key: Key, audio: np.ndarray, duration: float) -> str:
        """Live label for one finished utterance of `key[0]` (STT thread)."""
        ch = key[0]
        emb = self.embedder.embed(audio)
        index = self._online[ch].assign(emb, duration)
        self._history[ch].append((key, emb, duration))
        return speaker_label(ch, index)

    def speakers(self, channel: str) -> int:
        return self._online[channel].n_speakers if channel in self._online else 0

    def final_labels(self) -> dict[Key, str]:
        """Labels from clustering the whole session at once."""
        out: dict[Key, str] = {}
        for ch, hist in self._history.items():
            if not hist:
                continue
            labels = cluster_all([e for _, e, _ in hist], [d for _, _, d in hist], self.threshold)
            for (key, _, _), k in zip(hist, labels):
                out[key] = speaker_label(ch, k)
        return out
