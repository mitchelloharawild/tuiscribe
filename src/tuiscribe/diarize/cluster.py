"""Speaker clustering on L2-normalised embeddings (cosine similarity).

- `OnlineClusterer`: live labels. Each utterance joins the most similar
  speaker if the similarity reaches `threshold`, otherwise it starts a new
  speaker. Short utterances give unreliable embeddings, so they never start
  or move a speaker: they join one only if clearly similar, else they keep
  the previous label.
- `cluster_all`: average-linkage agglomerative clustering over a whole
  session, run once at the end to correct early live decisions.

Both return speaker indices numbered by first appearance (0, 1, ...).
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.5
MIN_RELIABLE_S = 1.0


class OnlineClusterer:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, min_reliable_s: float = MIN_RELIABLE_S) -> None:
        self.threshold = threshold
        self.min_reliable_s = min_reliable_s
        self._sums: list[np.ndarray] = []  # duration-weighted sum of embeddings per speaker
        self._last: int | None = None

    @property
    def n_speakers(self) -> int:
        return len(self._sums)

    def assign(self, emb: np.ndarray | None, duration: float) -> int:
        reliable = emb is not None and duration >= self.min_reliable_s
        if not self._sums:
            # Nothing to compare with: label it speaker 0, but only a reliable
            # embedding may found that speaker's voiceprint.
            if reliable:
                self._sums.append(emb * duration)
            return self._remember(0)
        if emb is None:
            return self._remember(self._last or 0)
        cents = np.stack([s / np.linalg.norm(s) for s in self._sums])
        sims = cents @ emb
        best = int(sims.argmax())
        if not reliable:
            return self._remember(best if sims[best] >= self.threshold else (self._last or 0))
        if sims[best] >= self.threshold:
            self._sums[best] = self._sums[best] + emb * duration
            return self._remember(best)
        self._sums.append(emb * duration)
        return self._remember(len(self._sums) - 1)

    def _remember(self, k: int) -> int:
        self._last = k
        return k


def cluster_all(
    embs: list[np.ndarray | None],
    durations: list[float],
    threshold: float = DEFAULT_THRESHOLD,
    min_reliable_s: float = MIN_RELIABLE_S,
) -> list[int]:
    """Cluster reliable utterances with AHC; attach the rest to their nearest cluster."""
    n = len(embs)
    reliable = [i for i in range(n) if embs[i] is not None and durations[i] >= min_reliable_s]
    labels = [-1] * n
    if reliable:
        X = np.stack([embs[i] for i in reliable])
        lab = np.array(_ahc(X, threshold))
        for i, k in zip(reliable, lab):
            labels[i] = int(k)
        w = np.array([durations[i] for i in reliable])[:, None]
        cents = np.stack([(X[lab == k] * w[lab == k]).sum(0) for k in range(lab.max() + 1)])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        for i in range(n):
            if labels[i] == -1 and embs[i] is not None:
                sims = cents @ embs[i]
                if sims.max() >= threshold:
                    labels[i] = int(sims.argmax())
    # still unlabelled (no embedding, or unlike everyone): previous label, else the next one, else 0
    for i in range(n):
        if labels[i] == -1 and i > 0:
            labels[i] = labels[i - 1]
    for i in reversed(range(n)):
        if labels[i] == -1:
            labels[i] = labels[i + 1] if i + 1 < n else 0
    return _by_first_appearance(labels)


def _ahc(X: np.ndarray, threshold: float) -> list[int]:
    """Average linkage on cosine similarity; merge while the best pair is >= threshold."""
    n = len(X)
    S = (X @ X.T).astype(np.float64)
    np.fill_diagonal(S, -np.inf)
    size = np.ones(n)
    owner = np.arange(n)
    for _ in range(n - 1):
        i, j = np.unravel_index(int(np.argmax(S)), S.shape)
        if S[i, j] < threshold:
            break
        merged = (S[i] * size[i] + S[j] * size[j]) / (size[i] + size[j])
        S[i, :] = merged
        S[:, i] = merged
        S[i, i] = -np.inf
        S[j, :] = -np.inf
        S[:, j] = -np.inf
        size[i] += size[j]
        owner[owner == j] = i
    return _by_first_appearance(list(owner))


def _by_first_appearance(labels: list[int]) -> list[int]:
    order: dict[int, int] = {}
    return [order.setdefault(k, len(order)) for k in labels]
