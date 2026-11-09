"""Order finals from both channels by start time before they are written.

STT finishes utterances out of order (a short mic remark can finish while a
long remote utterance that started earlier is still being spoken). Items are
held until the *watermark* passes their start: the watermark is the earliest
start of any speech that has not been finalised yet on any channel. Pure: the
pipeline computes the watermark and calls `release`.
"""

from __future__ import annotations

import heapq
import itertools

from .records import Item


class Merger:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Item]] = []
        self._counter = itertools.count()

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, item: Item) -> None:
        heapq.heappush(self._heap, (item.start, next(self._counter), item))

    def release(self, watermark: float) -> list[Item]:
        """Pop items starting at or before `watermark`, in start order."""
        out: list[Item] = []
        while self._heap and self._heap[0][0] <= watermark:
            out.append(heapq.heappop(self._heap)[2])
        return out

    def release_all(self) -> list[Item]:
        return self.release(float("inf"))
