from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator, Sequence

from torch.utils.data import BatchSampler, Sampler


class GroupedBatchSampler(BatchSampler):
    """Batch sampled indices without mixing different group IDs.

    The indices inside each group retain the order produced by the underlying
    sampler.  Completed group batches are then ordered by the first sampled
    position, matching the behavior used by maskrcnn-benchmark's aspect-ratio
    grouping without coupling this project to that package.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        group_ids: Sequence[int],
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        data_source = getattr(sampler, "data_source", None)
        if data_source is not None and len(group_ids) != len(data_source):
            raise ValueError("group_ids must contain one entry per dataset sample")
        self.sampler = sampler
        self.group_ids = tuple(int(group_id) for group_id in group_ids)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[list[int]]:
        sampled = list(self.sampler)
        positions = {index: position for position, index in enumerate(sampled)}
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in sampled:
            grouped[self.group_ids[index]].append(index)

        batches: list[list[int]] = []
        for indices in grouped.values():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        batches.sort(key=lambda batch: positions[batch[0]])
        yield from batches

    def __len__(self) -> int:
        counts: dict[int, int] = defaultdict(int)
        for group_id in self.group_ids:
            counts[group_id] += 1
        if self.drop_last:
            return sum(count // self.batch_size for count in counts.values())
        return sum(math.ceil(count / self.batch_size) for count in counts.values())
