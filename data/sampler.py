"""Shuffle complete sessions and balance token workload across GPUs."""
import torch
from torch.utils.data import Sampler


def balanced_partition(indices, lengths, world_size, quotas):
    assigned = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for index in sorted(indices, key=lambda i: lengths[i], reverse=True):
        rank = min((r for r in range(world_size) if len(assigned[r]) < quotas[r]),
                   key=lambda r: (loads[r], len(assigned[r]), r))
        assigned[rank].append(index)
        loads[rank] += lengths[index]
    return assigned


class ShuffledEpochSampler(Sampler):
    """Keep each shuffled optimizer batch intact and balance its rank workloads."""
    def __init__(self, dataset, rank, world_size, seed, accumulation):
        self.size = len(dataset)
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.accumulation = accumulation
        self.lengths = [entry['tokens'] for entry in dataset.entries]

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        sequence = torch.randperm(self.size, generator=generator).tolist()
        batch_size = self.world_size * self.accumulation
        assigned = []
        for start in range(0, self.size, batch_size):
            batch = sequence[start:start + batch_size]
            quotas = [len(range(rank, len(batch), self.world_size)) for rank in range(self.world_size)]
            assigned.extend(balanced_partition(batch, self.lengths, self.world_size, quotas)[self.rank])
        return iter(assigned)


def validation_partition(dataset, world_size):
    lengths = [entry['tokens'] for entry in dataset.entries]
    quotas = [len(dataset)] * world_size
    return balanced_partition(range(len(dataset)), lengths, world_size, quotas)
