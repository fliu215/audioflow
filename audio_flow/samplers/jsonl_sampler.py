import random

from audio_flow.utils import load_jsonl


class JsonlSampler:
    r"""Jsonl Sampler."""
    def __init__(self, jsonl_paths: list[str], weights: list[float]):
        self.jsonl_paths = jsonl_paths
        self.weights = weights

        self.metas = [load_jsonl(path) for path in self.jsonl_paths]  # list[list[str]]
        self.lens = [len(metas) for metas in self.metas]  # list[str]
        self.ptrs = [0 for _ in self.lens]  # list[int]
        self.indices = [self.random_indices(L) for L in self.lens]  # list[list[int]]

    def __iter__(self) -> dict:
        r"""Random sample a jsonl file and sample a line."""
        while True:

            # Randomly sample a jsonl file
            i = random.randint(0, len(self.metas) - 1)  # jsonl index

            # Reset pointer and shuffle indices
            if self.ptrs[i] == len(self.metas[i]):
                self.indices[i] = self.random_indices(self.lens[i])
                self.ptrs[i] = 0

            # Randomly sample an item
            j = self.indices[i][self.ptrs[i]]  # item index
            self.ptrs[i] += 1

            meta = self.metas[i][j]
            yield meta

    def random_indices(self, N: int) -> list[int]:
        indices = list(range(N))
        random.shuffle(indices)
        return indices