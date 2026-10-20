"""用同一进程内的线程预读小批量图片，避免 Windows 多进程共享内存。"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math

import torch
from torch.utils.data import default_collate


class ThreadedDataLoader:
    def __init__(
        self, dataset, batch_size=64, shuffle=False, num_workers=0,
        persistent_workers=False, pin_memory=False, generator=None,
        worker_threads=4, prefetch_batches=4,
    ):
        if num_workers != 0 or persistent_workers or pin_memory:
            raise ValueError("线程加载器不使用子进程或锁页内存")
        if batch_size < 1 or worker_threads < 1 or prefetch_batches < 1:
            raise ValueError("批次大小、线程数和预读数量必须为正数")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.worker_threads = worker_threads
        self.prefetch_batches = prefetch_batches
        self.generator = generator if generator is not None else torch.Generator().manual_seed(torch.initial_seed())

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def _read_batch(self, indices):
        return default_collate([self.dataset[i] for i in indices])

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        batches = iter(indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size))
        # 最多只保留几批预读数据，不把整个数据集载入内存。
        with ThreadPoolExecutor(max_workers=self.worker_threads) as executor:
            pending = deque()
            for _ in range(self.prefetch_batches):
                batch = next(batches, None)
                if batch is None:
                    break
                pending.append(executor.submit(self._read_batch, batch))
            while pending:
                result = pending.popleft().result()
                batch = next(batches, None)
                if batch is not None:
                    pending.append(executor.submit(self._read_batch, batch))
                yield result
