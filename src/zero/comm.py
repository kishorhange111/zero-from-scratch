"""The three collectives ZeRO needs, with the bytes each rank sends counted.

Bytes are counted as the bandwidth-optimal (ring) algorithms send them, which is also
what NCCL does for large messages. For p ranks and a full buffer of n bytes, each rank
sends:
- all-reduce: 2n(p − 1)/p (a reduce-scatter followed by an all-gather);
- reduce-scatter (n in, n/p out): n(p − 1)/p;
- all-gather (n/p in, n out): n(p − 1)/p.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist


class Comm:
    def __init__(self):
        self.rank, self.p = dist.get_rank(), dist.get_world_size()
        self.sent = {"all_reduce": 0, "reduce_scatter": 0, "all_gather": 0}
        self.seconds = 0.0  # wall time blocked in collectives (incl. waits on async ones)

    @property
    def total(self):
        return sum(self.sent.values())

    def _bytes(self, t):
        return t.numel() * t.element_size() * (self.p - 1) // self.p

    def all_reduce(self, t):
        t0 = time.perf_counter()
        dist.all_reduce(t)
        self.seconds += time.perf_counter() - t0
        self.sent["all_reduce"] += 2 * self._bytes(t)

    def reduce_scatter(self, full: torch.Tensor) -> torch.Tensor:
        out = torch.empty(full.numel() // self.p, dtype=full.dtype)
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(out, full)
        self.seconds += time.perf_counter() - t0
        self.sent["reduce_scatter"] += self._bytes(full)
        return out

    def all_gather(self, shard: torch.Tensor, out: torch.Tensor | None = None, async_op=False):
        if out is None:
            out = torch.empty(shard.numel() * self.p, dtype=shard.dtype)
        t0 = time.perf_counter()
        work = dist.all_gather_into_tensor(out, shard, async_op=async_op)
        self.seconds += time.perf_counter() - t0
        self.sent["all_gather"] += self._bytes(out)
        return (out, work) if async_op else out

    def wait(self, work):
        t0 = time.perf_counter()
        work.wait()
        self.seconds += time.perf_counter() - t0
