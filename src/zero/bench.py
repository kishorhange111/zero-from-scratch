"""Step time, communication and memory of every ZeRO stage on real processes.

    python -m zero.bench --out results --P 2 4

Each configuration runs in freshly spawned gloo processes (one thread each), so each
rank's peak resident memory (``ru_maxrss``, from the OS) is a measurement independent of
the engine's own accounting. Per rank, the benchmark records:
- the median step time over ``--steps`` steps, after one warm-up;
- the bytes sent per step, against the formula;
- the persistent bytes the engine holds, against the formula;
- the peak bytes of gathered stage-3 layers;
- resident memory (from ``/proc/self/statm``) just before the steps, and its peak during
  them, sampled every millisecond.

glibc normally raises its mmap threshold after a large free and then keeps later blocks
in the heap, so freed memory stays resident and RSS stops tracking what is live. The
workers therefore run with ``MALLOC_MMAP_THRESHOLD_=65536`` (set before spawning), and
every large tensor gets its own mapping that is returned to the OS when it is freed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import tempfile
import threading
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .engine import Engine
from .model import GPT, GPTConfig

CONFIGS = [(0, False), (1, False), (2, False), (3, False), (3, True)]


def _rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class RSSPeak(threading.Thread):
    """Samples resident memory every millisecond (torch ops release the GIL)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, _rss())
            time.sleep(0.001)


def _worker(rank, p, init, a, stage, prefetch, out_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=p)
    try:
        cfg = GPTConfig(vocab=256, ctx=a.ctx, d=a.d, layers=a.layers, heads=a.heads)
        torch.manual_seed(0)
        dtype = torch.bfloat16 if a.bf16 else torch.float32
        eng = Engine(GPT(cfg), stage, lr=1e-3, dtype=dtype, prefetch=prefetch, clip=1.0)
        g = torch.Generator().manual_seed(rank)
        times, comm, losses, comm_s, adam_s = [], [], [], [], []
        gc.collect()
        rss_before = _rss()
        sampler = RSSPeak()
        sampler.start()
        for s in range(a.steps + 1):
            x = torch.randint(0, 256, (a.batch, a.ctx + 1), generator=g)
            dist.barrier()
            before, t0 = eng.c.total, time.perf_counter()
            cs, ad = eng.c.seconds, eng.adam_seconds
            losses.append(float(eng.step(x[:, :-1], x[:, 1:])))
            el = time.perf_counter() - t0
            if s:
                times.append(el)
                comm.append(eng.c.total - before)
                comm_s.append(eng.c.seconds - cs)
                adam_s.append(eng.adam_seconds - ad)
        sampler.stop = True
        sampler.join()
        res = {
            "p": p, "stage": stage, "prefetch": prefetch, "dtype": str(dtype),
            "params": sum(u.n for u in eng.units),
            "psi_padded": sum(u.padded for u in eng.units),
            "step_s": statistics.median(times), "step_times": times,
            "comm_per_step": comm[0], "persistent": eng.persistent,
            "peak_gathered": eng.peak_gathered, "peak_engine": eng.peak,
            "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "rss_before_steps": rss_before, "rss_peak_steps": sampler.peak,
            "comm_s": statistics.median(comm_s), "adam_s": statistics.median(adam_s),
            "losses": losses,
        }  # fmt: skip
        with open(os.path.join(out_dir, f"{rank}.json"), "w") as f:
            json.dump(res, f)
    finally:
        dist.destroy_process_group()


def run(p, a, stage, prefetch):
    with tempfile.TemporaryDirectory() as d:
        init = "file://" + os.path.join(d, "rdzv")
        mp.spawn(_worker, args=(p, init, a, stage, prefetch, d), nprocs=p, join=True)
        ranks = [json.load(open(os.path.join(d, f"{r}.json"))) for r in range(p)]
    out = dict(ranks[0])
    out["step_s"] = max(r["step_s"] for r in ranks)  # the slowest rank sets the pace
    for k in ("max_rss", "rss_peak_steps", "rss_before_steps", "comm_s", "adam_s"):
        out[k] = max(r[k] for r in ranks)
    return out


def formula(stage, p, psi, e, k):
    """Persistent bytes per rank: e bytes per param / grad, k for master + Adam."""
    return {
        0: (e + e + k) * psi,
        1: (e + e) * psi + k * psi // p,
        2: e * psi + (e + k) * psi // p,
        3: (e + e + k) * psi // p,
    }[stage]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--P", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "65536")  # inherited by spawned ranks
    rows = []
    for p in a.P:
        for stage, pre in CONFIGS:
            r = run(p, a, stage, pre)
            rows.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("step_times", "losses")}),
                  flush=True)  # fmt: skip
    tag = "_bf16" if a.bf16 else ""
    with open(os.path.join(a.out, f"bench{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    e, k = (2, 12) if a.bf16 else (4, 8)
    psi = rows[0]["psi_padded"]
    L = [f"GPT with {rows[0]['params'] / 1e6:.1f}M parameters (d = {a.d}, {a.layers} layers), "
         f"{'bf16 compute, fp32 master' if a.bf16 else 'fp32'}, gloo on one machine, one "
         f"thread per rank, micro-batch {a.batch} × {a.ctx} tokens per rank.", "",
         "| p | stage | step (ms) | in collectives | in Adam | MB sent per step | formula "
         "| engine MB per rank | formula | peak gathered MB | RSS before steps (MB) "
         "| peak RSS in steps (MB) |", "|---" * 12 + "|"]  # fmt: skip
    for r in rows:
        name = f"{r['stage']}" + (" + prefetch" if r["prefetch"] else "")
        p = r["p"]
        one = psi * e * (p - 1) // p
        want_comm = (3 if r["stage"] == 3 else 2) * one
        L.append(
            f"| {p} | {name} | {r['step_s'] * 1e3:.0f} | {r['comm_s'] * 1e3:.0f} "
            f"| {r['adam_s'] * 1e3:.0f} | {r['comm_per_step'] / 1e6:.1f} "
            f"| {want_comm / 1e6:.1f} | {r['persistent'] / 1e6:.1f} "
            f"| {formula(r['stage'], p, psi, e, k) / 1e6:.1f} "
            f"| {r['peak_gathered'] / 1e6:.1f} | {r['rss_before_steps'] / 1e6:.0f} "
            f"| {r['rss_peak_steps'] / 1e6:.0f} |"
        )
    with open(os.path.join(a.out, f"bench{tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
