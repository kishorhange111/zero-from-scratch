# zero-from-scratch

[![CI](https://github.com/kishorhange111/zero-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/zero-from-scratch/actions/workflows/ci.yml)

The ZeRO stages behind DeepSpeed and PyTorch FSDP, written from scratch on three
`torch.distributed` collectives:
- stage 0, DDP;
- stage 1, sharded optimizer state;
- stage 2, sharded gradients as well;
- stage 3, sharded parameters as well, gathered one layer at a time.

Every stage is tested against single-process `torch.optim.AdamW`. Per-rank memory and
bytes on the wire match the ZeRO formulas exactly, and an independent OS-level
measurement of resident memory confirms the memory figures.

## How it works ([`engine.py`](src/zero/engine.py))

Each layer of the model is a *unit*. Its parameters are laid end to end in one flat
buffer, padded to a multiple of p, so every rank owns one contiguous 1/p slice. For Ψ
parameters and p ranks, each rank holds:

| stage | parameters | gradients | optimizer: master + Adam m, v | sent per step |
|---|---|---|---|---|
| 0 (DDP) | Ψ | Ψ | Ψ | all-reduce: 2Ψ |
| 1 | Ψ | Ψ | Ψ / p | reduce-scatter + all-gather: 2Ψ |
| 2 | Ψ | Ψ / p | Ψ / p | 2Ψ, each layer reduced as soon as its gradients are ready |
| 3 | Ψ / p | Ψ / p | Ψ / p | all-gather in forward and backward, then reduce-scatter: 3Ψ |

With bf16 compute and an fp32 master, that is the paper's 2 + 2 + 12 bytes per
parameter, sharded as above.

**Stage 3** needs four mechanisms:
1. A forward pre-hook all-gathers the unit, and a post-hook frees it.
2. Freeing alone does nothing, because autograd saves each weight for backward and the
   saved tensor keeps the gathered storage alive. So the forward runs under
   `torch.autograd.graph.saved_tensors_hooks`. Any saved view of a gathered buffer is
   replaced by a handle (unit, offset, shape, stride), and unpacking it in backward
   returns a view of a freshly gathered buffer.
3. A gradient hook on each unit's output gathers the unit again as its backward begins,
   before any parameter gradient is accumulated. The embedding, for instance, saves no
   weight at all, but its gradient still needs the parameter's shape.
4. `register_post_accumulate_grad_hook` counts arriving gradients. When a unit's last one
   lands, the gradients are reduce-scattered into the rank's shard, and the gathered copy
   is freed.

With `prefetch`, the next unit's all-gather is issued asynchronously while the current
one computes, in both directions. Gradient clipping sums squared norms across shards
with one scalar all-reduce.

## Results ([`bench.py`](src/zero/bench.py))

A 25.5M-parameter GPT (d = 512, 8 layers) runs on p gloo processes on one 8-core machine
(WSL2), one thread per rank, with a micro-batch of 2 × 64 tokens per rank and 8 steps
after a warm-up. Each configuration runs in fresh processes. The full tables are in
[results/bench.md](results/bench.md) and [results/bench_bf16.md](results/bench_bf16.md).

**Memory: the formulas hold, and the OS sees them.** p = 4:

| stage | engine MB per rank (= formula) | resident before steps | Δ resident | Δ formula | peak resident in steps |
|---|---|---|---|---|---|
| fp32, 0 | 408.2 | 650 | | | 719 |
| fp32, 1 | 255.2 | 496 | −154 | −153.1 | 557 |
| fp32, 2 | 178.6 | 420 | −76 | −76.6 | 497 |
| fp32, 3 | 102.1 | 344 | −76 | −76.5 | 433 |
| bf16 + fp32 master, 0 | 408.2 | 650 | | | 722 |
| bf16 + fp32 master, 1 | 178.6 | 421 | −229 | −229.6 | 468 |
| bf16 + fp32 master, 2 | 140.3 | 382 | −39 | −38.3 | 433 |
| bf16 + fp32 master, 3 | 102.1 | 344 | −38 | −38.2 | 402 |

Resident memory comes from `/proc/self/statm`. It falls by the formula's amount, to
within 1 MB at every step at p = 4 and within 2 MB at p = 2, on top of a constant 242 MB
runtime baseline. This needs `MALLOC_MMAP_THRESHOLD_` fixed. By default glibc raises the
threshold after a large free and keeps later blocks in its heap. Then stage 2 → 3 frees
77 MB while peak resident memory falls by only 19 MB. During a step, stage 3 holds at most one
gathered layer (12.6 MB), or two with prefetch.

**Bytes on the wire** equal the formula exactly in every run: 153.1 MB per step for
stages 0–2 and 229.6 MB for stage 3 at p = 4, and half that in bf16.

**Time.** On one CPU host, the same configuration varied by up to 46% in step time over
five runs. The claims below are the ones that held in every run:
- **ZeRO-1 beats DDP at p = 4** in all five runs, by 11–30% per step, even though it
  spends more time in collectives. A DDP rank runs Adam over the whole model, and that
  takes 3.5–4.6× longer than a ZeRO rank updating its quarter (for example 460 vs 130
  ms). At p = 2 the two are within noise.
- **Prefetch** cuts the time stage 3 spends blocked in collectives by 40–44% at p = 4, and
  the step by 9–16%. It costs one extra gathered layer.
- **Stage 3 costs 1.5× the bytes**, and its blocked collective time was 1.4× stage 1's
  (731 vs 511 ms) in the final run. With prefetch, stage 3 was 1–18% slower than stage 1
  across the runs, at 40% of its memory.
- **Stage 2 is 2–19% slower than stage 1 here.** Its reduce-scatters run inside backward,
  but gloo on CPU does not overlap them with compute. Many small messages then cost more
  than one per layer at the end. On GPUs with NCCL streams this is where stage 2 gains.

## Tests

`pytest -q` runs 32 tests on real gloo groups with p = 2, 3 and 4. With p = 3, no layer
size divides evenly, so padding is always exercised. They check:
- **every stage, with and without prefetch, equals single-process AdamW** with weight
  decay and gradient clipping, after 3 steps: losses to 1e-5 relative, and every
  parameter on every rank;
- all ranks end with bit-identical parameters, in fp32 and in bf16;
- bytes sent per step equal 2Ψ(p − 1)/p, or 3Ψ(p − 1)/p for stage 3, exactly;
- persistent bytes per rank equal the formula in the table, exactly, for fp32 and for
  bf16 with an fp32 master;
- stage 3 holds at most two gathered layers, and none once a step is over;
- **weak references to the gathered storages** prove that nothing survives a stage-3
  forward pass, and that without the saved-tensor hooks, autograd keeps every layer with
  a saved weight alive. This test failed once in eight full-suite runs, without its
  message being captured, and then passed 12 of 12 runs under load;
- bf16 compute tracks fp32 within 2% in every stage, and the flat-shard Adam equals
  `torch.optim.AdamW`.

One detail the equivalence tests ran into: a key bias gets a gradient of exactly 0,
because softmax ignores a per-query shift. Autograd returns about 1e-10 of rounding
noise there instead. With eps = 1e-8, Adam divides that noise by its own scale and takes
full ±lr steps whose sign depends on summation order. So two correct implementations
disagree by up to 4.7e-5 on exactly those elements. The tests use eps = 1e-5, which is
far below every real gradient (about 1e-3).

```
pip install -e ".[dev]" && pytest -q
python -m zero.bench --out results --P 2 4          # Linux: reads /proc for memory
python -m zero.bench --out results --P 4 --bf16
```
