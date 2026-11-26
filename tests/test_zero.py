import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from zero.engine import Engine
from zero.model import GPT, GPTConfig

CFG = GPTConfig(vocab=61, ctx=16, d=24, layers=2, heads=2)  # odd sizes: shards need padding
B, T, STEPS = 12, 16, 3
# eps: the key bias has a true gradient of exactly 0 (softmax ignores a per-query shift),
# so what autograd returns there is ~1e-10 of rounding noise. With eps = 1e-8 Adam divides
# that noise by its own scale and takes full ±lr steps whose sign depends on summation
# order, so any two correct implementations disagree on it. 1e-5 is far below every real
# gradient (~1e-3) and far above the noise.
HP = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1, clip=0.5)
CONFIGS = [(0, False), (1, False), (2, False), (3, False), (3, True)]


def batch(step):
    g = torch.Generator().manual_seed(100 + step)
    x = torch.randint(0, CFG.vocab, (B, T + 1), generator=g)
    return x[:, :-1], x[:, 1:]


def _worker(rank, p, init, out_dir):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=p)
    torch.set_num_threads(1)
    res = {}
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for stage, pre in CONFIGS:
                torch.manual_seed(0)
                eng = Engine(GPT(CFG), stage, dtype=dtype, prefetch=pre, **HP)
                losses, comm = [], []
                for s in range(STEPS):
                    x, y = batch(s)
                    lo, hi = rank * B // p, (rank + 1) * B // p
                    before = eng.c.total
                    losses.append(float(eng.step(x[lo:hi], y[lo:hi])))
                    comm.append(eng.c.total - before)
                gathered_after_step = eng.gathered_bytes()
                res[(str(dtype), stage, pre)] = {
                    "losses": losses, "comm": comm, "persistent": eng.persistent,
                    "peak_gathered": eng.peak_gathered, "left_gathered": gathered_after_step,
                    "params": eng.full_params(),
                    "psi": sum(u.padded for u in eng.units),
                    "units": [u.padded for u in eng.units],
                }  # fmt: skip
        torch.save(res, os.path.join(out_dir, f"{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=[2, 3, 4])
def runs(request, tmp_path_factory):
    p, d = request.param, tmp_path_factory.mktemp(f"p{request.param}")
    init = "file://" + str(d / "rdzv")
    mp.spawn(_worker, args=(p, init, str(d)), nprocs=p, join=True)
    return p, [torch.load(d / f"{r}.pt") for r in range(p)]


def reference():
    torch.manual_seed(0)
    m = GPT(CFG)
    lr, betas, eps, wd, clip = HP["lr"], HP["betas"], HP["eps"], HP["weight_decay"], HP["clip"]
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd,
                            foreach=False)  # fmt: skip
    losses = []
    for s in range(STEPS):
        loss = m(*batch(s))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, [q.detach().clone() for q in m.parameters()]


FP32 = str(torch.float32)


@pytest.mark.parametrize("stage,pre", CONFIGS)
def test_every_stage_equals_single_process_adamw(runs, stage, pre):
    p, out = runs
    ref_losses, ref_params = reference()
    for r in range(p):
        got = out[r][(FP32, stage, pre)]
        mean_loss = [sum(o[(FP32, stage, pre)]["losses"][s] for o in out) / p for s in range(STEPS)]
        assert mean_loss == pytest.approx(ref_losses, rel=1e-5)
        for a, b in zip(got["params"], ref_params, strict=True):
            # a wrong shard, offset or reduction is off by ~lr = 3e-3; summation order
            # alone stays below 1e-5
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_all_ranks_hold_identical_parameters(runs):
    p, out = runs
    for key in out[0]:
        for r in range(1, p):
            for a, b in zip(out[0][key]["params"], out[r][key]["params"], strict=True):
                assert torch.equal(a, b), key


def test_communication_bytes_match_the_zero_formulas(runs):
    p, out = runs
    for (dt, stage, pre), v in out[0].items():
        e = 4 if dt == FP32 else 2
        one = v["psi"] * e * (p - 1) // p  # one pass over every parameter
        want = {0: 2 * one, 1: 2 * one, 2: 2 * one, 3: 3 * one}[stage]
        assert v["comm"] == [want] * STEPS, (dt, stage, pre)


def test_persistent_memory_matches_the_zero_formulas(runs):
    p, out = runs
    for (dt, stage, pre), v in out[0].items():
        psi = v["psi"]
        e, k = (4, 8) if dt == FP32 else (2, 12)  # k: master (if separate) + Adam m, v
        want = {
            0: e * psi + e * psi + k * psi,
            1: e * psi + e * psi + k * psi // p,
            2: e * psi + e * psi // p + k * psi // p,
            3: (e + e + k) * psi // p,
        }[stage]
        assert v["persistent"] == want, (dt, stage, pre)


def test_stage3_holds_at_most_one_or_two_gathered_layers(runs):
    p, out = runs
    for (dt, stage, pre), v in out[0].items():
        if stage != 3:
            continue
        e = 4 if dt == FP32 else 2
        biggest = sorted(v["units"])[-2:]
        assert v["left_gathered"] == 0
        limit = e * (sum(biggest) if pre else max(biggest) + sorted(v["units"])[-2])
        assert 0 < v["peak_gathered"] <= limit
        assert v["peak_gathered"] < e * v["psi"]  # never the whole model


def test_bf16_compute_tracks_fp32_in_every_stage(runs):
    p, out = runs
    bf = str(torch.bfloat16)
    for stage, pre in CONFIGS:
        fp = out[0][(FP32, stage, pre)]["losses"]
        assert out[0][(bf, stage, pre)]["losses"] == pytest.approx(fp, rel=2e-2)


def test_adam_on_a_flat_shard_equals_torch_adamw():
    from zero.engine import Engine as E

    torch.manual_seed(0)
    w = torch.randn(50)
    ref = w.clone().requires_grad_()
    opt = torch.optim.AdamW([ref], lr=1e-2, weight_decay=0.1, foreach=False)

    class U:
        master, m, v = w.clone(), torch.zeros(50), torch.zeros(50)

    e = E.__new__(E)
    e.lr, e.b1, e.b2, e.eps, e.wd, e.t = 1e-2, 0.9, 0.999, 1e-8, 0.1, 0
    for s in range(5):
        g = torch.randn(50, generator=torch.Generator().manual_seed(s))
        ref.grad = g.clone()
        opt.step()
        e.t += 1
        e._adam(U, g)
    torch.testing.assert_close(U.master, ref.detach())  # fp32 default: 1e-5 abs, 1.3e-6 rel


def _forward_worker(rank, p, init, out_dir):
    import gc

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=p)
    try:
        res = {}
        for hooked in (True, False):
            torch.manual_seed(0)
            eng = Engine(GPT(CFG), 3, **HP)
            x, y = batch(0)
            loss = eng.forward(x, y) if hooked else eng.model(x, y)  # model(): no pack hooks
            gc.collect()
            res[hooked] = sum(not r.expired() for r in eng.gathered_refs), len(eng.gathered_refs)
            if hooked:
                loss.backward()  # still correct after everything was freed
                res["ready"] = [u.ready == len(u.params) for u in eng.units]
            del loss
        torch.save(res, os.path.join(out_dir, f"{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_stage3_forward_really_frees_every_gathered_layer(tmp_path):
    """Weak references to the storages, not the engine's own bookkeeping: after the forward
    pass no gathered buffer survives. Without the saved-tensor hooks, autograd keeps the
    buffer of every layer that saves a weight for backward (all but the embedding) alive,
    and the memory saving of stage 3 silently disappears."""
    mp.spawn(_forward_worker, args=(2, "file://" + str(tmp_path / "rdzv"), str(tmp_path)),
             nprocs=2, join=True)  # fmt: skip
    res = torch.load(tmp_path / "0.pt")
    n_units = CFG.layers + 2
    assert res[True] == (0, n_units), res
    assert res[False] == (n_units - 1, n_units), res
    assert all(res["ready"])
