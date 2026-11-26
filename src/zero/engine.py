"""ZeRO stages 0–3 by hand.

Every layer of the model (a *unit*) gets one flat buffer: its parameters laid end to end
and padded to a multiple of p, so each rank owns one contiguous 1/p slice. With p ranks
and Ψ parameters, each rank holds:

| stage | parameters | gradients | optimizer (master + Adam m, v) | collectives per step |
|---|---|---|---|---|
| 0, DDP | Ψ | Ψ | Ψ | all-reduce gradients |
| 1 | Ψ | Ψ | Ψ / p | reduce-scatter gradients, all-gather parameters |
| 2 | Ψ | Ψ / p | Ψ / p | the same, with each unit reduced during backward |
| 3 | Ψ / p | Ψ / p | Ψ / p | all-gather (forward), all-gather (backward), reduce-scatter |

With bf16 compute and an fp32 master copy this is the ZeRO paper's 2Ψ + 2Ψ + 12Ψ bytes,
divided as above. Stages 0–2 move 2Ψ per step and stage 3 moves 3Ψ.

Stage 3 never keeps a whole model. A unit's parameters are gathered by a forward
pre-hook and freed by a forward post-hook. Freeing alone is not enough, though: autograd
saves each weight for backward (``F.linear`` does), and the saved tensor keeps the
gathered storage alive, so backward stays correct but nothing is saved. So the forward
runs under ``saved_tensors_hooks``, which replaces any saved view of a gathered buffer
with a handle (unit, offset, shape, stride). In backward, a gradient hook on the unit's output
gathers the unit again, before any of its parameter gradients are accumulated, and
unpacking a handle returns a view of the new buffer. Once a unit's last parameter
gradient lands (``register_post_accumulate_grad_hook``), the gradients are
reduce-scattered into the rank's shard and the gathered copy is freed. With ``prefetch``
the next unit's all-gather is issued asynchronously while the current one computes.
"""

from __future__ import annotations

import math
import time

import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import StorageWeakRef

from .comm import Comm


class Unit:
    def __init__(self, idx, module, p, rank, dtype):
        self.idx, self.module = idx, module
        self.params = list(module.parameters())
        self.numels = [q.numel() for q in self.params]
        self.shapes = [q.shape for q in self.params]
        self.n = sum(self.numels)
        self.shard_n = math.ceil(self.n / p)
        self.padded = self.shard_n * p
        self.lo, self.hi = rank * self.shard_n, (rank + 1) * self.shard_n
        flat = torch.zeros(self.padded, dtype=torch.float32)
        torch.cat([q.detach().flatten().float() for q in self.params], out=flat[: self.n])
        self.init_flat = flat  # fp32, every rank identical (same seed); freed by the engine
        self.full = None  # gathered compute-dtype buffer, when present
        self.work = None  # pending async all-gather
        self.ready = 0  # parameter gradients accumulated so far this backward

    def views(self, buf):
        out, o = [], 0
        for n, s in zip(self.numels, self.shapes, strict=True):
            out.append(buf[o : o + n].view(s))
            o += n
        return out


class Engine:
    def __init__(self, model, stage, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, clip=None, dtype=torch.float32, prefetch=False):  # fmt: skip
        assert stage in (0, 1, 2, 3)
        self.model, self.stage, self.dtype, self.prefetch = model, stage, dtype, prefetch
        self.lr, self.b1, self.b2, self.eps, self.wd, self.clip = (
            lr,
            *betas,
            eps,
            weight_decay,
            clip,
        )
        self.c = Comm()
        p, r = self.c.p, self.c.rank
        self.units = [Unit(i, m, p, r, dtype) for i, m in enumerate(model.units)]
        self.live, self.peak, self.peak_gathered = {}, 0, 0
        self.gathered_refs = []  # weak refs to storages: proof that freed buffers really die
        self.t = 0
        self.adam_seconds = 0.0
        mixed = dtype != torch.float32
        for u in self.units:
            sl = slice(None) if stage == 0 else slice(u.lo, u.hi)
            if stage < 3:  # full compute-dtype parameters on every rank
                u.full = self._alloc(f"param{u.idx}", u.init_flat.to(dtype))
                self._bind(u)
                own = u.full[sl]
            else:  # only this rank's shard
                u.shard = self._alloc(f"param{u.idx}", u.init_flat[sl].to(dtype).clone())
                for q in u.params:
                    q.data = torch.empty(0, dtype=dtype)
                own = u.shard
            # the fp32 master is the parameter itself unless compute is lower precision
            u.master = self._alloc(f"master{u.idx}", u.init_flat[sl].clone()) if mixed else own
            u.m = self._alloc(f"m{u.idx}", torch.zeros_like(u.master, dtype=torch.float32))
            u.v = self._alloc(f"v{u.idx}", torch.zeros_like(u.master, dtype=torch.float32))
            if stage < 2:  # a full gradient buffer the parameters' .grad are views of
                u.grad_full = self._alloc(f"grad{u.idx}", torch.zeros(u.padded, dtype=dtype))
            else:
                u.grad_shard = self._alloc(f"grad{u.idx}", torch.zeros(u.shard_n, dtype=dtype))
            del u.init_flat
            for q in u.params:
                q.register_post_accumulate_grad_hook(lambda q, u=u: self._grad_ready(u))
            if stage == 3:
                u.module.register_forward_pre_hook(lambda m, a, u=u: self._pre_forward(u))
                u.module.register_forward_hook(lambda m, a, out, u=u: self._post_forward(u, out))
        self.persistent = sum(self.live.values())

    # -- memory accounting: every buffer the engine owns ------------------------------------
    def _alloc(self, name, t):
        self.live[name] = t.numel() * t.element_size()
        self.peak = max(self.peak, sum(self.live.values()))
        self.peak_gathered = max(self.peak_gathered, self.gathered_bytes())
        return t

    def gathered_bytes(self):
        """Bytes of stage-3 parameters currently gathered (beyond this rank's shards)."""
        return sum(v for k, v in self.live.items() if k.startswith("gathered"))

    def _free(self, name):
        self.live.pop(name, None)

    def _bind(self, u):
        for q, view in zip(u.params, u.views(u.full), strict=True):
            q.data = view

    # -- stage 3: gather, free, re-gather ------------------------------------------------
    def _gather(self, u, async_op=False):
        if u.full is not None:
            if u.work is not None and not async_op:
                self.c.wait(u.work)
                u.work = None
            return
        u.full = self._alloc(f"gathered{u.idx}", torch.empty(u.padded, dtype=self.dtype))
        self.gathered_refs.append(StorageWeakRef(u.full.untyped_storage()))
        if async_op:
            _, u.work = self.c.all_gather(u.shard, u.full, async_op=True)
        else:
            self.c.all_gather(u.shard, u.full)
        self._bind(u)

    def _release(self, u):
        if u.work is not None:
            self.c.wait(u.work)
            u.work = None
        for q in u.params:
            q.data = torch.empty(0, dtype=self.dtype)
        u.full = None
        self._free(f"gathered{u.idx}")

    def _pre_forward(self, u):
        self._gather(u)
        if self.prefetch and u.idx + 1 < len(self.units):
            self._gather(self.units[u.idx + 1], async_op=True)

    def _post_forward(self, u, out):
        if torch.is_grad_enabled():
            out.register_hook(lambda g, u=u: self._pre_backward(u))
        self._release(u)

    def _pre_backward(self, u):
        self._gather(u)
        if self.prefetch and u.idx > 0:
            self._gather(self.units[u.idx - 1], async_op=True)

    def _pack(self, t):
        ptr = t.untyped_storage().data_ptr()
        for u in self.units:
            if u.full is not None and u.full.untyped_storage().data_ptr() == ptr:
                return (u.idx, t.storage_offset(), tuple(t.shape), t.stride())
        return t

    def _unpack(self, h):
        if not isinstance(h, tuple):
            return h
        u = self.units[h[0]]
        self._gather(u)
        return torch.as_strided(u.full, h[2], h[3], h[1])

    # -- gradients -----------------------------------------------------------------------
    def _grad_ready(self, u):
        u.ready += 1
        if u.ready < len(u.params):
            return
        if self.stage == 0:
            self.c.all_reduce(u.grad_full)
        elif self.stage >= 2:
            self._reduce_unit(u)

    def _reduce_unit(self, u):
        flat = torch.zeros(u.padded, dtype=self.dtype)
        self._alloc(f"gradfull{u.idx}", flat)
        torch.cat([q.grad.flatten() for q in u.params], out=flat[: u.n])
        for q in u.params:
            q.grad = None
        u.grad_shard += self.c.reduce_scatter(flat)
        self._free(f"gradfull{u.idx}")
        if self.stage == 3:
            self._release(u)

    def zero_grad(self):
        for u in self.units:
            u.ready = 0
            if self.stage < 2:
                u.grad_full.zero_()
                for q, view in zip(u.params, u.views(u.grad_full), strict=True):
                    q.grad = view
            else:
                u.grad_shard.zero_()

    # -- one training step ---------------------------------------------------------------
    def forward(self, *args):
        if self.stage < 3:
            return self.model(*args)
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            return self.model(*args)

    def step(self, *args):
        self.zero_grad()
        loss = self.forward(*args)
        loss.backward()
        p = self.c.p
        reduced = []  # summed gradients this rank updates, still in compute dtype
        for u in self.units:
            assert u.ready == len(u.params), "a parameter received no gradient"
            if self.stage == 0:
                reduced.append(u.grad_full)
            elif self.stage == 1:
                reduced.append(self.c.reduce_scatter(u.grad_full))
            else:
                reduced.append(u.grad_shard)
        scale = 1.0
        if self.clip is not None:  # fp32 temporaries one unit at a time
            sq = torch.stack([(g.float() / p).pow(2).sum() for g in reduced]).sum()
            if self.stage > 0:  # each rank holds a disjoint shard: sum the squares
                dist.all_reduce(sq)  # one scalar, left out of the byte count
            scale = min(1.0, self.clip / (float(sq.sqrt()) + 1e-6))
        self.t += 1
        for u, g in zip(self.units, reduced, strict=True):
            t0 = time.perf_counter()
            g = g.float() / p
            self._adam(u, g * scale if scale < 1 else g)
            del g
            self.adam_seconds += time.perf_counter() - t0
            self._publish(u)
        return loss.detach()

    def _adam(self, u, g):
        b1, b2, t = self.b1, self.b2, self.t
        u.m.mul_(b1).add_(g, alpha=1 - b1)
        u.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        u.master.mul_(1 - self.lr * self.wd)
        denom = (u.v.sqrt() / math.sqrt(1 - b2**t)).add_(self.eps)
        u.master.addcdiv_(u.m, denom, value=-self.lr / (1 - b1**t))

    def _publish(self, u):
        """Make the updated master visible to the forward pass."""
        mixed = self.dtype != torch.float32
        if self.stage == 0:
            if mixed:
                u.full.copy_(u.master)
        elif self.stage < 3:
            src = u.master.to(self.dtype) if mixed else u.master.clone()  # never alias full
            self.c.all_gather(src, u.full)
        elif mixed:
            u.shard.copy_(u.master)

    # -- inspection ----------------------------------------------------------------------
    def full_params(self):
        """Every parameter as fp32 (a collective in stage ≥ 1, not counted in ``Comm``)."""
        out = []
        for u in self.units:
            flat = u.master
            if self.stage:
                flat = torch.empty(u.padded)
                dist.all_gather_into_tensor(flat, u.master.contiguous())
            out += [v.clone() for v in u.views(flat)]
        return out
