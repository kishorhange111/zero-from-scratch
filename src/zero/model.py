"""A small GPT whose layers are the units ZeRO-3 gathers and frees one at a time."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 256
    ctx: int = 64
    d: int = 64
    layers: int = 4
    heads: int = 4


class Embed(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)

    def forward(self, idx):
        return self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.h = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.proj = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.up, self.down = nn.Linear(c.d, 4 * c.d), nn.Linear(4 * c.d, c.d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class Head(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(c.d)
        self.out = nn.Linear(c.d, c.vocab, bias=False)

    def forward(self, x):
        return self.out(self.ln(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.units = nn.ModuleList([Embed(c), *[Block(c) for _ in range(c.layers)], Head(c)])

    def forward(self, idx, targets):
        x = idx
        for u in self.units:
            x = u(x)
        return F.cross_entropy(x.flatten(0, 1).float(), targets.flatten())
