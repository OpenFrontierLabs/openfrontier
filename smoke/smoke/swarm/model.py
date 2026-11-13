"""The swarm PoC model: a tiny char-level GPT, plus the flat-weight wire format.

Weights travel as one flat float32 little-endian vector in `parameters()` order. Every party
hashes that exact byte string, so equal hashes mean bit-identical weights.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab: int
    seq_len: int = 128
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8

    def to_json(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**{k: int(v) for k, v in data.items()})


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        out: torch.Tensor = x + self.mlp(self.ln2(x))
        return out


class CharGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for block in self.blocks:
            h = block(h)
        out: torch.Tensor = self.head(self.norm(h))
        return out

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self(x).reshape(-1, self.cfg.vocab), y.reshape(-1))


def get_flat(model: nn.Module) -> torch.Tensor:
    """The model's weights as one contiguous float32 CPU vector."""
    vec: torch.Tensor = nn.utils.parameters_to_vector(model.parameters())
    return vec.detach().to("cpu", torch.float32).contiguous()


def set_flat(model: nn.Module, flat: torch.Tensor) -> None:
    with torch.no_grad():
        dev = next(model.parameters()).device
        nn.utils.vector_to_parameters(flat.to(dev), model.parameters())


def to_bytes(flat: torch.Tensor) -> bytes:
    data: bytes = flat.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    return data


def from_bytes(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.float32)


def weights_hash(flat: torch.Tensor) -> str:
    return hashlib.sha256(to_bytes(flat)).hexdigest()


def batch(
    stream: torch.Tensor, gen: torch.Generator, size: int, seq: int
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, stream.numel() - seq - 1, (size,), generator=gen)
    window = stream[starts[:, None] + torch.arange(seq + 1)[None, :]]
    return window[:, :-1], window[:, 1:]


def sample(model: CharGPT, prompt: list[int], length: int, seed: int) -> list[int]:
    """Temperature-0.8 sampling on CPU from a fixed seed."""
    gen = torch.Generator().manual_seed(seed)
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(length):
            ctx = torch.tensor([ids[-model.cfg.seq_len :]])
            logits = model(ctx)[0, -1] / 0.8
            ids.append(int(torch.multinomial(F.softmax(logits, -1), 1, generator=gen)))
    return ids
