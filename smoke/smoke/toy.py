"""Deterministic toy corpus shared by both smoke tests.

A noisy first-order Markov chain over `vocab` tokens: the next token is (5 * x + 3) mod vocab with
probability 1 - noise, otherwise uniform. A model that learns the rule drives the cross-entropy from
ln(vocab) toward roughly the noise entropy; one that does not stays near ln(vocab).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_tokens(n: int, vocab: int, noise: float, seed: int) -> NDArray[np.uint16]:
    rng = np.random.default_rng(seed)
    flips = rng.random(n) < noise
    randoms = rng.integers(0, vocab, size=n)
    out = np.empty(n, dtype=np.uint16)
    x = int(rng.integers(0, vocab))
    for i in range(n):
        out[i] = x
        x = int(randoms[i]) if flips[i] else (5 * x + 3) % vocab
    return out


def drop_ratio(losses: list[float], window: int = 10) -> float:
    """Mean of the last `window` losses over the mean of the first `window`."""
    first = sum(losses[:window]) / window
    last = sum(losses[-window:]) / window
    return last / first
