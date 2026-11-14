# Copyright 2026 Open Frontier Labs contributors. Licensed under the Apache License, Version 2.0.
#
# Simplified CPU reimplementation of the CRAM-style pairwise energy described in
# Hoover, Blayney, and Whittington, "Classical Energies Learn to Reason 1,000x Faster than
# Looped Models", arXiv:2611.17309 (2026); official code https://github.com/Prefrontal-AI/CRAM
# (Apache-2.0, Prefrontal). No code is copied from that repository.
"""CRAM-style pairwise energy on 8x8 N-Queens, standard library only.

Board: 64 binary cells, row-major, 1 = queen. Energy (paper Eq. 1, two cell states):

    E(x) = sum_{i<j} J[offset(i, j)] * x_i * x_j + b * sum_i x_i

``offset(i, j)`` is the unordered relative displacement (dr, dc) between two squares, so the
only prior is "relative position only": 112 shared couplings plus one bias, 113 parameters.
Nothing in the learned model knows which offsets attack; it learns that from solutions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

N = 8
CELLS = N * N


# --- Puzzle -------------------------------------------------------------------------------


def all_solutions(n: int = N) -> list[tuple[int, ...]]:
    """Every n-queens solution by backtracking, as a column per row, in lexicographic order."""
    out: list[tuple[int, ...]] = []
    cols: list[int] = []

    def place(row: int) -> None:
        if row == n:
            out.append(tuple(cols))
            return
        for c in range(n):
            if all(c != pc and abs(c - pc) != row - pr for pr, pc in enumerate(cols)):
                cols.append(c)
                place(row + 1)
                cols.pop()

    place(0)
    return out


def board_from_cols(cols: tuple[int, ...]) -> tuple[int, ...]:
    board = [0] * CELLS
    for r, c in enumerate(cols):
        board[r * N + c] = 1
    return tuple(board)


def attacks(i: int, j: int) -> bool:
    dr, dc = abs(i // N - j // N), abs(i % N - j % N)
    return dr == 0 or dc == 0 or dr == dc


def attacking_pairs(board: tuple[int, ...]) -> int:
    queens = [i for i, x in enumerate(board) if x]
    return sum(attacks(a, b) for k, a in enumerate(queens) for b in queens[k + 1 :])


def is_solution(board: tuple[int, ...]) -> bool:
    """Ground-truth rule check, used only for evaluation and labeling, never by the energy."""
    return sum(board) == N and attacking_pairs(board) == 0


def symmetry_orbits(sols: list[tuple[int, ...]]) -> list[list[tuple[int, ...]]]:
    """Group solutions into the 12 orbits under the 8 symmetries of the square."""

    def images(cols: tuple[int, ...]) -> set[tuple[int, ...]]:
        cells = [(r, c) for r, c in enumerate(cols)]
        found = set()
        for _ in range(4):
            cells = [(c, N - 1 - r) for r, c in cells]  # rotate 90 degrees
            for pts in (cells, [(r, N - 1 - c) for r, c in cells]):  # and its mirror
                found.add(tuple(c for _, c in sorted(pts)))
        return found

    seen: set[tuple[int, ...]] = set()
    orbits = []
    for s in sols:
        if s not in seen:
            orbit = sorted(images(s))
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


# --- Parameterization ---------------------------------------------------------------------


def _canonical(dr: int, dc: int) -> tuple[int, int]:
    return (dr, dc) if dr > 0 or (dr == 0 and dc > 0) else (-dr, -dc)


OFFSETS: list[tuple[int, int]] = sorted(
    {
        _canonical(dr, dc)
        for dr in range(-N + 1, N)
        for dc in range(-N + 1, N)
        if (dr, dc) != (0, 0)
    }
)
_OFFSET_ID = {o: k for k, o in enumerate(OFFSETS)}
# OFF[i][j]: coupling index shared by every pair of squares with the same relative offset.
OFF: list[list[int]] = [
    [
        -1 if i == j else _OFFSET_ID[_canonical(j // N - i // N, j % N - i % N)]
        for j in range(CELLS)
    ]
    for i in range(CELLS)
]
NUM_PARAMS = len(OFFSETS) + 1  # 112 couplings + 1 bias
BIAS = len(OFFSETS)


def energy(theta: list[float], board: tuple[int, ...]) -> float:
    queens = [i for i, x in enumerate(board) if x]
    e = theta[BIAS] * len(queens)
    for k, a in enumerate(queens):
        row = OFF[a]
        for b in queens[k + 1 :]:
            e += theta[row[b]]
    return e


def fields(theta: list[float], board: tuple[int, ...]) -> list[float]:
    """h_i = E(x with x_i=1) - E(x with x_i=0): the energy cost of a queen on cell i."""
    h = [theta[BIAS]] * CELLS
    for j, x in enumerate(board):
        if x:
            row = OFF[j]
            for i in range(CELLS):
                if i != j:
                    h[i] += theta[row[i]]
    return h


def hand_set_theta(attack: float = 2.0, bias: float = -1.0) -> list[float]:
    """Hand-set baseline, NOT learned: E = 2 * attacking_pairs - queens.

    Every solution scores exactly -8. Any other board scores higher: fewer than 8 queens
    score at least -7, and 8 + m queens (m >= 0) on 8 rows force at least m attacking pairs,
    each extra pair costing +2. So threshold -8 with zero tolerance is exact.
    """
    theta = [
        attack if dr == 0 or dc == 0 or abs(dr) == abs(dc) else 0.0
        for dr, dc in OFFSETS
    ]
    return [*theta, bias]


# --- Pseudolikelihood training (paper Eq. 5) -------------------------------------------------


def _softplus(z: float) -> float:
    return z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def pl_loss_and_grad(
    theta: list[float], boards: list[tuple[int, ...]]
) -> tuple[float, list[float]]:
    """Mean negative log pseudolikelihood per cell, and its exact gradient.

    p(x_i = 1 | rest) = sigmoid(-h_i). Minimizing -log p(x_i | rest) lowers the energy of the
    labeled board relative to its single-cell flip at i. dL/dh_i = x_i - p(x_i = 1 | rest).
    """
    grad = [0.0] * NUM_PARAMS
    loss = 0.0
    for board in boards:
        queens = [j for j, x in enumerate(board) if x]
        h = fields(theta, board)
        for i in range(CELLS):
            hi = h[i]
            loss += _softplus(hi) if board[i] else _softplus(-hi)
            g = board[i] - _sigmoid(-hi)
            grad[BIAS] += g
            row = OFF[i]
            for j in queens:
                if j != i:
                    grad[row[j]] += g
    scale = 1.0 / (len(boards) * CELLS)
    return loss * scale, [g * scale for g in grad]


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    lr: float = 2.0
    momentum: float = 0.9


def train(
    boards: list[tuple[int, ...]], cfg: TrainConfig, log_every: int = 0
) -> tuple[list[float], list[tuple[int, float]]]:
    """Full-batch SGD with momentum and a cosine learning-rate schedule, from zero weights.

    Deterministic: no sampling, so the result depends only on ``boards`` and ``cfg``.
    Returns the weights and a sparse (step, loss) curve.
    """
    theta = [0.0] * NUM_PARAMS
    vel = [0.0] * NUM_PARAMS
    curve = []
    for step in range(cfg.steps):
        loss, grad = pl_loss_and_grad(theta, boards)
        if log_every and (step % log_every == 0 or step == cfg.steps - 1):
            curve.append((step, loss))
        lr = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * step / cfg.steps))
        for k in range(NUM_PARAMS):
            vel[k] = cfg.momentum * vel[k] - lr * grad[k]
            theta[k] += vel[k]
    return theta, curve


def fit_threshold(
    theta: list[float], train_boards: list[tuple[int, ...]], tol: float
) -> float:
    """Accept threshold: highest training-solution energy plus tolerance."""
    return max(energy(theta, b) for b in train_boards) + tol


# --- Invalid boards for the verifier ---------------------------------------------------------


MAX_PERMUTATION_INVALID = 40320 - 92  # 8! minus the 92 solutions


def invalid_sets(rng: random.Random, n_random: int) -> dict[str, list[tuple[int, ...]]]:
    """Documented invalid boards. Every set is deduplicated and contains no solution."""
    sols = [list(s) for s in all_solutions()]
    sets: dict[str, set[tuple[int, ...]]] = {
        "empty": {tuple([0] * CELLS)},
        "missing_queen": set(),  # a solution minus one queen: 7 queens, no attacking pair
        "extra_queen": set(),  # a solution plus one queen: 9 queens
        "row_shift": set(),  # one queen moved along its row: 8 queens, >= 1 attacking pair
        "permutation": set(),  # one queen per row and column, diagonal attacks only
        "random_8": set(),  # 8 queens on uniformly random squares
        "random_density": set(),  # each square holds a queen with probability 1/8
    }
    for cols in sols:
        board = list(board_from_cols(tuple(cols)))
        for i in range(CELLS):
            flipped = board.copy()
            flipped[i] ^= 1
            sets["missing_queen" if board[i] else "extra_queen"].add(tuple(flipped))
        for r in range(N):
            for c in range(N):
                if c != cols[r]:
                    sets["row_shift"].add(
                        board_from_cols(tuple(cols[:r] + [c] + cols[r + 1 :]))
                    )

    def fill(name: str, draw, target: int) -> None:
        while len(sets[name]) < target:
            board = draw()
            if not is_solution(board):
                sets[name].add(board)

    def perm() -> tuple[int, ...]:
        cols = list(range(N))
        rng.shuffle(cols)
        return board_from_cols(tuple(cols))

    def rand8() -> tuple[int, ...]:
        squares = set(rng.sample(range(CELLS), N))
        return tuple(int(i in squares) for i in range(CELLS))

    if n_random < 1 or n_random > MAX_PERMUTATION_INVALID:
        raise ValueError(
            f"n_random must be between 1 and {MAX_PERMUTATION_INVALID}, got {n_random}"
        )
    fill("permutation", perm, n_random)
    fill("random_8", rand8, n_random)
    fill(
        "random_density",
        lambda: tuple(int(rng.random() < 1 / N) for _ in range(CELLS)),
        n_random,
    )
    return {name: sorted(boards) for name, boards in sets.items()}


# --- Solver: single-cell descent with clamped clues -----------------------------------------


@dataclass(frozen=True)
class SolveResult:
    board: tuple[
        int, ...
    ]  # halting board, or the lowest-energy board seen if no chain halted
    energy: float
    halted: bool  # a chain reached the threshold
    updates: int  # single-cell update attempts across all chains
    chains: int  # chains started


def solve(
    theta: list[float],
    clues: list[int],
    threshold: float,
    rng: random.Random,
    chains: int,
    max_updates: int,
    beta: float = 1.0,
) -> SolveResult:
    """Random-scan Gibbs search at inverse temperature beta, one free cell per update.

    Each chain starts from the clue queens only (every other cell empty, the paper's N-Queens
    init) with the clues clamped. An update picks a free cell uniformly and resamples it from
    p(queen | rest) = sigmoid(-beta * h_i). A chain halts as soon as its energy is at or below
    ``threshold``; chains run in turn until one halts or all exhaust ``max_updates``.
    This is plain Gibbs, not the paper's focused Gibbs search.
    """
    clue_set = set(clues)
    free = [i for i in range(CELLS) if i not in clue_set]
    best: tuple[float, tuple[int, ...]] | None = None
    spent = 0
    for chain in range(1, chains + 1):
        board = [int(i in clue_set) for i in range(CELLS)]
        h = fields(theta, tuple(board))
        e = energy(theta, tuple(board))
        if best is None or e < best[0]:
            best = (e, tuple(board))
        for _ in range(max_updates):
            if e <= threshold:
                exact = energy(
                    theta, tuple(board)
                )  # guard against accumulated rounding
                if exact <= threshold:
                    return SolveResult(tuple(board), exact, True, spent, chain)
            spent += 1
            i = free[rng.randrange(len(free))]
            new = int(rng.random() < _sigmoid(-beta * h[i]))
            if new == board[i]:
                continue
            delta = 1.0 if new else -1.0
            e += delta * h[i]
            board[i] = new
            row = OFF[i]
            for k in range(CELLS):
                if k != i:
                    h[k] += delta * theta[row[k]]
            if e < best[0]:
                best = (e, tuple(board))
        if e <= threshold and energy(theta, tuple(board)) <= threshold:
            return SolveResult(
                tuple(board), energy(theta, tuple(board)), True, spent, chain
            )
    assert best is not None
    return SolveResult(best[1], energy(theta, best[1]), False, spent, chains)
