# Copyright 2026 Open Frontier Labs contributors. Licensed under the Apache License, Version 2.0.
"""Tests for the CRAM-style N-Queens experiment. Standard library only, no network.

python3 experiments/cram/test_nqueens.py
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nqueens as q
from train import split_by_orbit


def row_shifts(board: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Every board made by moving one queen along its row: 8 queens, two share a column."""
    cols = [i % q.N for i in range(q.CELLS) if board[i]]
    return [
        q.board_from_cols(tuple(cols[:r] + [c] + cols[r + 1 :]))
        for r in range(q.N)
        for c in range(q.N)
        if c != cols[r]
    ]


class PuzzleTests(unittest.TestCase):
    def test_exactly_92_distinct_solutions_in_12_orbits(self) -> None:
        sols = q.all_solutions()
        self.assertEqual(len(sols), 92)
        self.assertEqual(len(set(sols)), 92)
        self.assertTrue(all(q.is_solution(q.board_from_cols(s)) for s in sols))
        orbits = q.symmetry_orbits(sols)
        self.assertEqual(sorted(len(o) for o in orbits), [4] + [8] * 11)
        self.assertEqual(sorted(s for o in orbits for s in o), sols)

    def test_offsets_are_unordered_relative_positions(self) -> None:
        self.assertEqual(q.NUM_PARAMS, 113)
        a, b = 3 * q.N + 2, 5 * q.N + 6
        self.assertEqual(q.OFF[a][b], q.OFF[b][a])
        self.assertEqual(q.OFF[a][b], q.OFF[a - 2][b - 2])  # translation invariant
        self.assertNotEqual(q.OFF[a][b], q.OFF[a][5 * q.N + 2])


class EnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = random.Random(7)
        self.theta = [rng.uniform(-1, 1) for _ in range(q.NUM_PARAMS)]
        self.boards = [
            tuple(int(rng.random() < 0.2) for _ in range(q.CELLS)) for _ in range(4)
        ]

    def test_fields_are_single_flip_energy_differences(self) -> None:
        for board in self.boards:
            h = q.fields(self.theta, board)
            for i in range(q.CELLS):
                on = tuple(1 if k == i else x for k, x in enumerate(board))
                off = tuple(0 if k == i else x for k, x in enumerate(board))
                self.assertAlmostEqual(
                    h[i], q.energy(self.theta, on) - q.energy(self.theta, off), places=9
                )

    def test_pl_gradient_matches_finite_differences(self) -> None:
        _, grad = q.pl_loss_and_grad(self.theta, self.boards)
        eps = 1e-6
        for k in [0, 5, 40, 111, q.BIAS]:
            up, down = self.theta.copy(), self.theta.copy()
            up[k] += eps
            down[k] -= eps
            numeric = (
                q.pl_loss_and_grad(up, self.boards)[0]
                - q.pl_loss_and_grad(down, self.boards)[0]
            ) / (2 * eps)
            self.assertAlmostEqual(grad[k], numeric, places=6)

    def test_hand_set_energy_is_exact(self) -> None:
        hand = q.hand_set_theta()
        valid = q.board_from_cols(q.all_solutions()[0])
        self.assertEqual(q.energy(hand, valid), -8.0)
        for attacking in row_shifts(valid):
            self.assertLess(q.energy(hand, valid), q.energy(hand, attacking))
        invalid = q.invalid_sets(random.Random(1), 200)
        self.assertTrue(
            all(q.energy(hand, b) > -8.0 for boards in invalid.values() for b in boards)
        )


class TrainedVerifierTests(unittest.TestCase):
    """A short real training run (500 full-batch steps, a few seconds on one core)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train, cls.test, _, _ = split_by_orbit(seed=0, heldout_orbits=3)
        cls.theta, curve = q.train(cls.train, q.TrainConfig(steps=500), log_every=499)
        cls.curve = curve
        cls.threshold = q.fit_threshold(cls.theta, cls.train, tol=0.5)

    def test_training_lowers_pseudolikelihood_loss(self) -> None:
        (_, first), (_, last) = self.curve[0], self.curve[-1]
        self.assertAlmostEqual(
            first, 0.6931471805599453, places=9
        )  # zero weights: log 2
        self.assertLess(last, 0.05)

    def test_attacking_boards_are_rejected(self) -> None:
        for valid in self.test:
            for attacking in row_shifts(valid):
                self.assertGreater(q.energy(self.theta, attacking), self.threshold)

    def test_heldout_solutions_are_local_minima(self) -> None:
        for valid in self.test:
            e = q.energy(self.theta, valid)
            for attacking in row_shifts(valid):
                self.assertLess(e, q.energy(self.theta, attacking))

    def test_a_heldout_solution_is_accepted(self) -> None:
        best = min(self.test, key=lambda b: q.energy(self.theta, b))
        self.assertLessEqual(q.energy(self.theta, best), self.threshold)


class InvalidSetTests(unittest.TestCase):
    def test_n_random_above_permutation_capacity_is_rejected(self):
        rng = random.Random(0)
        with self.assertRaises(ValueError):
            q.invalid_sets(rng, q.MAX_PERMUTATION_INVALID + 1)
        with self.assertRaises(ValueError):
            q.invalid_sets(rng, 0)
        with self.assertRaises(ValueError):
            q.invalid_sets(rng, -1)


class PathTests(unittest.TestCase):
    def test_results_path_is_fixed_in_the_experiment_dir(self):
        import train as tr

        self.assertEqual(tr.HERE, Path(tr.__file__).resolve().parent)
        self.assertEqual(tr.HERE.name, "cram")


if __name__ == "__main__":
    unittest.main()
