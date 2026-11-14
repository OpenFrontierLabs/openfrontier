# Copyright 2026 Open Frontier Labs contributors. Licensed under the Apache License, Version 2.0.
"""Train the CRAM-style N-Queens energy, evaluate verifier and solver, write results.json.

Standard library only:  python3 experiments/cram/train.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import nqueens as q


def cpu_description() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return f"{line.split(':', 1)[1].strip()}, {os.cpu_count()} logical CPUs"
    except OSError:
        pass
    return f"{platform.machine()} {platform.processor()}".strip()


def split_by_orbit(seed: int, heldout_orbits: int):
    """Hold out whole symmetry orbits, so no held-out board is a rotation or mirror of a
    training board. Training is single-threaded and deterministic; the split is seeded."""
    orbits = q.symmetry_orbits(q.all_solutions())
    order = list(range(len(orbits)))
    random.Random(f"{seed}:split").shuffle(order)
    held = sorted(order[:heldout_orbits])
    train = [
        q.board_from_cols(s) for k, o in enumerate(orbits) if k not in held for s in o
    ]
    test = [q.board_from_cols(s) for k, o in enumerate(orbits) if k in held for s in o]
    return train, test, held, len(orbits)


def rate(hits: int, n: int) -> dict:
    return {"n": n, "count": hits, "rate": round(hits / n, 6) if n else None}


def verify(theta, threshold, train, test, invalid) -> dict:
    out = {
        "train_accept": rate(
            sum(q.energy(theta, b) <= threshold for b in train), len(train)
        ),
        "heldout_accept": rate(
            sum(q.energy(theta, b) <= threshold for b in test), len(test)
        ),
        "invalid": {},
    }
    total = accepted = 0
    for name, boards in invalid.items():
        energies = [q.energy(theta, b) for b in boards]
        hits = sum(e <= threshold for e in energies)
        out["invalid"][name] = {
            **rate(hits, len(boards)),
            "min_energy_minus_threshold": round(min(energies) - threshold, 4),
        }
        total += len(boards)
        accepted += hits
    out["invalid_all"] = rate(accepted, total)
    return out


def run_solver(
    theta, threshold, beta, test, clue_counts, puzzles_per, chains, max_updates, seed
):
    rng = random.Random(f"{seed}:solver")
    out = {}
    for k in clue_counts:
        solved = wrong = miss = miss_valid = updates = 0
        n = 0
        for board in test:
            queens = [i for i, x in enumerate(board) if x]
            for _ in range(puzzles_per):
                clues = sorted(rng.sample(queens, k))
                res = q.solve(theta, clues, threshold, rng, chains, max_updates, beta)
                valid = q.is_solution(res.board) and all(res.board[c] for c in clues)
                n += 1
                updates += res.updates
                if res.halted:
                    solved += valid
                    wrong += not valid
                else:
                    miss += 1
                    miss_valid += valid
        out[f"clues_{k}"] = {
            "puzzles": n,
            "solved": rate(solved, n),
            "wrong_halt": rate(wrong, n),
            "miss_budget_exhausted": rate(miss, n),
            "miss_but_lowest_energy_board_valid": miss_valid,
            "mean_updates": round(updates / n, 1),
        }
    return out


def coupling_summary(theta) -> dict:
    kinds: dict[str, list[float]] = {
        "row": [],
        "column": [],
        "diagonal": [],
        "non_attacking": [],
    }
    for (dr, dc), w in zip(q.OFFSETS, theta):
        kind = (
            "row"
            if dr == 0
            else "column"
            if dc == 0
            else "diagonal"
            if abs(dr) == abs(dc)
            else "non_attacking"
        )
        kinds[kind].append(w)
    summary = {
        k: {
            "count": len(v),
            "min": round(min(v), 3),
            "mean": round(sum(v) / len(v), 3),
            "max": round(max(v), 3),
        }
        for k, v in kinds.items()
    }
    summary["bias"] = round(theta[q.BIAS], 3)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--heldout-orbits", type=int, default=3, help="symmetry orbits held out (of 12)"
    )
    p.add_argument("--steps", type=int, default=q.TrainConfig.steps)
    p.add_argument("--lr", type=float, default=q.TrainConfig.lr)
    p.add_argument("--momentum", type=float, default=q.TrainConfig.momentum)
    p.add_argument(
        "--tol", type=float, default=0.5, help="threshold = max train energy + tol"
    )
    p.add_argument(
        "--n-random", type=int, default=2000, help="boards per random invalid set"
    )
    p.add_argument("--clues", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument(
        "--puzzles-per", type=int, default=3, help="clue draws per held-out solution"
    )
    p.add_argument("--chains", type=int, default=20)
    p.add_argument("--max-updates", type=int, default=2000, help="per chain")
    p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="solver inverse temperature, learned energy",
    )
    p.add_argument(
        "--hand-beta",
        type=float,
        default=2.0,
        help="solver inverse temperature, hand-set energy",
    )
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    args = p.parse_args(argv)

    train, test, held, n_orbits = split_by_orbit(args.seed, args.heldout_orbits)
    cfg = q.TrainConfig(steps=args.steps, lr=args.lr, momentum=args.momentum)

    t0 = time.perf_counter()
    theta, curve = q.train(train, cfg, log_every=max(1, args.steps // 10))
    threshold = q.fit_threshold(theta, train, args.tol)
    train_s = time.perf_counter() - t0

    hand = q.hand_set_theta()
    hand_threshold = q.fit_threshold(hand, train, 0.0)

    t0 = time.perf_counter()
    invalid = q.invalid_sets(random.Random(f"{args.seed}:invalid"), args.n_random)
    verifier = {
        "learned": verify(theta, threshold, train, test, invalid),
        "hand_set": verify(hand, hand_threshold, train, test, invalid),
    }
    max_train = threshold - args.tol
    sweep = []
    for tol in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        v = verify(theta, max_train + tol, train, test, invalid)
        sweep.append(
            {
                "tol": tol,
                "heldout_accept": v["heldout_accept"]["rate"],
                "invalid_false_accept": v["invalid_all"]["rate"],
            }
        )
    verify_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver_args = (
        test,
        args.clues,
        args.puzzles_per,
        args.chains,
        args.max_updates,
        args.seed,
    )
    solver = {
        "learned": run_solver(theta, threshold, args.beta, *solver_args),
        "hand_set": run_solver(hand, hand_threshold, args.hand_beta, *solver_args),
    }
    solve_s = time.perf_counter() - t0

    e_train = [q.energy(theta, b) for b in train]
    e_test = [q.energy(theta, b) for b in test]
    results = {
        "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "machine": {
            "cpu": cpu_description(),
            "python": platform.python_version(),
            "os": platform.system(),
        },
        "command": " ".join(
            [
                "python3 experiments/cram/train.py",
                *(sys.argv[1:] if argv is None else argv),
            ]
        ),
        "seed": args.seed,
        "split": {
            "orbits": n_orbits,
            "heldout_orbit_ids": held,
            "train_solutions": len(train),
            "heldout_solutions": len(test),
        },
        "train_config": {
            "objective": "pseudolikelihood",
            "steps": cfg.steps,
            "lr": cfg.lr,
            "momentum": cfg.momentum,
            "schedule": "cosine",
            "init": "zeros",
        },
        "params": q.NUM_PARAMS,
        "timing_s": {
            "train_and_threshold": round(train_s, 3),
            "verifier_eval": round(verify_s, 3),
            "solver_eval": round(solve_s, 3),
        },
        "loss_curve": [[s, round(loss, 6)] for s, loss in curve],
        "threshold": {
            "tol": args.tol,
            "learned": round(threshold, 4),
            "hand_set": hand_threshold,
        },
        "energies": {
            "train_min": round(min(e_train), 4),
            "train_max": round(max(e_train), 4),
            "heldout_min": round(min(e_test), 4),
            "heldout_max": round(max(e_test), 4),
        },
        "learned_couplings": coupling_summary(theta),
        "verifier": verifier,
        "tolerance_sweep_learned": sweep,
        "solver": {
            "method": "random-scan Gibbs, clues clamped, init: clue queens only",
            "beta": {"learned": args.beta, "hand_set": args.hand_beta},
            "chains": args.chains,
            "max_updates_per_chain": args.max_updates,
            **solver,
        },
    }
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {k: results[k] for k in ("params", "timing_s", "threshold", "energies")},
            indent=2,
        )
    )
    lv = verifier["learned"]
    print(
        f"learned verifier: train accept {lv['train_accept']['rate']}, held-out accept {lv['heldout_accept']['rate']}, invalid false-accept {lv['invalid_all']['rate']}"
    )
    for name, s in solver["learned"].items():
        print(
            f"learned solver {name}: solved {s['solved']['rate']}, wrong halt {s['wrong_halt']['rate']}, miss {s['miss_budget_exhausted']['rate']}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
