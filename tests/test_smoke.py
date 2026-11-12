from pathlib import Path

import numpy as np

from ofl import configs
from ofl.smoke import diloco_ref, olmo_tiny
from ofl.smoke.toy import drop_ratio, generate_tokens


def test_toy_corpus_is_deterministic_and_follows_the_rule() -> None:
    a = generate_tokens(5000, 64, 0.1, seed=0)
    b = generate_tokens(5000, 64, 0.1, seed=0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, generate_tokens(5000, 64, 0.1, seed=1))
    follows = np.mean(a[1:] == (5 * a[:-1].astype(np.int64) + 3) % 64)
    assert 0.85 < follows < 0.95


def test_drop_ratio_compares_last_window_to_first() -> None:
    assert drop_ratio([4.0] * 10 + [2.0] * 10) == 0.5


def test_olmo_smoke_passes_a_falling_loss_and_fails_everything_else(root: Path) -> None:
    cfg = configs.load_smoke_tiny(root)
    falling = [4.0 - 0.05 * i for i in range(60)]
    assert olmo_tiny.evaluate(falling, ["step60"], cfg) == []
    assert olmo_tiny.evaluate([4.0] * 60, ["step60"], cfg)  # flat loss
    assert olmo_tiny.evaluate(falling, [], cfg)  # no checkpoint
    assert olmo_tiny.evaluate(falling[:5], ["step60"], cfg)  # too few steps
    assert olmo_tiny.evaluate([*falling[:-1], float("nan")], ["step60"], cfg)


def test_diloco_smoke_requires_learning_divergence_and_identical_weights_after_sync(
    root: Path,
) -> None:
    cfg = configs.load_smoke_diloco(root)

    def result(final: float, before: float, after: float) -> dict[str, object]:
        rounds = [{"loss": 1.0, "divergence_before": before, "after": after}] * (
            cfg.outer_rounds - 1
        )
        rounds.append({"loss": final, "divergence_before": before, "after": after})
        return {"initial_loss": 4.0, "rounds": rounds}

    assert diloco_ref.evaluate(result(1.0, 0.05, 0.0), cfg) == []
    assert diloco_ref.evaluate(result(4.0, 0.05, 0.0), cfg)  # did not learn
    assert diloco_ref.evaluate(result(1.0, 0.0, 0.0), cfg)  # workers never diverged
    assert diloco_ref.evaluate(result(1.0, 0.05, 1e-6), cfg)  # not synchronised
    assert diloco_ref.evaluate({"initial_loss": 4.0, "rounds": []}, cfg)


def test_diloco_reference_really_trains_two_workers_end_to_end(root: Path) -> None:
    cfg = configs.load_smoke_diloco(root)
    result = diloco_ref.run(cfg)
    assert diloco_ref.evaluate(result, cfg) == []
    assert result["rounds"][-1]["loss"] < result["initial_loss"]
