"""Phase 1 smoke: a real, tiny OLMo-core training run on CPU (the pinned olmo-core itself).

Run as `python -m smoke.olmo_tiny [--steps N]`. Exits non-zero unless the trainer's own
cross-entropy loss falls and the trainer wrote a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from smoke.configs import SmokeError, SmokeTiny, find_root, load_smoke_tiny
from smoke.toy import drop_ratio, generate_tokens


def evaluate(losses: list[float], checkpoint_dirs: list[str], cfg: SmokeTiny) -> list[str]:
    """Return the reasons this run does not pass (empty means pass)."""
    if len(losses) < 20:
        return [f"only {len(losses)} loss values recorded, need at least 20"]
    if any(x != x or x in (float("inf"), float("-inf")) for x in losses):
        return ["a loss value is NaN or infinite"]
    problems: list[str] = []
    ratio = drop_ratio(losses)
    if ratio > cfg.loss_drop_ratio:
        problems.append(
            f"loss did not fall enough: last-10 mean / first-10 mean = {ratio:.3f}, "
            f"required <= {cfg.loss_drop_ratio}"
        )
    if not checkpoint_dirs:
        problems.append("the trainer wrote no step checkpoint")
    return problems


def run(cfg: SmokeTiny, steps: int, workdir: Path) -> tuple[list[float], list[str]]:
    # Imported here: olmo_core is heavy and only this smoke needs it.
    from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
    from olmo_core.data.numpy_dataset import NumpyDatasetDType
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.optim import AdamWConfig, CosWithWarmup
    from olmo_core.train import (
        Duration,
        TrainerConfig,
        prepare_training_environment,
        teardown_training_environment,
    )
    from olmo_core.train.callbacks import Callback
    from olmo_core.train.train_module import TransformerTrainModuleConfig
    from olmo_core.utils import seed_all

    seq = cfg.sequence_length
    corpus = workdir / "toy.npy"
    generate_tokens(cfg.sequences * seq, cfg.vocab, cfg.noise, cfg.corpus_seed).tofile(corpus)

    losses: list[float] = []

    class RecordLoss(Callback):
        def post_step(self) -> None:
            metric = self.trainer.get_metric("train/CE loss")
            if metric is not None:
                losses.append(float(metric))

    tokenizer = TokenizerConfig(
        vocab_size=cfg.vocab, eos_token_id=cfg.vocab - 1, pad_token_id=cfg.vocab - 1
    )
    factory = getattr(TransformerConfig, cfg.factory)
    model_cfg = factory(vocab_size=tokenizer.padded_vocab_size())
    dataset_cfg = NumpyFSLDatasetConfig(
        paths=[str(corpus)],
        sequence_length=seq,
        tokenizer=tokenizer,
        work_dir=str(workdir / "dataset-work"),
        dtype=NumpyDatasetDType.uint16,
    )
    loader_cfg = NumpyDataLoaderConfig(
        global_batch_size=seq * cfg.batch_sequences, seed=0, num_workers=0
    )
    train_module_cfg = TransformerTrainModuleConfig(
        rank_microbatch_size=seq * cfg.batch_sequences,
        max_sequence_length=seq,
        optim=AdamWConfig(lr=cfg.lr),
        scheduler=CosWithWarmup(warmup_steps=cfg.warmup_steps),
        compile_model=False,
    )
    save_folder = workdir / "save"
    trainer_cfg = TrainerConfig(
        save_folder=str(save_folder),
        save_overwrite=True,
        max_duration=Duration.steps(steps),
        metrics_collect_interval=1,
        cancel_check_interval=1,
    ).with_callback("record_loss", RecordLoss())

    prepare_training_environment(backend=None)
    try:
        seed_all(cfg.init_seed)
        model = model_cfg.build(init_device="meta")
        train_module = train_module_cfg.build(model)
        loader = loader_cfg.build(
            dataset_cfg.build(), dp_process_group=train_module.dp_process_group
        )
        trainer = trainer_cfg.build(train_module, loader)
        trainer.fit()
    finally:
        teardown_training_environment()
    return losses, sorted(p.name for p in save_folder.glob("step*"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m smoke.olmo_tiny")
    parser.add_argument("--steps", type=int, help="override [train].steps")
    parser.add_argument("--workdir", type=Path, help="keep artefacts here instead of a temp dir")
    args = parser.parse_args(argv)
    try:
        cfg = load_smoke_tiny(find_root())
    except SmokeError as exc:
        print(exc, file=sys.stderr)
        return exc.code
    steps = args.steps or cfg.steps

    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        losses, ckpts = run(cfg, steps, args.workdir)
    else:
        with tempfile.TemporaryDirectory(prefix="ofl-olmo-smoke-") as tmp:
            losses, ckpts = run(cfg, steps, Path(tmp))

    problems = evaluate(losses, ckpts, cfg)
    summary = {
        "steps": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "drop_ratio": drop_ratio(losses) if len(losses) >= 10 else None,
        "checkpoints": ckpts,
        "problems": problems,
    }
    print("OFL_SMOKE_RESULT " + json.dumps(summary))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
