# Phase 2: post-training

**Status: documented, not built.** There is no phase-2 launcher: `ofl post launch` exits with `not built`. There is no base model to post-train yet.

The stack, pinned in [`configs/upstreams.toml`](../configs/upstreams.toml) and listed in [`configs/phase2-post/post-training.toml`](../configs/phase2-post/post-training.toml):

| Stage | Upstream | Pin | Entry point at the pin |
| --- | --- | --- | --- |
| SFT | [allenai/open-instruct](https://github.com/allenai/open-instruct) | `v0.3.0` = `26de99604002d6a7b25a2c5f2ee880e388a21dca` | `open_instruct/finetune.py` |
| DPO | open-instruct | same | `open_instruct/dpo_tune_cache.py` |
| RLVR | open-instruct | same | `open_instruct/grpo_fast.py` |
| Async RL | [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | `v0.10.0` = `9e15490dcab1d506ce3462e69b8aa5dde45c6b3a` | console script `rl` (also `sft`) |
| RL environments | [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) | `v0.4.1` = `540f40faab8c376b85c6afa2f872683b0ed1d4d1` | console script `vf-run` |

The files exist at those commits (read from shallow clones). We have not run any of them, and we have written no flags for them: a launcher flag goes in only after it is checked against the pinned source. The same rule applies when someone builds the launcher.

Licences: open-instruct and prime-rl Apache-2.0, verifiers MIT (GitHub licence metadata). prime-rl's `main` is already on `v0.10.1.dev205`; we pin the last non-dev release.
