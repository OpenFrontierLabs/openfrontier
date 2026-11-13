# Run 0: single-node dense baseline (phase 1)

**Status: scaffolding. Nothing has been trained.** The launcher, config and pins exist and are checked against the pinned upstream. The 1B run itself has not been started on any GPU, so there are no loss curves, throughput numbers or memory numbers to report.

## What it is

Run 0 is the first phase-1 run: a dense decoder of about 1B parameters, trained on one GPU node with the official OLMo-core trainer. It exists to prove the fully open pipeline end to end (data manifest, config, logs, every checkpoint, weights) before anything distributed is attempted. The earlier plan had Run 0 as a 1B DiLoCo run across nodes; that is now phase 3, see [phase3-diloco.md](phase3-diloco.md).

## Upstream, pinned

| | |
| --- | --- |
| Trainer | [allenai/OLMo-core](https://github.com/allenai/OLMo-core) `v3.0.0` = `a800c06edcec8e4652de2f80fe0ea572d689fcc3`, Apache-2.0 |
| Recipes (reference) | [allenai/olmo-cookbook](https://github.com/allenai/olmo-cookbook) `c226eb6a07ccef068c26379d9c90304fbaeea841` |

All pins are in [`configs/upstreams.toml`](../configs/upstreams.toml), verified with `git ls-remote` and `gh api` on 2026-11-12.

**Which OLMo-core script.** OLMo-core's *official* scripts (`src/scripts/official/`) cover OLMo 2 32B, OLMo 3 7B and 32B, and OLMo-hybrid 7B; none is a 1B. A 1B recipe exists, `src/scripts/train/OLMo2/OLMo2-1B.py`, but that directory's README says its scripts use an internal module and Ai2-only data paths and "won't work out-of-the-box for external users". So the launcher runs the public example `src/examples/llm/train.py` (the one OLMo-core's own docs use), selects the 1B model with `--model-factory olmo2_1B_v2`, and sets the optimiser, batch and schedule values that the internal 1B script uses. Two differences from that script: the example has plain AdamW (the internal script uses `SkipStepAdamWConfig`), and the example's default data-parallel mode is FSDP (the internal script uses HSDP).

## The model, and how it differs from the plan

`TransformerConfig.olmo2_1B_v2`: 16 layers, `d_model` 2048, 16 attention heads, SwiGLU FFN of width 8192, RoPE theta 500,000, QK-Norm (RMSNorm on queries and keys), RMSNorm, reordered (post-sublayer) norm placement. With the 100,352-row padded dolma2 vocabulary that is **1.485B parameters including embeddings, 1.279B without** (computed from the config with `olmo_core`, not measured).

The plan's target was a Qwen3-style dense decoder with QK-Norm **and GQA**. OLMo-core's 1B has QK-Norm but plain multi-head attention (no grouped KV heads), and a different norm placement. Qwen3-style GQA is not what runs here.

## Recipe

From [`configs/phase1-dense/run0.toml`](../configs/phase1-dense/run0.toml):

| | |
| --- | --- |
| Sequence length | 4096 |
| Global batch | 2,097,152 tokens (512 sequences) |
| Optimiser | AdamW, lr 4e-4, weight decay 0.033, betas (0.9, 0.95); z-loss 1e-5; grad-clip 1.0 (example default) |
| Schedule | cosine with 2000 warmup steps (`CosWithWarmup`) |
| Token budget | 30B tokens, about 20 per parameter: a budget we chose, not a result |
| Seeds | `init_seed` 12536 (upstream default), `data_loader.seed` 0 |
| Checkpoints | every 2000 steps, plus ephemeral every 500, into `checkpoints/run0/` |

## Data

Default: the Dolma 3 pre-tokenised mix named `OLMo-mix-0625-official` (1016 shard paths defined in OLMo-core at the pinned commit), streamed from `https://olmo-data.org`, tokenised with `allenai/dolma2-tokenizer`. Dataset ids, licences and Hugging Face revisions are in [`manifests/datasets.toml`](../manifests/datasets.toml). The cards say ODC-By 1.0; we have not audited the individual sources inside Dolma 3. The dolma2 tokenizer's card declares no licence; confirm with Ai2 before releasing weights that ship it.

FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, ODC-By 1.0, which requires attribution) is supported by [`run0-fineweb-edu.toml`](../configs/phase1-dense/run0-fineweb-edu.toml), but OLMo-core reads pre-tokenised `.npy` files, not parquet text, and **`ofl` does not tokenise**. You bring shards tokenised with the dolma2 tokenizer and an EOS token after each document (OLMo-core's data-loading guide describes the format), then pass `--data-glob`. No FineWeb-Edu shards exist yet. No content hashes are recorded anywhere because none were computed.

## Run it

```bash
cargo run -- run0 launch --dry-run          # print the exact command; needs no GPU, no download
cargo run -- run0 verify                    # fetch the pin; the upstream script accepts every override (--dry-run)
```

On a GPU node: create an environment with a CUDA build of torch and `ai2-olmo-core==3.0.0` (the environment `make setup` creates has CPU torch on purpose), then

```bash
cargo run -- run0 launch --python /path/to/gpu-env/bin/python --nproc 8
```

This runs `python -m torch.distributed.run --standalone --nproc-per-node=N <checkout>/src/examples/llm/train.py run0 ...` with the overrides in the config. The launcher fetches the OLMo-core checkout at the pinned commit into `.cache/upstreams/olmo-core` and refuses to run if it is at another commit. We have **not** run this on a GPU. OLMo-core's README lists optional kernels (flash-attn, TransformerEngine, Liger) that you may want installed; none of that has been tried here.

`ofl run0 verify` is the only upstream check we have done on the real config: it runs the upstream script with `--dry-run`, which parses every override into the upstream's config classes. It does not start training.

## What the smoke test covers

`make smoke` runs the same pinned `olmo_core` trainer (data loader, train module, trainer, checkpointer) on CPU with the 1.08M-parameter `olmo2_14M` model and a generated toy corpus, and fails unless the trainer's own cross-entropy falls to under half its starting value and a checkpoint is written. It exercises the code path, not the 1B configuration, the real data or any GPU kernel. Config: [`smoke-tiny.toml`](../configs/phase1-dense/smoke-tiny.toml).
