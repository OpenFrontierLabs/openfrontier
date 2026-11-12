# Phase 1, MoE track

**Status: scaffolding. Nothing has been trained, and the command below has not been run on a GPU.**

## Source

[The Ultimate MoE Training Handbook](https://huggingface.co/spaces/huggingface/moe-blogpost) (Nouamane Tazi, Leandro von Werra, Thomas Wolf; Hugging Face; published 2026-11-05), a guide to training Mixture-of-Experts models with Megatron-LM. We read the Space at commit `ef70726e6f8b99d74298a16a4894340f288a58c2`, chapter 2 in full (setup, data, dense baseline, first MoE) and its companion gist [`first_run.sbatch`](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139) at `283029be10b078ac02cf334174d5620fbef8cc0f`, which holds the exact flags. We did not work through chapters 3 to 5 and the appendix, which cover router and expert design choices in depth.

## What the handbook says (checked against its text and script)

| | Dense baseline | MoE |
| --- | --- | --- |
| Layers / hidden / FFN | 16 / 2048 / 8192 | 16 / 2048, 64 experts of width 1024 |
| Attention | 32 heads, GQA with 4 query groups, QK-LayerNorm, RoPE base 500,000 | same |
| Routing | | top-8 of 64, `aux_loss` balancing, coefficient 1e-3, fp32 router, all-to-all dispatcher |
| Parallelism | TP 1, PP 1, CP 1, EP 1 | EP 8 (8 experts per GPU) |
| Parameters | about 1.22B (vocab 128,256) | **6.86B total, about 1.22B active** (1.221B) |
| Hardware | one node, 8 GPUs of about 80 GB (H100s in their runs) | same |

So the "about 6.9B total / 1.2B active, 64 experts, top-8" description matches the handbook's numbers: 6.86B and 1.22B, rounded.

Other details from the handbook: sequence length 4096; global batch 512 (2,097,152 tokens per step); AdamW (betas 0.9, 0.95; weight decay 0.1; clip 1); WSD schedule, peak lr 5e-4, 2000 warmup samples, linear decay over the last 732,145 samples of 27,732,145; full activation recomputation; bf16. Data is FineWeb-Edu tokenised with datatrove (pinned there at commit `2c895af7a95010a96fa53e2dc77cc0ce92799e3f`) into Megatron `.bin`/`.idx` shards, a random 7.4% of documents. Software is the NGC `pytorch:26.06-py3` image with Megatron-LM at `d6316eb45bf1bfbb6f1f848c0b56e55166564d80`, under Slurm with Enroot and Pyxis, logged with Trackio. **The first run in the handbook stops after 120 iterations** (`--exit-interval 120`) as a setup check; the 27.7M-sample schedule in the script is what a full run would follow.

## What we reproduce, and where we differ

[`configs/phase1-moe/handbook-moe-6p9b.toml`](../configs/phase1-moe/handbook-moe-6p9b.toml) carries the handbook's flags verbatim. `ofl moe verify` checks all 88 distinct flag names (these, plus `--exit-interval` and the launch-time ones) against the pinned Megatron-LM source, and passed. Differences, on purpose:

- **No Slurm, Enroot, Pyxis or Trackio.** `ofl moe launch` builds a plain `torch.distributed.run ... pretrain_gpt.py` command for one node. The handbook runs a Trackio-wrapping `pretrain_trackio.py` from its gist; we run Megatron's own `pretrain_gpt.py` and write TensorBoard logs. We did not run either.
- **Checkpoints are saved** (`--save`, `--save-interval 1000`), because published checkpoints are the point. The handbook's run disables saving.
- **`--exit-interval 120` is off by default**; `--setup-check` turns it on to reproduce the handbook's behaviour.
- **Tokenizer is an argument.** The handbook uses the Llama 3.2 tokenizer (`unsloth/Llama-3.2-1B` is the ungated copy, licence `llama3.2`, a use-restricted community licence). We release under Apache-2.0 without use restrictions, so that tokenizer is not our default. With a different vocabulary the embedding and output matrices change size and the 6.86B / 1.22B figures no longer hold exactly.
- **Pinned commit is the handbook's** (`d6316eb45`), not Megatron-LM's current default branch, because the flags target that commit.

## Run it

```bash
uv run ofl moe verify                                    # fetch the pin, check the flag names
uv run ofl moe launch --dry-run --tokenizer-dir /data/tokenizer --data-args-path /data/data-args.txt
```

A real launch needs: a node with 8 GPUs of about 80 GB; a Python environment with Megatron-LM's dependencies (the handbook uses the NGC image above; `--python` selects the interpreter); a tokenizer directory; and a `data-args.txt` listing Megatron dataset prefixes, produced by the handbook's `tokenize_first_run.py` (not part of this repo). `ofl` does none of the tokenisation. The Megatron-LM licence is reported as unclassified by GitHub; read its `LICENSE` before depending on it.
