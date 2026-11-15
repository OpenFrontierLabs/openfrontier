# Evals for the 1B baseline

**Status: plan. Nothing has been evaluated.** No Open Frontier Labs checkpoint exists, so there are no scores in this file. The commands below were checked to parse with the pinned harness; none has been run against a model.

[RFC 0001](../rfcs/0001-milestone-ladder-and-first-diloco-proof-run.md) rung 3 publishes the swarm 1B "next to Run 0 with the same evals". This file fixes those evals before Run 0 finishes, so the task list, metrics and command cannot be chosen after the numbers are known. Run 0's own config turns the in-training evaluators off (`lm_evaluator` and `downstream_evaluator` in [`run0.toml`](../configs/phase1-dense/run0.toml); the upstream defaults use the gpt2 tokenizer and c4), so everything below runs after training, on a converted checkpoint.

## What is evaluated

[Run 0](run0.md): OLMo-core `olmo2_1B_v2`, 16 layers, `d_model` 2048, dolma2 tokenizer (`allenai/dolma2-tokenizer`), 30B-token budget. The swarm run is phase 3 ([phase3-diloco.md](phase3-diloco.md)). Both are base models: no instruction tuning, no chat template.

The current phase-3 config, [`diloco-1b.toml`](../configs/phase3-diloco/diloco-1b.toml), is not Run 0's recipe: it is a Llama-style model (`type_model = "llama2"`, no QK-Norm) with the Mistral-7B tokenizer, reading `datasets/fineweb-edu`. RFC 0001 says "the same recipe", so that may change; until it does, "the same evals" means the same tasks, command and harness commit applied to two models that differ in architecture, tokenizer and data. The harness gives both the same eval strings and each model scores them with its own tokenizer. Converting a prime-diloco checkpoint to Hugging Face format is **not verified**; no conversion path has been checked.

## Harness, pinned

| | |
| --- | --- |
| Harness | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tag `v0.4.13` = `ddd67220430a2470529f25fd5c05a576ca1057a0`, MIT; the pin lives in [`configs/upstreams.toml`](../configs/upstreams.toml) `[lm-evaluation-harness]`, which `make check` validates. The default branch (`d6de816`, `v0.4.13-11`) is not used. |
| Verified | 2026-11-15, by cloning and checking out that commit with git, then `pip`-installing the clone into a scratch venv |

Install on the evaluation machine:

```bash
uv pip install "lm_eval[hf] @ git+https://github.com/EleutherAI/lm-evaluation-harness@ddd67220430a2470529f25fd5c05a576ca1057a0"
```

Checked: the clone installed (without the `hf` extra), `lm-eval ls tasks` lists all nine task names below, and `lm-eval validate --tasks <the nine>` reports "All tasks found and valid". Not checked: the `hf` extra (torch, transformers, accelerate) and any actual evaluation, because that needs a model and downloads the eval datasets.

The harness does not pin its datasets: none of the task yamls below has a dataset `revision`, so each run fetches whatever the Hugging Face repo holds. Record every dataset's Hugging Face revision beside the results.

## Tasks

All nine exist at the pinned commit (task yaml under `lm_eval/tasks/`). Metric keys are the ones each yaml's `metric_list` declares; the harness prints them as `acc,none`, `acc_norm,none`, with `_stderr` siblings. Splits are the ones the yaml names.

| Task | Why | Split | Metric keys | Headline |
| --- | --- | --- | --- | --- |
| `hellaswag` | Sentence completion; in OLMo-1B's core suite | validation | `acc`, `acc_norm` | `acc_norm` |
| `arc_easy` | Grade-school science multiple choice, easy set; in OLMo-1B's core suite | test | `acc`, `acc_norm` | `acc_norm` |
| `arc_challenge` | The hard set; in OLMo-1B's core suite | test | `acc`, `acc_norm` | `acc_norm` |
| `piqa` | Physical commonsense, two choices; in OLMo-1B's core suite | validation | `acc`, `acc_norm` | `acc_norm` |
| `winogrande` | Pronoun resolution, two choices; in OLMo-1B's core suite | validation | `acc` | `acc` |
| `openbookqa` | Science facts plus commonsense; in OLMo-1B's core suite | test | `acc`, `acc_norm` | `acc_norm` |
| `boolq` | Yes/no reading comprehension; in OLMo-1B's core suite | validation | `acc` | `acc` |
| `sciq` | Science questions with a support passage; in OLMo-1B's core suite | test | `acc`, `acc_norm` | `acc_norm` |
| `lambada_openai` | Last-word prediction from a passage, tests long-range context; **not** in OLMo's suite, so it has no reference number | test | `perplexity`, `acc` | `acc` |

`acc_norm` at this commit divides each choice's log-likelihood by its length in characters (`completion_len = len(choice)` in `lm_eval/api/task.py`). The headline column is our choice, made now: `acc_norm` where the task has it and the answers differ in length, `acc` otherwise. Every key is still published.

**Shots: 0 for every task.** None of the nine yamls sets `num_fewshot`, and at this commit the harness uses 0 when the flag is absent (`lm_eval/evaluator.py`), so the harness default and this plan agree; the command passes `--num_fewshot 0` anyway so the choice is on record. Zero-shot is what OLMo-1B's paper reports (below). With 0 shots no few-shot examples are drawn, so the few-shot seed has nothing to select; `--seed` is left at the harness default (`0,1234,1234,1234`).

MMLU and the generative tasks OLMo 2 reports are left out on purpose. MMLU is left out because a 30B-token 1B model is expected to sit near the four-choice chance rate on it, so it would not separate Run 0 from the swarm; that is an inference from OLMo 2's reported 1B pretraining-stage results (Table 9 of the OLMo 2 paper, below), not something we ran. The generative tasks need OLMES's generation setup, which this harness command does not reproduce.

## The command

```bash
HF_DIR=checkpoints/run0-hf/final            # converted checkpoint, see below
OUT=results/evals-1b/run0-final

lm-eval run \
  --model hf \
  --model_args pretrained=$HF_DIR,dtype=bfloat16 \
  --tasks hellaswag arc_easy arc_challenge piqa winogrande openbookqa boolq sciq lambada_openai \
  --num_fewshot 0 \
  --batch_size 16 \
  --output_path $OUT \
  --log_samples
```

Parse-checked on 2026-11-15 by feeding this exact argument list (with the shell variables substituted) to the pinned harness's own CLI parser (`lm_eval._cli.harness.HarnessCLI`): `model=hf`, `model_args={'pretrained': ..., 'dtype': 'bfloat16'}`, the nine tasks, `num_fewshot=0`, `batch_size=16`, `log_samples=True`. It was not executed. The `dtype` and `batch_size` values are not tuned; `bfloat16` is the converter's default save dtype.

**Converting the checkpoint.** `ofl run0 launch` writes an OLMo-core checkpoint. The harness reads a Hugging Face directory. The pinned OLMo-core (`a800c06edcec8e4652de2f80fe0ea572d689fcc3`) has `src/examples/huggingface/convert_checkpoint_to_hf.py` with `-i/--checkpoint-input-path`, `-o/--huggingface-output-dir` and `-t/--tokenizer`, which should be given `allenai/dolma2-tokenizer`. We have read its arguments and have **not** run it, on this model or any other.

## Reference: OLMo

The reference is Ai2's OLMo. Two sources, both read for this file:

- **OLMo-1B, zero-shot, 8 tasks.** [Groeneveld et al. 2024, arXiv:2402.00838](https://arxiv.org/abs/2402.00838), §4.1 and Table 3. Core suite: `arc` (easy and challenge), `boolq`, `openbookqa`, `sciq`, `hellaswag`, `piqa`, `winogrande`. All zero-shot, by rank classification over answer log-likelihoods in Ai2's Catwalk framework. It normalises per task: unconditional for arc and openbookqa, per token for hellaswag, piqa and winogrande, none for boolq and sciq. Our eight non-LAMBADA tasks are exactly this suite.
- **OLMo 2 1B, OLMES.** [Team OLMo et al. 2024, arXiv:2501.00656](https://arxiv.org/abs/2501.00656), Appendix A.1 and Table 20 (5-shot curated examples, both multiple-choice and cloze formulations tried and the better one kept; accuracy for multiple-choice tasks, F1 for generative), and Table 9 for the 1B pretraining and mid-training scores (MMLU, ARC-C, HellaSwag, WinoGrande and others). The harness is [allenai/olmes](https://github.com/allenai/olmes) (HEAD `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e` on 2026-11-15, not pinned by us). Its README says it builds on this harness's features with its own modifications, so it is not this harness.

OLMo-1B's own zero-shot results for these eight tasks are in Table 3 of the OLMo paper. They are not copied here: this repo does not restate third-party benchmark results, and they were produced with a different framework.

**Those published numbers are not a target our harness run should reproduce.** Same task suite, but a different framework, and the normalisations differ: the harness yamls above report only `acc` and `acc_norm` (per character), with no unconditional normalisation. OLMo 2's 5-shot OLMES numbers are a different protocol again. We do not copy them into a table with ours.

**The like-for-like comparison is to run the command above on the OLMo checkpoints with the same harness.** `allenai/OLMo-2-0425-1B` is a Hugging Face model (Apache-2.0 per its card metadata); revisions listed by the Hugging Face API on 2026-11-15, with the commit each resolves to:

| Revision | Commit |
| --- | --- |
| `main` | `a1847dff35000b4271fa70afc5db10fd29fedbdf` |
| `stage1-step10000-tokens21B` | `22f7d18a8b40f2c4356a6cb1bcd8af1e8fb5882c` |
| `stage1-step20000-tokens42B` | `555c6099a0a17a20f07089377bcd45b2aa3a23a9` |

Run 0's budget is 30B tokens and no OLMo 2 1B revision sits at 30B; the two nearest bracket it. Even so they are an anchor, not a match: OLMo 2 trained on a different mix (the model card names OLMo-mix-1124; Run 0 reads Dolma 3), and whether an intermediate stage-1 checkpoint is at the same point of its learning-rate schedule as Run 0's final one is not verified. `main` is the end of 4T pretraining plus 50B mid-training tokens and is not comparable at 30B tokens. Any comparison table must say which revision it used.

## Contamination

Question: do the eval documents appear in what the model trained on? Only the training side tells us, so the check is per run, against that run's own data.

**What the repo records today.** [`manifests/datasets.toml`](../manifests/datasets.toml) lists dataset ids, licences and Hugging Face revisions, and the Run 0 mix `OLMo-mix-0625-official` (1016 pre-tokenised `.npy` shard paths named in OLMo-core, streamed from `https://olmo-data.org`, tokenised with `allenai/dolma2-tokenizer`). Every `sha256` is `pending`: nothing was downloaded or hashed ([run0.md §Data](run0.md#data)). So today we can name *which* data Run 0 reads, but not prove which bytes.

**Procedure.** One run per training manifest, before any score is published:

1. For each of the nine tasks, dump the eval split named in the table above from its dataset (`dataset_path`, `dataset_name` and split come from the task yaml). Use the same fields the harness scores: context plus every answer choice. For tasks with `doc_to_decontamination_query` (all but `hellaswag`, which does not set it) that field is the minimum; for `hellaswag` use the full context and choices.
2. Normalise as the harness's own cleaning code does: lowercase, strip punctuation, split on whitespace. That code is in the pinned tag: `lm_eval/decontamination/janitor.py` (`Janitor`, `word_ngrams`) and `scripts/clean_training_data/` (`generate_13_grams.py` and the README, which follows the GPT-3 paper's 13-gram method). Reuse them; do not write a second scheme.
3. Decode the training shards (the default mix, or the tokenised FineWeb-Edu shards if Run 0 is launched with `--data-glob`) to text with the dolma2 tokenizer, normalise the same way, and flag every eval document that shares at least one 13-gram with the training text. The Run 0 shards hold token ids, not text, hence the decode; the dataset repo `allenai/dolma3_mix-6T-1025-7B` in the manifest is marked a reference there, and its relation to the shards is not verified, so it is not used instead.
4. Publish, per task: the number of eval documents, the number flagged, and the 13-gram hit list. Add the dataset revision and the training-manifest hash (once one exists).

The swarm run reads its own data (the `PrimeIntellect/fineweb-edu` parquet shards in the manifest), so it gets its own run of steps 1 to 4. We do not check OLMo's data; for the OLMo reference models, contamination is Ai2's to document, and we do not claim a result about it.

**Not automated.** Nothing here exists as code: `ofl` has no eval, decode or n-gram command; the harness at the pinned commit has no run-time flag for decontamination (`should_decontaminate` and `doc_to_decontamination_query` are set in the yamls, but `lm_eval/evaluator.py` and `lm_eval/_cli/` never mention decontamination); its 13-gram scripts target the Pile. The hashes are `pending`. The compute needed to decode the 1016 shards has not been estimated. If a task has material overlap, the follow-up is to re-score on the non-flagged documents with a custom task through `--include_path`; that is not written either.

## Publishing

Not implemented. `ofl deploy --env huggingface` exits with `not implemented`, and no evidence receipt exists yet ([ARCHITECTURE.md layer 5](ARCHITECTURE.md#5-storage--evidence)). The plan, when it exists, is to publish with each checkpoint:

- the harness's `results_<date>.json` and `samples_<task>_<date>.jsonl` files from `$OUT`, unedited;
- the harness commit above, the exact command, the task list, and the Hugging Face revision of every eval dataset;
- the contamination table above, or `not run`;
- standard errors with every score. `acc_stderr,none` is in the harness output; a difference between Run 0 and the swarm smaller than about two standard errors is reported as no difference.

Failed or partial runs are published too.
