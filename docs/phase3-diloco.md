# Phase 3: DiLoCo outer loops across islands

**Status: configured, not built.** `ofl diloco launch` exits with `not built`. No island setup on prime-diloco exists. Separately, a small swarm proof of concept (our own HTTP coordinator and plain-PyTorch DiLoCo, a 4.8M-parameter char model, no auth) has run across two real machines: see [join-a-node.md](join-a-node.md#swarm-proof-of-concept-two-commands).

## Upstream

[PrimeIntellect-ai/prime-diloco](https://github.com/PrimeIntellect-ai/prime-diloco) (Apache-2.0; the Python package inside is `zeroband`), pinned to default-branch HEAD `de5b9317da5f57247f11abed2fca259076795460`. Its last tag is `v0.5` (October 2024); we pin the commit, not the tag. What its README and source at that commit say it provides: an `ElasticDeviceMesh` for fault-tolerant training with a heartbeat, live checkpoint recovery for joining nodes, a custom int8 all-reduce kernel for pseudo-gradients, FSDP2 inside a node, and CPU offload of the outer optimiser.

## Config

[`configs/phase3-diloco/diloco-1b.toml`](../configs/phase3-diloco/diloco-1b.toml) is a config in the pinned upstream's own format. It parses with the upstream `Config` class (we checked that once by hand in a scratch environment; `ofl check` does not install prime-diloco). The values:

- **500 inner steps** (`diloco.inner_steps`): OpenDiLoCo's 1.1B setting as the plan states it. INTELLECT-1's 10B run used 100.
- **Outer optimiser**: SGD with Nesterov momentum. In the pinned source the momentum (0.9) and Nesterov are hard-coded in `diloco.py`; only `outer_lr` (default 0.7) is a config key.
- **Compression `uint8`**: the upstream option for the quantised pseudo-gradient all-reduce.
- **Data**: `PrimeIntellect/fineweb-edu` (a pre-shuffled FineWeb-Edu derivative, ODC-By 1.0) as local parquet with a `text` column, the format the upstream reads.

## What the upstream model is, versus the plan

The upstream model definition is **Llama-style** (`type_model = "llama2"` or `"llama3"`), implemented in `src/zeroband/models/llama/model.py`. It has **no QK-Norm** (we found no QK-norm in that file) and it is not the Qwen3-style architecture the plan names. `llama2` loads `mistralai/Mistral-7B-v0.1`'s tokenizer (Apache-2.0 per its card); `llama3` loads the gated `meta-llama/Meta-Llama-3-8B` tokenizer. Using QK-Norm or a Qwen3-style model means changing the upstream model code, which the plan calls "fork and own the trainer"; that has not been done.

## It does not run on CPU

We tried. In a scratch environment with the upstream's locked dependencies, `train.py` on its own `configs/debug/diloco.toml` fails at CUDA initialisation (`Found no NVIDIA driver`): `train.py` calls `torch.cuda.set_device` unconditionally, shards the model onto a CUDA device mesh, and its data loader moves batches to `"cuda"`. So it cannot be used for a CPU smoke test.

## What `make smoke` checks instead

[`smoke/smoke/diloco_ref.py`](../smoke/smoke/diloco_ref.py) is a small **reference implementation of the DiLoCo algorithm in plain PyTorch**. It is not prime-diloco and tests nothing about it. Two worker processes on `torch.distributed` (gloo) each run inner AdamW for 25 steps on their own data shard, average the pseudo-gradient (global weights minus local weights), take an outer SGD step with Nesterov momentum 0.9 on the global weights, and reset to them, for 8 outer rounds on a toy transformer. The smoke fails unless the held-out loss falls to under half its start, the workers' weights genuinely diverge during the inner steps, and they are bit-identical after every outer sync. With the outer learning rate set to 0 (a negative control) it fails, as it should.

## Hybrid islands, as intended

Each island is a phase-1-style single-node (or single-cluster) trainer; DiLoCo outer loops run across islands. Joining and scoring contributions is not built: see [join-a-node.md](join-a-node.md). The plan's larger targets (a 10B run, then 30 to 70B with SparseLoCo, which prime-diloco does not provide) come after this phase exists and are not scheduled.
