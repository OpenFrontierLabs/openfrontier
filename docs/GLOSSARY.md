# Glossary

Terms used across this repo, in plain words. If a term in a doc or issue is not here, add it in a PR.

**Bootstrap peer.** A stable, well-known libp2p node that a new peer contacts first to join the network and discover other peers; a peer that knows no addresses has nobody else to ask. Planned for layer 3 in [ARCHITECTURE.md](ARCHITECTURE.md); see the [libp2p specs](https://github.com/libp2p/specs).

**Checkpoint.** A saved snapshot of the model weights (and optimiser state) at a point in training, so a run can resume after a crash or its weights can be published. Run 0 saves every 2000 steps plus an ephemeral one every 500, see [run0.md](run0.md); saving and loading in PyTorch: [saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

**Compute ledger.** A public JSON record of who contributed which compute to which run, validated against [`ledger/schema.json`](../ledger/schema.json).

**Data parallel vs pipeline parallel.** Two ways to split one model's training across many GPUs. Data parallel: every GPU holds a full copy of the model and trains on a different slice of the batch, syncing gradients each step. Pipeline parallel: the model's layers are split across GPUs in sequence and activations flow along the pipeline. Inside an island we use data parallel (FSDP); Mode B plans pipeline parallel across machines. See [ARCHITECTURE.md](ARCHITECTURE.md); introduced by [Megatron-LM](https://arxiv.org/abs/1909.08053).

**DiLoCo** (Distributed Low-Communication training). Several workers each train their own copy of the model for many local steps, then exchange only the difference from the shared weights, about once every few hundred steps instead of every step. That lets training run over ordinary internet links. Introduced by [Douillard et al., 2023](https://arxiv.org/abs/2311.08105); reproduced in the open by [OpenDiLoCo](https://arxiv.org/abs/2407.07852). Our CPU reference is [`src/ofl/smoke/diloco_ref.py`](../src/ofl/smoke/diloco_ref.py).

**DPO** (Direct Preference Optimization). A post-training method that tunes a model on pairs of preferred versus rejected answers without training a separate reward model, by rewriting the objective so the policy itself encodes the preference. Phase 2 runs it with open-instruct's `dpo_tune_cache.py`, see [phase2-post.md](phase2-post.md); paper: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290).

**FSDP** (Fully Sharded Data Parallel). A data-parallel scheme where each GPU holds only a shard of the weights, gradients and optimiser states instead of the whole model, which is what lets a large model fit across many GPUs. The OLMo-core example's default data-parallel mode and prime-diloco's in-node sharding both use it; see [run0.md](run0.md) and [phase3-diloco.md](phase3-diloco.md). [PyTorch FSDP docs](https://pytorch.org/docs/stable/fsdp.html), paper: [Zhao et al., 2023](https://arxiv.org/abs/2304.11277).

**Inner step.** One ordinary optimiser step (AdamW here) a worker takes on its own data, without talking to anyone. Our 1B config uses 500 inner steps per outer step.

**Island.** A single node or a donated cluster that trains as one DiLoCo worker. Inside an island, normal data-parallel or sharded training runs over fast links; between islands, only pseudo-gradients cross the internet.

**Manifest.** A file that records exactly which data went into a run: source, licence, revision and hash. See [`manifests/`](../manifests/).

**Mode A / Mode B.** Mode A is DiLoCo across islands, each holding a full copy of the model. Mode B is sharded training where no single node holds the whole model. See [ARCHITECTURE.md](ARCHITECTURE.md).

**MoE and top-k routing.** A Mixture-of-Experts layer replaces the single feed-forward block with many small "expert" blocks plus a router that sends each token to only the top-k of them (top-8 of 64 in our phase-1 config), so the model has far more total parameters than any token actually uses. See [phase1-moe.md](phase1-moe.md); introduced by [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538), guide: [Hugging Face's Ultimate MoE Training Handbook](https://huggingface.co/spaces/huggingface/moe-blogpost).

**Multiaddr.** The self-describing network address format libp2p peers use to say where they can be reached, for example `/ip4/1.2.3.4/tcp/4001`, combining protocol and value in one string. Spec: [multiformats/multiaddr](https://github.com/multiformats/multiaddr).

**NAT traversal.** Techniques that let two peers behind home routers (NATs) connect directly, typically hole punching, with a relay as the fallback. Planned for layer 3; see [ARCHITECTURE.md](ARCHITECTURE.md) and [libp2p's NAT docs](https://docs.libp2p.io/concepts/nat/).

**Outer step.** The synchronisation point in DiLoCo: workers average their pseudo-gradients, and the shared weights take one step of the outer optimiser (SGD with Nesterov momentum) on that average. Every worker then resets to the new shared weights.

**Pin.** An upstream dependency fixed to an exact commit in [`configs/upstreams.toml`](../configs/upstreams.toml), so a run can be reproduced.

**Pseudo-gradient.** Shared weights at the start of a round minus a worker's weights after its inner steps. It plays the role of a gradient for the outer optimiser.

**Relay.** A libp2p peer that forwards traffic between two peers that cannot reach each other directly, often because of NAT. Planned for layer 3; see [ARCHITECTURE.md](ARCHITECTURE.md); spec: [circuit relay v2](https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md).

**RLVR** (Reinforcement Learning with Verifiable Rewards). RL where the reward is a hard check — a unit test, a maths verifier, a compiler — rather than a learned reward model, so the signal cannot be gamed. Phase 2 runs it with open-instruct's `grpo_fast.py`, see [phase2-post.md](phase2-post.md); introduced in [DeepSeekMath](https://arxiv.org/abs/2402.14740).

**Run 0.** The first planned training run: a dense ~1B baseline on one GPU node with OLMo-core. See [run0.md](run0.md).

**Safetensors.** Hugging Face's file format for storing model weights: one JSON header of metadata plus raw tensor data and no executable code, so loading a file cannot run arbitrary code. Planned for published checkpoints in layer 5, see [ARCHITECTURE.md](ARCHITECTURE.md); docs: [safetensors](https://huggingface.co/docs/safetensors/index).

**SFT** (Supervised Fine-Tuning). Training a base model further on curated input-output examples so it learns a format or skill (instruction following, chat) rather than only next-token prediction. First stage of phase 2, with open-instruct's `finetune.py`; see [phase2-post.md](phase2-post.md) and [allenai/open-instruct](https://github.com/allenai/open-instruct).

**Tokens vs parameters.** Tokens are the pieces text is cut into for the model, roughly sub-words; parameters are the learned numbers inside the model. A model's size is counted in parameters, the data it eats in tokens: Run 0 is about 1.5B parameters fed 30B tokens, see [run0.md](run0.md).

