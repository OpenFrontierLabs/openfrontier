# Glossary

Terms used across this repo, in plain words. If a term in a doc or issue is not here, add it in a PR.

**DiLoCo** (Distributed Low-Communication training). Several workers each train their own copy of the model for many local steps, then exchange only the difference from the shared weights, about once every few hundred steps instead of every step. That lets training run over ordinary internet links. Introduced by [Douillard et al., 2023](https://arxiv.org/abs/2311.08105); reproduced in the open by [OpenDiLoCo](https://arxiv.org/abs/2407.07852). Our CPU reference is [`src/ofl/smoke/diloco_ref.py`](../src/ofl/smoke/diloco_ref.py).

**Inner step.** One ordinary optimiser step (AdamW here) a worker takes on its own data, without talking to anyone. Our 1B config uses 500 inner steps per outer step.

**Outer step.** The synchronisation point in DiLoCo: workers average their pseudo-gradients, and the shared weights take one step of the outer optimiser (SGD with Nesterov momentum) on that average. Every worker then resets to the new shared weights.

**Pseudo-gradient.** Shared weights at the start of a round minus a worker's weights after its inner steps. It plays the role of a gradient for the outer optimiser.

**Island.** A single node or a donated cluster that trains as one DiLoCo worker. Inside an island, normal data-parallel or sharded training runs over fast links; between islands, only pseudo-gradients cross the internet.

**Mode A / Mode B.** Mode A is DiLoCo across islands, each holding a full copy of the model. Mode B is sharded training where no single node holds the whole model. See [ARCHITECTURE.md](ARCHITECTURE.md).

**Run 0.** The first planned training run: a dense ~1B baseline on one GPU node with OLMo-core. See [run0.md](run0.md).

**Compute ledger.** A public JSON record of who contributed which compute to which run, validated against [`ledger/schema.json`](../ledger/schema.json).

**Manifest.** A file that records exactly which data went into a run: source, licence, revision and hash. See [`manifests/`](../manifests/).

**Pin.** An upstream dependency fixed to an exact commit in [`configs/upstreams.toml`](../configs/upstreams.toml), so a run can be reproduced.
