# RFC 0002: Energy-based constraint reasoners (CRAM) as verifiers and an eval-hygiene track

| Field | Value |
| --- | --- |
| **Status** | `draft` |
| **Authors** | @bradlishman (founding steward); drafted by an agent (omp / claude-opus-5-5) |
| **Created** | 2026-11-14 |
| **Layers** | 4 Training, 8 Outputs |
| **Discussion** | [PR #56](https://github.com/OpenFrontierLabs/openfrontier/pull/56), [Discussion #57](https://github.com/OpenFrontierLabs/openfrontier/discussions/57) |
| **Tracking issue** | added once accepted |

## Summary

Open Frontier Labs should treat energy-based constraint reasoners as a side research track, not as a replacement for the pretraining or post-training stack. The concrete starting point is CRAM (Compact Reasoning Associative Memory): a tiny trainable Hopfield–Tank energy that learns pairwise costs from solved puzzles, then reasons by descending that energy with a stopping threshold fitted from training solutions ([Hoover, Blayney, and Whittington, 2026](https://arxiv.org/abs/2611.17309); [blog](https://prefrontal.ai/news/cram); official code [Prefrontal-AI/CRAM](https://github.com/Prefrontal-AI/CRAM), Apache-2.0). This RFC proposes four uses: (1) cheap local solvers and verifiers for constraint-heavy subproblems in the planned Phase 2 stack (RLVR, async RL, [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [verifiers](https://github.com/PrimeIntellect-ai/verifiers)); (2) an open, inspectable research track; (3) eval hygiene, because many popular "reasoning" puzzles are exactly representable by a pairwise energy, plus cheap generation and filtering of constraint data and negatives; (4) tiny on-device or contributor-hardware helpers. It also records the paper's limits, a CPU prototype plan, success metrics, and open questions. CRAM is not a general reasoner and must not be described as one.

## Motivation

Phase 2 is documented and not built ([docs/phase2-post.md](../docs/phase2-post.md)): SFT, DPO and RLVR are pinned to open-instruct, async RL to prime-rl, and RL environments to Prime Intellect's verifiers package. Those launchers do not exist yet, and there is no base model to post-train. When they do exist, constraint-heavy subproblems (logic, planning steps, simple puzzle or program-constraint checks) will need verifiers that are fast, stable, and cheap enough to run on heterogeneous contributor nodes. A multi-million-parameter looped model is a poor default for that job if a much smaller energy can do it.

CRAM is evidence that this class of problem is often easier than recent looped-model papers make it look. The claims below are taken from the paper, not the blog's more informal lede ("over 50,000× faster"), and are not results OFL has reproduced unless a later section says so.

- **What it is.** A trainable form of Hopfield and Tank's 1985 energy, plus recurrent descent at inference time. Weights memorize associations between pairs of valid cells. Training uses a local pseudolikelihood objective at labeled solutions (no backpropagation through the solver). Inference clamps the clues, runs one or more chains from a task-specific start, and updates one cell at a time until a chain's energy falls to a threshold fitted from the training solutions (the highest training-solution energy plus a small tolerance), or until a local stopping test is met.
- **Accuracy and fit time, paper numbers, one NVIDIA H200.** CRAM solves **99.996%** of the 412,786-puzzle Sudoku-Extreme test set after **42 ms** of training, and **100%** of the 1,000 Maze-Unique test mazes after **132 ms**. (The abstract and §4.1 use 99.996%; Table 2 rounds the three-seed Sudoku mean to 100.00±0.00. Fit times are the median of seven warm fits after a first fit in the same process.) Official README parameter counts: about **7,700** nonzero of 263,169 free weights on Sudoku-Extreme, and **18** on Maze-Unique.
- **Training and inference efficiency, versus looped models the authors retrained.** Training is **at least 1,000×** faster than the looped baselines on those two tasks. On a 10,000-puzzle Sudoku-Extreme slice, CRAM matches each baseline's highest accuracy with **at least 282×** fewer inference FLOPs; the 282× figure is against EBR, the cheapest looped model in that comparison (EBR 99.86% at 5.4×10^11 FLOPs/puzzle vs CRAM 1.9×10^9). The paper also reports that CRAM never halted at an incorrect Sudoku or Maze-Unique board under its energy-threshold rule, unlike the looped models' learned or gradient-norm halt heads.
- **Other tasks.** The same energy is evaluated on N-Queens and graph coloring (Baek et al.'s sets). On 8×8 N-Queens, learned CRAM reports 99.95% accuracy with one chain and 100% with 20 lowest-energy chains, 113 parameters, and a 2.9 s fit; a hand-set energy with **two** unique nonzero weights reaches 100%.
- **Data efficiency and interpretability.** On Sudoku-Extreme, CRAM reaches 98% on a 10,000-puzzle test slice from **30** unique training puzzles, against **700** for EBR in the same protocol (an order of magnitude fewer). Learned Sudoku costs concentrated on the 810 cell pairs that share a row, column or box; Maze-Unique's 18 costs implement dead-end filling.
- **The caution that matters for our evals.** A hand-set CRAM energy with **at most three unique nonzero weights** solves every test puzzle of Sudoku-Extreme (1 unique weight), Maze-Unique (3), N-Queens (2) and graph coloring (1). High accuracy on those benchmarks therefore shows that a model has learned a few pairwise rules, not that looped reasoning will scale to ARC-style or open-ended work.

OFL should absorb that caution before we grow an eval suite, and should try the cheap verifier use in Phase 2 rather than ignore a public, Apache-2.0 result that is directly about constraint checking.

## Proposal

### Status of this track

This is a research and tooling track. It does not change the milestone ladder in [RFC 0001](0001-milestone-ladder-and-first-diloco-proof-run.md), the Phase 1–3 launchers, the ledger, or any protocol. It must not delay Run 0 or the Hello, swarm proof. Hybridize with larger models; do not replace them.

### Four uses

1. **Fast verifiers and local solvers for Phase 2.** When `ofl post` and the pinned verifiers / prime-rl / open-instruct RLVR paths are built, a CRAM-style energy is a candidate module for constraint-heavy subproblems: a complete candidate is accepted if its energy is at or below a threshold fitted on valid solutions, and a partial assignment can be completed by descent with clues clamped. Training is cheap enough for contributor machines; inference compute follows difficulty and can stop early. The module sits beside, not instead of, the larger model that proposes plans, code or answers.

2. **An open research track.** Keep a public experiment in this repository (see prototype below), linked from this RFC and discussed in the RFCs category. The interesting parts for OFL are the BPTT-free training recipe, the energy-based stopping test, data efficiency, and inspectable weights — including whether those properties help on bandwidth-limited or heterogeneous nodes. Start from the public paper and [Prefrontal-AI/CRAM](https://github.com/Prefrontal-AI/CRAM). Do not vendor that repository: it targets a GPU with Triton kernels. A small, attributed, CPU reimplementation in `experiments/cram/` is enough for a first artifact.

3. **Eval hygiene and synthetic constraint data.** Eval design should label tasks that a pairwise energy can represent exactly (Sudoku, Maze-Unique, N-Queens, graph coloring, and similar) as **pairwise-exact constraint tasks**, not as evidence that a general reasoner has arrived. ARC-style, multi-step abstraction, and open-ended agent work stay in a different bucket. The same energies can cheaply generate or filter constraint puzzles and negative samples for training or verification, which is useful even if we never ship a CRAM solver in the loop.

4. **Small-footprint helpers.** Sudoku's learned model is thousands of nonzero parameters; Maze-Unique uses 18; a hand-set energy can use three or fewer unique values. That is small enough for on-device helpers, rapid local experiments, and contributor laptops. This use is optional and comes after a working verifier prototype.

### Limitations (do not overstate)

- CRAM shines on **exact constraint satisfaction that fits a pairwise energy**. The paper is explicit: these puzzles belong to a class the energy can represent exactly, which is why hand-set weights solve them.
- It is **not a general reasoner**. Harder tasks (ARC-style, latent structure, open-ended agents) need hidden quantities or larger models. The authors' Maze-Hard extension adds a task-informed hidden layer and 11 extra parameters (29 in total) and solves 99.9% of that test set after less than 30 seconds on one GPU; they call learning such hidden quantities **in general, without task knowledge**, an open problem.
- Paper fit times and FLOP ratios were measured on **one H200**, with CRAM times reported as warm fits. They are not CPU numbers, and this RFC does not treat them as OFL measurements.
- A low energy is a useful verifier only if the threshold does not accept invalid boards. The paper reports zero wrong halts on Sudoku-Extreme and Maze-Unique for its energy-threshold rule; any OFL prototype must measure false accepts and false rejects itself.

### Concrete prototype

Land a self-contained CPU experiment at `experiments/cram/`, licensed Apache-2.0, that implements a CRAM-style pairwise energy on **8×8 N-Queens**:

- Generate the 92 solutions in the experiment itself (no external dataset).
- Train with pseudolikelihood on a training split of those solutions, starting from zero weights, without differentiating through search. Do not tell the energy the rules; relative-position parameter sharing, as in the paper's N-Queens prior, is allowed.
- Fit a stopping / accept threshold from training-solution energies.
- **Verifier mode** (the Phase 2-relevant artifact): score complete boards. Report true-accept rate on held-out solutions and false-accept rate on invalid boards (attacks, wrong queen count).
- **Optional solver mode**: clamp a subset of queens from a held-out solution and descend the energy. Report solve rate honestly, including failures.
- README records the command, machine class (CPU), wall time, parameter count, and the numbers from an actual run. A test covers generation, training, and the verifier's reject of an attacking board.
- Depend only on the Python standard library, so `python3` on a contributor machine is enough. This is not a reproduction of the paper's H200 results and must not be labeled as one.
- Later PRs, after this RFC is accepted and Phase 2 work exists, may wrap the verifier as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment or a reward hook for the pinned prime-rl / RLVR launchers. That wrap is not part of the first experiment.

The first version is in [`experiments/cram/`](../experiments/cram/README.md): on one laptop CPU (2026-11-14, seed 0, 3 of 12 symmetry orbits held out), the learned 113-parameter energy trained in 19.3 s, accepted 8 of 24 held-out solutions, falsely accepted 3 of 17,025 invalid boards (all diagonal-only permutations), and solved 34, 25 and 24 of 72 held-out puzzles with 2, 4 and 6 clues, while the hand-set two-weight energy accepted every held-out solution, rejected every invalid board and solved all 216 puzzles.

### Success metrics

The track is succeeding if all of the following hold:

1. This RFC is discussed in the open, and the `experiments/cram/` README reports a real CPU run (command, hardware, wall time, accept and reject rates) rather than a copied paper table.
2. The verifier's false-accept rate on a documented set of invalid N-Queens boards is zero, or the README states the failures. Held-out valid solutions are accepted.
3. Docs that add Sudoku-, maze-, or N-Queens-style evals label them as pairwise-exact constraint tasks and do not use them as a scale test for general reasoning.
4. After Phase 2 launchers exist, a follow-up PR either plugs a CRAM-style energy into a verifiers environment or prime-rl reward path, or records why that wrap is not worth doing.

### What stays fixed

- No new upstream pin is required for the CPU prototype.
- No change to `ofl post`, ledger schema, or network protocol in this RFC.
- Official CRAM code may be used as a reference (Apache-2.0) but is not vendored.

## Alternatives

| Option | Pros | Cons |
| --- | --- | --- |
| This proposal (research track + CPU prototype + later Phase 2 hook) | Cheap, public, honest about limits; gives Phase 2 a concrete verifier candidate; records eval hygiene early | A side track can distract from Run 0 if we let it |
| Vendor or wrap Prefrontal-AI/CRAM as-is | Faithful to the paper's GPU recipes | Needs NVIDIA GPU and Triton; violates "do not vendor upstream"; too large for a first OFL artifact |
| Wait until Phase 2 launchers exist | Less work now | We may ship naive evals and miss a cheap verifier design while the stack is still being designed |
| Treat CRAM as a general reasoner research bet | — | Contradicts the paper; would overclaim |

## Risks and security

- **Overclaiming.** Repeating the blog lede or Table 2's rounded 100% as if it were an OFL result. Mitigation: cite the abstract's 99.996% and 1,000× / 282× figures, always name the paper and hardware, and keep our CPU numbers in `experiments/cram/` separate.
- **Eval distortion.** A strong score on N-Queens or Sudoku could be marketed as reasoning progress. Mitigation: the pairwise-exact label in this RFC, and the hand-set-weight caution.
- **Licence.** Official CRAM and this repo are both Apache-2.0. Attribution belongs in the experiment README. Do not copy their data-prep or Triton kernels into the tree.
- **Scope creep.** An open-ended energy-based reasoner is out of scope. New latent or ARC-style work needs its own RFC.
- **Contributor cost.** The first prototype must stay CPU-only and dependency-light.

## Rollout

1. Merge this RFC as `draft`. Open a GitHub Discussion in the RFCs category that links the RFC and the paper.
2. Land `experiments/cram/` (same change or a follow-up PR) with a real CPU run recorded in its README.
3. After discussion, move the RFC to `proposed`, then `accepted` or `rejected` under the usual maintainer rule.
4. If accepted, open a tracking issue, add a short pointer from [docs/phase2-post.md](../docs/phase2-post.md), and only then consider a verifiers / prime-rl wrap.
5. Withdraw or supersede this RFC if the prototype cannot tell valid N-Queens boards from invalid ones, or if maintainers decide the track pulls effort off the ladder in RFC 0001.

## Out of scope

Replacing looped or transformer reasoners; changing RFC 0001's rungs; building `ofl post`; training on Sudoku-Extreme or Maze-Unique in the first prototype; claiming OFL reproduction of the paper's H200 timings; learning hidden structure for ARC or Maze-Hard without a later RFC.

## Open questions

- Should a CRAM-style verifier stay in `experiments/cram/`, or move to a small separate research repository once it is more than a prototype?
- When Phase 2 exists, is a `verifiers` environment the right hook, a direct reward in prime-rl or open-instruct RLVR, or only an offline filter for synthetic data?
- Which future OFL evals should be tagged pairwise-exact, and who owns that tag?
- Are there constraint checks in code or tool-use traces that actually fit a pairwise energy, or is N-Queens only a toy?

## References

- Benjamin Hoover, Hugh Blayney, and James C. R. Whittington, [Classical Energies Learn to Reason 1,000× Faster than Looped Models](https://arxiv.org/abs/2611.17309), arXiv:2611.17309, 2026. Official code: [Prefrontal-AI/CRAM](https://github.com/Prefrontal-AI/CRAM). Blog: [prefrontal.ai/news/cram](https://prefrontal.ai/news/cram).
- John J. Hopfield and David W. Tank, “Neural” computation of decisions in optimization problems, *Biological Cybernetics*, 1985.
- Julian Besag, Statistical analysis of non-lattice data, *The Statistician*, 1975 (pseudolikelihood).
- Junyeob Baek et al., Generative Recursive Reasoning, arXiv:2605.19376, 2026 (N-Queens and graph-coloring sets used in the CRAM paper).
- This repo: [docs/phase2-post.md](../docs/phase2-post.md), [RFC 0001](0001-milestone-ladder-and-first-diloco-proof-run.md).
