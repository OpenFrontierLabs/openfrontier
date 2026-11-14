# RFCs

An RFC (request for comments) is how Open Frontier Labs makes decisions that are hard to reverse. It is a markdown file in this folder, opened as a pull request and discussed in the open.

## When you need one

Write an RFC when a change is hard to undo once people depend on it:

- protocol messages between nodes, and anything a node signs;
- the compute ledger schema or how credit is computed;
- licences of anything we release;
- layer boundaries, or a new repository;
- the milestone ladder, or what a run must publish to count as done.

Everything else is a normal issue and pull request. When in doubt, open an [Ideas discussion](https://github.com/OpenFrontierLabs/openfrontier/discussions/categories/ideas) first and ask.

## How it works

1. **Discuss first (optional).** Post the idea in the [RFCs discussion category](https://github.com/OpenFrontierLabs/openfrontier/discussions/categories/rfcs) or open an issue with the RFC form to see if there is interest.
2. **Write it.** Copy [`0000-template.md`](0000-template.md) to `NNNN-short-slug.md`, using the next free number, and fill in every section. Set the status to `draft`.
3. **Open a pull request** titled `RFC NNNN: <title>`, labelled `type/rfc` and the affected `layer/*` labels. Set the status to `proposed` when it is ready for a decision.
4. **Review in the open.** Maintainers of the affected layers comment on the PR. Anyone may comment. Substantive changes are new commits, so the history stays readable.
5. **Decide.** The RFC is accepted when the maintainers of the affected layers agree and nobody has an unresolved objection. If they cannot agree, the founding steward decides and writes the reason into the RFC. The PR merges with status `accepted`, or with `rejected` and the reason, so rejected ideas stay on record.
6. **Build it.** Accepted RFCs get a tracking issue on the [roadmap](https://github.com/orgs/OpenFrontierLabs/projects/1). Implementation PRs link the RFC.

There is no minimum comment period. Expect about a week for anything that affects more than one layer, so people in every time zone can read it.

## Statuses

| Status | Meaning |
| --- | --- |
| `draft` | Being written; feedback welcome, not ready for a decision. |
| `proposed` | Ready for a decision. |
| `accepted` | Agreed; implementation may start. |
| `rejected` | Declined, with the reason recorded. Kept for history. |
| `superseded` | Replaced by a later RFC, which is linked. |
| `withdrawn` | Abandoned by its author before a decision. |

Numbers are permanent and never reused. Filenames do not change after merge.

## Index

| RFC | Title | Status |
| --- | --- | --- |
| [0001](0001-milestone-ladder-and-first-diloco-proof-run.md) | Milestone ladder and first DiLoCo proof run | proposed |
| [0002](0002-energy-based-constraint-reasoners-cram.md) | Energy-based constraint reasoners (CRAM) as verifiers and an eval-hygiene track | draft |
