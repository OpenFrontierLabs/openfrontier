"""The Python training smokes, orchestrated by the Rust `ofl` CLI.

These drive the pinned olmo_core and torch trainers, which are Python; the rest
of the toolchain is Rust (src/). Run via `ofl smoke` / `ofl dev`, or directly as
`python -m smoke.olmo_tiny` / `python -m smoke.diloco_ref` from the repo root.
"""