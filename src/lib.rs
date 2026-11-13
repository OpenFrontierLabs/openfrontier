//! Open Frontier Labs command line: doctor, smoke orchestration, Run 0 and `MoE`
//! launchers, pinned-upstream fetches, ledger and config checks.
//!
//! The two training smokes are Python (`smoke/`), because they drive the pinned
//! `olmo_core` and `torch` trainers; everything else is this crate.

pub mod cli;
pub mod configs;
pub mod deploy;
pub mod doctor;
pub mod error;
pub mod ledger;
pub mod moe;
pub mod phases;
pub mod repo;
pub mod run0;
pub mod swarm;
pub mod tasks;
pub mod upstreams;
