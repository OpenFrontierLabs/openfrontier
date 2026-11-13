//! The swarm proof of concept: `ofl swarm ...` and `ofl node ...`.
//!
//! The coordinator and the nodes train real torch models over HTTP, so they are Python
//! (`smoke/smoke/swarm`). This module only forwards the arguments to them, unparsed, so the
//! Python argparse definition stays the single source of the flags.

use crate::error::Result;
use crate::repo::{RunSpec, find_root, python_module, run, smoke_python};
use crate::tasks::ensure_smoke_env;

/// Run `python -m smoke.swarm <command> <args...>` from the repo root.
///
/// # Errors
///
/// [`crate::error::OflError`] when the repo root or smoke environment is missing, or the
/// interpreter cannot start.
pub fn forward(command: &str, args: &[String]) -> Result<i32> {
    let root = find_root(None)?;
    ensure_smoke_env(&root)?;
    let mut forwarded = vec![command.to_owned()];
    forwarded.extend(args.iter().cloned());
    let command_line = python_module(&smoke_python(&root), "smoke.swarm", &forwarded);
    run(&RunSpec { argv: &command_line, cwd: Some(&root), env: None }, None)
}
