//! The `ofl` command line. One entry point; the Makefile forwards every target here.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::error::{OflError, Result};
use crate::repo::find_root;
use crate::{deploy, doctor, moe, phases, run0, tasks, upstreams};

/// Make-target descriptions, in `make help` order.
pub const TARGETS: [(&str, &str); 11] = [
    ("setup", "uv sync the Python smoke environment; cargo builds on demand"),
    ("dev", "longer, streaming CPU run of the real OLMo-core trainer (tiny model)"),
    ("check", "cargo fmt --check, clippy, ledger schema + example, configs and pins"),
    ("test", "cargo test"),
    ("fix", "cargo clippy --fix and cargo fmt"),
    ("build", "cargo build: the ofl binary into target/"),
    ("deploy", "ENV=dry-run lists what would be published; real publishing is not implemented"),
    ("smoke", "CPU, no downloads: real OLMo-core trainer run, then reference DiLoCo"),
    ("doctor", "GPU, CUDA, network, disk, clock and toolchain report"),
    ("clean", "remove caches, build output and scratch inside the repo"),
    ("help", "list targets"),
];

/// Open Frontier Labs command line.
#[derive(Parser)]
#[command(name = "ofl", about = "Open Frontier Labs command line.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Every `ofl` subcommand.
#[derive(Subcommand)]
enum Command {
    /// Uv sync the Python smoke environment; cargo builds on demand.
    Setup,
    /// Longer, streaming CPU run of the real OLMo-core trainer (tiny model).
    Dev,
    /// Cargo fmt --check, clippy, ledger schema + example, configs and pins.
    Check,
    /// Cargo test.
    Test,
    /// Cargo clippy --fix and cargo fmt.
    Fix,
    /// Cargo build: the ofl binary into target/.
    Build,
    /// List what would be published; real publishing is not implemented.
    Deploy {
        /// One of: dry-run, huggingface.
        #[arg(long)]
        env: String,
    },
    /// CPU smokes: real OLMo-core trainer run, then reference `DiLoCo`.
    Smoke,
    /// GPU, CUDA, network, disk, clock and toolchain report.
    Doctor,
    /// Remove caches, build output and scratch inside the repo.
    Clean,
    /// List targets.
    Help,
    /// Clone a pinned upstream into .cache/upstreams/NAME.
    Fetch {
        /// Upstream name from configs/upstreams.toml.
        name: String,
    },
    /// List pinned upstreams (configs/upstreams.toml).
    Upstreams,
    /// Phase 1 dense baseline (OLMo-core).
    Run0 {
        #[command(subcommand)]
        action: Run0Action,
    },
    /// Phase 1 `MoE` track (Megatron-LM, HF handbook).
    Moe {
        #[command(subcommand)]
        action: MoeAction,
    },
    /// Phase 2 post-training (not built).
    Post {
        #[command(subcommand)]
        action: UnbuiltAction,
    },
    /// Phase 3 `DiLoCo` islands (not built).
    Diloco {
        #[command(subcommand)]
        action: UnbuiltAction,
    },
}

/// Run 0 actions.
#[derive(Subcommand)]
enum Run0Action {
    /// Build and run the OLMo-core command for one node.
    Launch {
        /// Default: configs/phase1-dense/run0.toml.
        #[arg(long)]
        config: Option<PathBuf>,
        /// GPUs on this node (default 8).
        #[arg(long, default_value_t = 8)]
        nproc: u32,
        /// Interpreter with GPU torch and the pinned olmo-core installed.
        #[arg(long)]
        python: Option<String>,
        /// Run name override.
        #[arg(long)]
        run_name: Option<String>,
        /// Default: checkpoints/run0.
        #[arg(long)]
        save_folder: Option<PathBuf>,
        /// Default: .cache/run0/work.
        #[arg(long)]
        work_dir: Option<PathBuf>,
        /// Tokenised .npy shards, for the fineweb-edu config.
        #[arg(long)]
        data_glob: Option<String>,
        /// Print the command and exit.
        #[arg(long)]
        dry_run: bool,
    },
    /// Fetch the pin and have the upstream script accept every override.
    Verify,
}

/// `MoE` actions.
#[derive(Subcommand)]
enum MoeAction {
    /// Build and run the Megatron-LM command for one node.
    Launch {
        /// Default: configs/phase1-moe/handbook-moe-6p9b.toml.
        #[arg(long)]
        config: Option<PathBuf>,
        /// HF tokenizer directory (Megatron --tokenizer-model).
        #[arg(long)]
        tokenizer_dir: PathBuf,
        /// File of Megatron dataset prefixes.
        #[arg(long)]
        data_args_path: PathBuf,
        /// Default: .cache/moe/data-cache.
        #[arg(long)]
        data_cache_path: Option<PathBuf>,
        /// Checkpoint directory, default: checkpoints/moe.
        #[arg(long)]
        save: Option<PathBuf>,
        /// Interpreter inside the Megatron training environment.
        #[arg(long)]
        python: Option<String>,
        /// Stop after 120 iterations, as the handbook run does.
        #[arg(long)]
        setup_check: bool,
        /// Print the command and exit.
        #[arg(long)]
        dry_run: bool,
    },
    /// Fetch the pin and check every flag name against its source.
    Verify,
}

/// The only action unbuilt phases expose.
#[derive(Subcommand)]
enum UnbuiltAction {
    /// Not built.
    Launch,
}

/// Parse, dispatch, report. Returns the process exit code.
#[must_use]
pub fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(cli.command) {
        Ok(code) => exit_code(code),
        Err(e) => {
            eprintln!("ofl: {e}");
            exit_code(i32::from(e.code))
        }
    }
}

/// Map an exit code to the process type.
fn exit_code(code: i32) -> ExitCode {
    u8::try_from(code).map_or(ExitCode::FAILURE, ExitCode::from)
}

/// Run one command.
fn dispatch(command: Command) -> Result<i32> {
    match command {
        Command::Setup => tasks::setup(),
        Command::Dev => tasks::dev(),
        Command::Check => tasks::check(),
        Command::Test => tasks::test(),
        Command::Fix => tasks::fix(),
        Command::Build => tasks::build(),
        Command::Deploy { env } => deploy::deploy(&env),
        Command::Smoke => tasks::smoke(),
        Command::Doctor => doctor::doctor(None),
        Command::Clean => tasks::clean(),
        Command::Help => Ok(help()),
        Command::Fetch { name } => fetch(&name),
        Command::Upstreams => list_upstreams(),
        Command::Run0 { action } => dispatch_run0(action),
        Command::Moe { action } => dispatch_moe(action),
        Command::Post { action: UnbuiltAction::Launch } => phases::not_built("post"),
        Command::Diloco { action: UnbuiltAction::Launch } => phases::not_built("diloco"),
    }
}

/// Run 0 dispatch.
fn dispatch_run0(action: Run0Action) -> Result<i32> {
    match action {
        Run0Action::Launch {
            config,
            nproc,
            python,
            run_name,
            save_folder,
            work_dir,
            data_glob,
            dry_run,
        } => run0::launch(&run0::LaunchArgs {
            config,
            nproc,
            python,
            run_name,
            save_folder,
            work_dir,
            data_glob,
            dry_run,
        }),
        Run0Action::Verify => run0::verify(&find_root(None)?),
    }
}

/// `MoE` dispatch.
fn dispatch_moe(action: MoeAction) -> Result<i32> {
    match action {
        MoeAction::Launch {
            config,
            tokenizer_dir,
            data_args_path,
            data_cache_path,
            save,
            python,
            setup_check,
            dry_run,
        } => moe::launch(&moe::LaunchArgs {
            config,
            tokenizer_dir,
            data_args_path,
            data_cache_path,
            save,
            python,
            setup_check,
            dry_run,
        }),
        MoeAction::Verify => moe::verify(&find_root(None)?),
    }
}

/// Print the target list and the extra commands.
fn help() -> i32 {
    println!("targets (make <target> or ofl <target>):");
    for (name, text) in TARGETS {
        println!("  {name:<8} {text}");
    }
    println!("\nother commands:");
    println!("  fetch NAME          clone a pinned upstream into .cache/upstreams/NAME");
    println!("  upstreams           list pinned upstreams (configs/upstreams.toml)");
    println!(
        "  run0 launch|verify  phase 1 dense: OLMo-core command for one node / check it upstream"
    );
    println!(
        "  moe launch|verify   phase 1 MoE: Megatron-LM command for one node / check its flags"
    );
    println!("  post launch         phase 2: not built");
    println!("  diloco launch       phase 3: not built");
    0
}

/// Fetch one pinned upstream.
fn fetch(name: &str) -> Result<i32> {
    let root = find_root(None)?;
    let ups = upstreams::load(&root)?;
    let up = ups.iter().find(|u| u.name == name).ok_or_else(|| {
        let known: Vec<&str> = ups.iter().map(|u| u.name.as_str()).collect();
        OflError::misuse(format!("unknown upstream {name:?}; known: {}", known.join(", ")))
    })?;
    println!("{}", upstreams::ensure_checkout(&root, up)?.display());
    Ok(0)
}

/// List every pinned upstream.
fn list_upstreams() -> Result<i32> {
    for up in upstreams::load(&find_root(None)?)? {
        let license = up.license.split(" (").next().unwrap_or(&up.license);
        println!("{:<24} {:<10} {}  {license}", up.name, up.reference, up.sha);
    }
    Ok(0)
}
