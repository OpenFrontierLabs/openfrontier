//! `ofl-node`: keygen, sign, verify. Output goes to stdout; the secret key is never printed.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use ofl_node::envelope::{self, SignedReport};
use ofl_node::error::{NodeError, Result};
use ofl_node::identity::{self, Origin};
use ofl_node::report::HardwareReport;

#[derive(Parser)]
#[command(version, about = "Open Frontier Node device identity")]
struct Cli {
    /// Key file. Default: <platform config dir>/ofl-node/device.key
    #[arg(long, global = true)]
    key_file: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create the device key if it does not exist; print its peer id either way.
    Keygen,
    /// Sign a hardware report of this machine; print the envelope JSON.
    Sign,
    /// Verify an envelope (a file, or stdin when omitted); exit 1 if it does not verify.
    Verify { file: Option<PathBuf> },
}

fn main() -> ExitCode {
    match run(&Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    let key_path = || cli.key_file.clone().map_or_else(identity::default_key_path, Ok);
    match &cli.command {
        Command::Keygen => {
            let path = key_path()?;
            let (keypair, origin) = identity::load_or_create(&path)?;
            let verb = if origin == Origin::Created { "created" } else { "exists" };
            println!("{verb} {}", path.display());
            println!("peer_id {}", keypair.public().to_peer_id());
            println!("public_key {}", hex::encode(keypair.public().encode_protobuf()));
        }
        Command::Sign => {
            let (keypair, _) = identity::load_or_create(&key_path()?)?;
            let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let signed = envelope::sign(&keypair, &HardwareReport::collect(now))?;
            println!("{}", serde_json::to_string_pretty(&signed)?);
        }
        Command::Verify { file } => {
            let text = read_input(file.as_deref())?;
            let envelope: SignedReport =
                serde_json::from_str(&text).map_err(|e| NodeError::BadEnvelope(e.to_string()))?;
            let verified = envelope::verify(&envelope)?;
            println!("ok peer_id {}", verified.peer_id);
            println!("{}", serde_json::to_string_pretty(&verified.report)?);
        }
    }
    Ok(())
}

fn read_input(file: Option<&Path>) -> Result<String> {
    if let Some(path) = file {
        return std::fs::read_to_string(path).map_err(|e| NodeError::io("read", path, e));
    }
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .map_err(|e| NodeError::io("read", Path::new("<stdin>"), e))?;
    Ok(text)
}
