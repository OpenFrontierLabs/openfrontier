//! `ofl-node`: keygen, sign, verify, peer. Output goes to stdout; the secret key is never printed.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::{ArgGroup, Args, Parser, Subcommand};
use ofl_node::envelope::{self, SignedReport};
use ofl_node::error::{NodeError, Result};
use ofl_node::identity::{self, Origin};
use ofl_node::peer::{self, BootstrapPeer, PeerConfig, Transport};
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
    /// Join the swarm: start a bootstrap peer (--listen) or join one (--bootstrap), and list peers.
    Peer(PeerArgs),
}

#[derive(Args)]
#[command(group(ArgGroup::new("role").required(true).multiple(true).args(["listen", "bootstrap"])))]
struct PeerArgs {
    /// Listen and serve as a Kademlia server (a bootstrap peer); prints its multiaddr.
    #[arg(long)]
    listen: bool,
    /// Address to listen on with --listen. Use 0.0.0.0 to accept connections from other machines.
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST), requires = "listen")]
    host: IpAddr,
    /// Port to listen on with --listen (UDP for QUIC, TCP with --tcp); 0 lets the OS pick.
    #[arg(long, default_value_t = 0, requires = "listen")]
    port: u16,
    /// Multiaddr of a peer to join, ending in /p2p/<peer id>; repeatable.
    #[arg(long, value_name = "MULTIADDR")]
    bootstrap: Vec<BootstrapPeer>,
    /// Use TCP (Noise, Yamux) instead of the default QUIC.
    #[arg(long)]
    tcp: bool,
    /// Seconds between peer listings.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    interval_secs: u64,
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
        Command::Peer(args) => run_peer(args, &key_path()?)?,
    }
    Ok(())
}

fn run_peer(args: &PeerArgs, key_file: &Path) -> Result<()> {
    let (keypair, _) = identity::load_or_create(key_file)?;
    let config = PeerConfig {
        transport: if args.tcp { Transport::Tcp } else { Transport::Quic },
        listen: args.listen.then(|| SocketAddr::new(args.host, args.port)),
        bootstrap: args.bootstrap.clone(),
        interval: Duration::from_secs(args.interval_secs),
    };
    peer::run(&config, keypair).map(|never| match never {})
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
