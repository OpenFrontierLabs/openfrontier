//! A minimal swarm peer: rust-libp2p with QUIC (or TCP), Identify and Kademlia.
//!
//! The libp2p identity is the node's device key, so the peer id printed here is the one
//! `ofl-node keygen` prints. The peer carries no payload yet; it only finds other peers.
//!
//! Output, one record per line on stdout (stderr carries errors):
//!
//! - `peer_id <id>` once at start;
//! - `listening <multiaddr>` for each address the peer listens on, ending in `/p2p/<id>`, so the
//!   line can be passed to another peer's `--bootstrap`;
//! - every interval, `peers <n>` followed by one `peer <id> connected=<bool> routing_table=<bool>`
//!   line per known peer.
//!
//! A peer is *known* when it completed Identify with us (it was connected and answered) or sits
//! in our Kademlia routing table. `connected` is whether a connection to it is open right now;
//! `routing_table` is whether Kademlia holds it in a bucket. Kademlia only admits peers that
//! announce its protocol, so a client-mode peer (one run without `--listen`) shows up on the
//! bootstrap peer as connected but not in its routing table.

use std::collections::{BTreeSet, HashSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::str::FromStr;
use std::time::Duration;

use libp2p::futures::StreamExt;
use libp2p::identity::Keypair;
use libp2p::multiaddr::Protocol;
use libp2p::swarm::{NetworkBehaviour, StreamProtocol, SwarmEvent};
use libp2p::{Multiaddr, PeerId, Swarm, SwarmBuilder, identify, kad, noise, tcp, yamux};
use tokio::time::{Instant, interval_at};

use crate::error::{NodeError, Result};

/// Kademlia protocol name. Not `/ipfs/kad/1.0.0`: this swarm must never join the public IPFS DHT.
const KAD_PROTOCOL: StreamProtocol = StreamProtocol::new("/ofl/kad/1.0.0");
/// Identify protocol version string, shared by every peer of this swarm.
const IDENTIFY_PROTOCOL: &str = "/ofl/peer/0.1.0";
/// How long an idle connection is kept. The libp2p default (zero) would drop a joiner that is not
/// a Kademlia server as soon as its first queries finish.
const IDLE_CONNECTION_TIMEOUT: Duration = Duration::from_secs(60);

/// Which libp2p transport the peer speaks. Exactly one is used per run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// QUIC v1 over UDP (TLS 1.3 and multiplexing built in).
    Quic,
    /// TCP with Noise and Yamux.
    Tcp,
}

/// A bootstrap peer: its id and a dialable address (without the `/p2p/<id>` suffix).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapPeer {
    pub peer_id: PeerId,
    pub addr: Multiaddr,
}

impl FromStr for BootstrapPeer {
    type Err = NodeError;

    /// Parse a multiaddr that ends in `/p2p/<peer id>`, the form `listening` lines print.
    fn from_str(text: &str) -> Result<Self> {
        let bad = |reason: String| NodeError::BadBootstrap { addr: text.to_owned(), reason };
        let mut addr: Multiaddr = text.parse().map_err(|e| bad(format!("not a multiaddr: {e}")))?;
        match addr.pop() {
            Some(Protocol::P2p(peer_id)) if !addr.is_empty() => Ok(Self { peer_id, addr }),
            _ => Err(bad("it has no address in front of a final /p2p/<peer id>".to_owned())),
        }
    }
}

/// What to run.
#[derive(Debug, Clone)]
pub struct PeerConfig {
    pub transport: Transport,
    /// Listen here and act as a Kademlia server. `None`: dial only, as a Kademlia client.
    pub listen: Option<SocketAddr>,
    /// Peers to join; may be empty when `listen` is set.
    pub bootstrap: Vec<BootstrapPeer>,
    /// Delay between peer listings.
    pub interval: Duration,
}

#[derive(NetworkBehaviour)]
struct Behaviour {
    identify: identify::Behaviour,
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
}

impl Behaviour {
    fn new(key: &Keypair, server: bool) -> Self {
        let local = key.public().to_peer_id();
        let identify = identify::Behaviour::new(
            identify::Config::new(IDENTIFY_PROTOCOL.to_owned(), key.public())
                .with_agent_version(format!("ofl-node/{}", env!("CARGO_PKG_VERSION"))),
        );
        let mut kademlia = kad::Behaviour::with_config(
            local,
            kad::store::MemoryStore::new(local),
            kad::Config::new(KAD_PROTOCOL),
        );
        // Auto mode would stay client until an external address is confirmed, which a bootstrap
        // peer on a LAN or loopback never does; a listening peer is a server by declaration.
        kademlia.set_mode(Some(if server { kad::Mode::Server } else { kad::Mode::Client }));
        Self { identify, kademlia }
    }
}

/// Run the peer until the process is killed; only an error ends it.
///
/// Prints as described in the module docs.
///
/// # Errors
///
/// [`NodeError::Network`] when the transport cannot be built, the listen address cannot be bound,
/// the runtime cannot start, or the swarm stops.
pub fn run(config: &PeerConfig, keypair: Keypair) -> Result<Infallible> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| NodeError::Network(format!("cannot start the async runtime: {e}")))?;
    runtime.block_on(run_async(config, keypair))
}

async fn run_async(config: &PeerConfig, keypair: Keypair) -> Result<Infallible> {
    let mut peer = Peer { swarm: build_swarm(keypair, config)?, identified: HashSet::new() };
    println!("peer_id {}", peer.swarm.local_peer_id());
    if let Some(listen) = config.listen {
        let addr = listen_multiaddr(config.transport, listen);
        peer.swarm
            .listen_on(addr.clone())
            .map_err(|e| NodeError::Network(format!("cannot listen on {addr}: {e}")))?;
    }
    if let Some(own) = config.bootstrap.iter().find(|b| b.peer_id == *peer.swarm.local_peer_id()) {
        return Err(NodeError::BadBootstrap {
            addr: own.addr.to_string(),
            reason: "that is this node's own peer id".to_owned(),
        });
    }
    peer.join(config)?;

    let mut ticker = interval_at(Instant::now() + config.interval, config.interval);
    loop {
        tokio::select! {
            event = peer.swarm.select_next_some() => peer.on_event(event),
            _ = ticker.tick() => {
                peer.print_peers();
                // The first query can lose a race with a bootstrap peer that is still starting.
                if peer.identified.is_empty() {
                    peer.join(config)?;
                }
            }
        }
    }
}

fn build_swarm(keypair: Keypair, config: &PeerConfig) -> Result<Swarm<Behaviour>> {
    let server = config.listen.is_some();
    let builder = SwarmBuilder::with_existing_identity(keypair).with_tokio();
    let behaviour = move |key: &Keypair| Behaviour::new(key, server);
    let swarm_config =
        |c: libp2p::swarm::Config| c.with_idle_connection_timeout(IDLE_CONNECTION_TIMEOUT);
    let network =
        |e: &dyn std::fmt::Display| NodeError::Network(format!("cannot build transport: {e}"));
    let swarm = match config.transport {
        Transport::Quic => builder
            .with_quic()
            .with_behaviour(behaviour)
            .map_err(|e| network(&e))?
            .with_swarm_config(swarm_config)
            .build(),
        Transport::Tcp => builder
            .with_tcp(tcp::Config::default(), noise::Config::new, yamux::Config::default)
            .map_err(|e| network(&e))?
            .with_behaviour(behaviour)
            .map_err(|e| network(&e))?
            .with_swarm_config(swarm_config)
            .build(),
    };
    Ok(swarm)
}

fn listen_multiaddr(transport: Transport, socket: SocketAddr) -> Multiaddr {
    let base = Multiaddr::empty().with(Protocol::from(socket.ip()));
    match transport {
        Transport::Quic => base.with(Protocol::Udp(socket.port())).with(Protocol::QuicV1),
        Transport::Tcp => base.with(Protocol::Tcp(socket.port())),
    }
}

struct Peer {
    swarm: Swarm<Behaviour>,
    /// Peers that answered Identify on a connection that is still open.
    identified: HashSet<PeerId>,
}

impl Peer {
    /// Put the bootstrap peers in the routing table and start a Kademlia bootstrap query. Adding
    /// an address that is already known is a no-op, so repeating the join is safe; it also
    /// guarantees the table is not empty when the query starts.
    fn join(&mut self, config: &PeerConfig) -> Result<()> {
        if config.bootstrap.is_empty() {
            return Ok(());
        }
        let kademlia = &mut self.swarm.behaviour_mut().kademlia;
        for boot in &config.bootstrap {
            kademlia.add_address(&boot.peer_id, boot.addr.clone());
        }
        kademlia
            .bootstrap()
            .map(drop)
            .map_err(|e| NodeError::Network(format!("kademlia bootstrap: {e}")))
    }

    fn on_event(&mut self, event: SwarmEvent<BehaviourEvent>) {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                println!("listening {}", address.with(Protocol::P2p(*self.swarm.local_peer_id())));
            }
            SwarmEvent::Behaviour(BehaviourEvent::Identify(identify::Event::Received {
                peer_id,
                info,
                ..
            })) => {
                // Kademlia learns a dialer's address only from Identify, and only for peers that
                // speak its protocol (that is, servers).
                if info.protocols.contains(&KAD_PROTOCOL) {
                    for addr in info.listen_addrs {
                        self.swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
                    }
                }
                self.identified.insert(peer_id);
            }
            SwarmEvent::ConnectionClosed { peer_id, num_established: 0, .. } => {
                self.identified.remove(&peer_id);
            }
            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                eprintln!(
                    "error: dial {}: {error}",
                    peer_id.map_or_else(|| "?".to_owned(), |p| p.to_string())
                );
            }
            _ => {}
        }
    }

    fn routing_table(&mut self) -> BTreeSet<PeerId> {
        let mut peers = BTreeSet::new();
        for bucket in self.swarm.behaviour_mut().kademlia.kbuckets() {
            peers.extend(bucket.iter().map(|entry| *entry.node.key.preimage()));
        }
        peers
    }

    fn print_peers(&mut self) {
        let table = self.routing_table();
        let known: BTreeSet<PeerId> = self.identified.iter().chain(&table).copied().collect();
        println!("peers {}", known.len());
        for id in known {
            println!(
                "peer {id} connected={} routing_table={}",
                self.swarm.is_connected(&id),
                table.contains(&id)
            );
        }
    }
}
