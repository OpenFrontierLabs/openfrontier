# ofl-node

The device identity of an Open Frontier Node: an Ed25519 keypair that is generated on first run,
kept on disk, and used to sign a small hardware report. This is the first piece of the node
([docs/ARCHITECTURE.md §2](../docs/ARCHITECTURE.md#2-open-frontier-node)); it does not run workers.
`peer` (below) is the first piece of layer 3: it finds other peers and carries no payload yet.
It is a separate crate in the repo's Cargo workspace, so `make check` and `make test` at the
repo root cover it (`cargo clippy --workspace`, `cargo test --workspace`).

The key is `libp2p-identity`'s Ed25519 keypair, so the same key becomes the node's libp2p peer id.

## Commands

From `node/` (or `cargo run -p ofl-node --` from the repo root):

```
cargo run -- keygen                  # create the key if missing; print peer id and public key
cargo run -- sign > report.json      # sign a hardware report of this machine
cargo run -- verify report.json      # exit 0 and print the report if the signature holds
cargo run -- sign | cargo run -- verify   # verify reads stdin when no file is given
```

`--key-file <path>` (before or after the subcommand) overrides the key location.
`peer` is described in [Peer](#peer).

`keygen` is idempotent: if the key file exists it is left untouched and its peer id is printed.
`sign` creates the key too if this is the first run.

## Peer

`peer` joins a swarm with rust-libp2p 0.57: Identify and Kademlia over QUIC (default) or TCP with
Noise and Yamux (`--tcp`). Its libp2p identity is the device key, so its peer id is the one
`keygen` prints. Two terminals on one machine, from `node/`:

```
# terminal 1: a bootstrap peer on 127.0.0.1, OS-chosen UDP port
cargo run -- peer --listen
#   peer_id 12D3Koo...A
#   listening /ip4/127.0.0.1/udp/54321/quic-v1/p2p/12D3Koo...A

# terminal 2: a second peer with its own key, joined to the first (paste the printed line)
cargo run -- --key-file /tmp/peer2.key peer --bootstrap /ip4/127.0.0.1/udp/54321/quic-v1/p2p/12D3Koo...A
```

Without `--key-file` both terminals would use the same device key and the same peer id, so the
second peer needs its own key file. Add `--tcp` to every peer for TCP; the address then looks like
`/ip4/127.0.0.1/tcp/54321/p2p/12D3Koo...A`. A peer speaks one transport per run, so a `--tcp` peer
cannot dial a QUIC address: it prints `error: dial ...` on stderr, keeps retrying, and lists the
bootstrap peer as `connected=false routing_table=true` (Kademlia keeps an address it has not reached).

| Flag | Meaning |
|---|---|
| `--listen` | Listen and act as a Kademlia server. At least one of `--listen` and `--bootstrap` is required. |
| `--host <ip>` | Address to listen on with `--listen`. Default `127.0.0.1`; use `0.0.0.0` to accept other machines. |
| `--port <n>` | Port to listen on (UDP, or TCP with `--tcp`). Default `0`: the OS picks and `listening` shows it. |
| `--bootstrap <multiaddr>` | Peer to join; must end in `/p2p/<peer id>`. Repeatable. Added to the routing table, then Kademlia bootstraps. |
| `--tcp` | TCP instead of QUIC. |
| `--interval-secs <n>` | Seconds between listings. Default `5`. |

A peer started with `--listen --bootstrap <addr>` serves and joins at once. Without `--listen` it
listens nowhere and is a Kademlia client: other peers can reach it only over the connection it
opened.

### What it prints

```
peer_id 12D3Koo...B
peers 1
peer 12D3Koo...A connected=true routing_table=true
```

`peer_id` and `listening` print once (each `listening` line is a full multiaddr to pass to
`--bootstrap`); `peers <n>` and one `peer` line per known peer repeat every interval. A peer is
*known* if it completed Identify with this node over a connection that is still open, or sits in
this node's Kademlia routing table. `connected` says whether a connection to it is open now;
`routing_table` says whether Kademlia holds it. The list is what this node has seen, not the whole
swarm: Kademlia's queries and routing-table fills widen it as more peers join, but nothing here
guarantees completeness.

Kademlia admits only peers that announce its protocol, i.e. servers. So in the two-terminal example
the second peer lists the first with `routing_table=true`, and the first lists the second with
`routing_table=false` (connected and identified, but a client). Start the second peer with
`--listen` too and both show `routing_table=true`.

The swarm uses its own protocol names (`/ofl/kad/1.0.0`, `/ofl/peer/0.1.0`), not the public IPFS
DHT's `/ipfs/kad/1.0.0`, so it cannot join that network by accident. While a peer has identified
nobody it repeats the join (routing-table entry plus Kademlia bootstrap) every interval, so a
joiner started before its bootstrap peer connects once that peer is up.

## Key file

- Default path: `<platform config dir>/ofl-node/device.key` (on Linux `~/.config/ofl-node/device.key`).
- Content: the libp2p protobuf encoding of the keypair, raw bytes. It contains the secret key;
  nothing in this crate prints it.
- Unix: the file is created `0600` in a `0700` directory, and any file with group or other
  permission bits is refused with a `chmod 600` hint. Windows has no mode bits; the file inherits
  the profile directory's ACL.
- A file that does not decode as an Ed25519 keypair is an error, never silently replaced: move it
  aside to make a new identity (the old peer id is lost).

## Signed envelope

`sign` prints `{report, public_key, signature}`:

| Field | Value |
|---|---|
| `report` | A JSON **string** holding the exact bytes that were signed. |
| `public_key` | Hex of the libp2p protobuf encoding of the public key. |
| `signature` | Hex of the Ed25519 signature over `"ofl-node/hardware-report/v1\n"` followed by the `report` bytes. |

The report is compact JSON (no whitespace, fields in the order below) produced once by the signer.
Verification checks the signature over the `report` string exactly as received and only then
parses it, so no canonicalisation has to agree between signer and verifier. A report that is
re-formatted, even to an equal JSON value, fails verification. The domain prefix keeps the
signature from being valid for any other message the same key signs.

Report fields, all read directly from the machine:

| Field | Meaning |
|---|---|
| `schema` | `ofl.node.hardware-report/v1` |
| `generated_at_unix` | Seconds since the Unix epoch, from the local clock (not attested). |
| `os`, `arch` | `std::env::consts::OS` and `ARCH` of the build. |
| `cpu_threads` | Logical CPUs available to the process, or `null`. |
| `memory_bytes` | `MemTotal` from `/proc/meminfo` in bytes; `null` off Linux. |

There are no GPU fields yet: `ofl doctor` reads GPUs through `nvidia-smi` and torch, and this
report carries nothing the crate cannot read itself.

`verify` proves the report was signed by the holder of the embedded public key. It does not say
whose key that is, and the report contents are self-declared by the node. Compare the printed
`peer_id` with one you already trust.
