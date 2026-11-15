# Runbook: bootstrap and relay node on a small VPS

How to run a libp2p circuit relay v2 and bootstrap peer on a small Linux VPS, so peers behind home NAT have a public address to dial. This is the Rung 1 network piece: [ARCHITECTURE §3](../ARCHITECTURE.md#3-decentralized-network), [RFC 0001](../../rfcs/0001-milestone-ladder-and-first-diloco-proof-run.md) (one bootstrap peer, and a relay if a worker is behind NAT).

## Verification status

**This runbook has not been run on a public VPS.** No provider was rented, no provider firewall was configured, and nothing was tested across a real home NAT. The issue's definition of done asks for the runbook to be followed once on a real provider and for that provider and OS to be named. **That item is not met.** Whoever does it first should fill in the table below and delete this paragraph.

What was checked, and where:

| Step | Status | Where |
| --- | --- | --- |
| Build `relay-server-example` at the pinned tag | Verified | Ubuntu 24.04.5 LTS, Linux 6.8.0-147-generic, aarch64, rustc 1.99.1. A fleet host picked by `heavy-build-route`, not a VPS. |
| `--help`, required flags | Verified | Same host; output quoted below |
| Run on a high port, print the peer id | Verified | Same host, loopback, port 41234 |
| Same seed gives the same peer id after a restart; another seed gives another | Verified | Same host |
| A second process connects over TCP and over QUIC | Verified | Same host, loopback, using the upstream `ping` example (TCP) and a small variant of it with the QUIC transport added (see [Health](#how-to-tell-it-is-healthy)) |
| Relay v2: a client gets a reservation, another client opens a circuit through the relay | Verified | Same host, loopback, upstream `dcutr` example. On loopback the direct-connection upgrade also succeeded; that says nothing about NAT. |
| Relay limits (2 minutes, 128 KiB per circuit) | Verified | Seen in the clients' `Limit { duration: Some(120s), data_in_bytes: Some(131072) }` log line, and in the tag's `relay::Config::default` |
| `ofl-relay.service` passes `systemd-analyze verify` | Verified | NixOS, systemd 260.5, run against the exact unit text below installed in a throwaway `--root`. Not enabled, not started. |
| The unit actually starts the relay under `DynamicUser` and the hardening options | **Not verified** | |
| Public IPv4 reachability, provider firewall, host firewall commands | **Not verified** | |
| NAT traversal between two real home networks | **Not verified** | |
| IPv6 (`--use-ipv6 true`) | **Not verified**; read from source only | |
| Upgrade procedure end to end | **Partly**: only the "same seed, same peer id after restart" part | |
| Resource sizing for a VPS | **Not measured** | |

## What you are running

The peer binary has not landed: this repo has a device key (`ofl-node`, [ARCHITECTURE §2](../ARCHITECTURE.md#2-open-frontier-node)) but no libp2p peer. Until it does, run the upstream `relay-server` example from rust-libp2p, pinned like every upstream here by tag and commit together:

| | |
| --- | --- |
| Repository | `https://github.com/libp2p/rust-libp2p` |
| Tag | `libp2p-v0.56.0` |
| Commit | `70082df7e6181722630eabc5de5373733aac9a21` |
| Crate | `examples/relay-server` (package `relay-server-example`, `publish = false`) |
| Minimum Rust | 1.83.0 (the tag's `rust-version`) |

What the example does, read from `examples/relay-server/src/main.rs` at that commit:

- Relay v2 server with the library's default limits, plus ping and identify.
- Listens on one port number on all interfaces of one IP family: TCP on `<port>` and QUIC (`udp/<port>/quic-v1`) on `<port>`. Both protocols use the same number.
- It is **not** a Kademlia or rendezvous node. "Bootstrap" here means only a stable public address that peers dial first. Anything richer comes with the peer binary.
- It prints behaviour events to stdout and does not print its own peer id there. The peer id appears in a tracing line when `RUST_LOG=info` is set (shown below).

### Identity: read this before publishing an address

The example takes `--secret-key-seed <u8>` and builds the Ed25519 key from 32 bytes whose first byte is the seed and the rest zero. Two consequences:

- **The peer id is stable across restarts** (verified: seed 7 gave the same id before and after a restart). A published multiaddr stays valid until you change the seed. No key file is needed or supported.
- **The key is not secret.** There are only 256 possible identities and anyone can compute the private key for each. The peer id names the relay but does not authenticate it: anybody can run a node with your peer id. Circuit traffic is encrypted end to end between the two peers, so an impersonating relay should be able to refuse traffic or observe metadata but not read payloads; this is expected from the design and was not tested.

Never use the seed as a credential. Choose any seed from 0 to 255; two relays with the same seed have the same peer id. Seed `7` is used below as a test value so you can check that your build matches: it must give `12D3KooWLnZUpcaBwbz9uD1XsyyHnbXUrJRmxnsMiRnuCmvPix67`. A persisted random key needs a custom binary; the peer binary should do that and supersede this section (`ofl-node` already stores an Ed25519 device key in libp2p's encoding).

### What the relay will and will not carry

The defaults at the pinned tag (`relay::Config::default`): 128 reservations (4 per peer), 16 circuits (4 per peer), reservations last 1 hour, **each circuit is cut after 2 minutes or 128 KiB**, plus per-peer and per-IP rate limits. The example exposes no flag to change them. A relayed connection is therefore a way to introduce two peers and let them hole-punch, not a path for pseudo-gradients. If hole punching fails between two peers, they cannot train over this relay.

There is no allow-list: any peer that can reach the port can ask for a reservation, within the limits above.

## 1. The VPS

Any provider. Requirements:

- A Linux distribution with systemd, one public IPv4 address (IPv6 optional), SSH access with key authentication, and a non-root sudo user.
- Inbound TCP and UDP on one port (below). Nothing else needs to be open apart from SSH.
- Sizing: not measured. The relay holds no model data; start with the smallest plan and watch it.

Use documentation addresses when you write notes: `192.0.2.10` (IPv4) and `2001:db8::10` (IPv6) stand for your server below. Do not commit a real address of a host you do not want to publish.

## 2. Build

Build on the VPS if it can compile Rust. Otherwise build on a machine with the same CPU architecture and the same distribution release, or at least a glibc no newer than the VPS's, then copy the binary: matching only the OS family does not make a dynamically linked binary start on an older VPS.

```sh
git clone https://github.com/libp2p/rust-libp2p
cd rust-libp2p
git checkout libp2p-v0.56.0
git rev-parse HEAD        # must print 70082df7e6181722630eabc5de5373733aac9a21
cargo build --release --locked -p relay-server-example
./target/release/relay-server-example --help
```

`--help` at the pinned commit prints exactly these options; do not pass any other:

```
Usage: relay-server-example [OPTIONS] --secret-key-seed <SECRET_KEY_SEED> --port <PORT>

Options:
      --use-ipv6 <USE_IPV6>
          Determine if the relay listen on ipv6 or ipv4 loopback address. the default is ipv4 [possible values: true, false]
      --secret-key-seed <SECRET_KEY_SEED>
          Fixed value to generate deterministic peer id
      --port <PORT>
          The port used to listen on all interfaces
  -h, --help
          Print help
```

(The help text says "loopback"; the source binds the unspecified address, `0.0.0.0` or `::`. `--use-ipv6 true` selects IPv6 instead of IPv4: one process listens on one family.)

## 3. Try it by hand first

Use a high port and the test seed. This part was run on loopback (see [status](#verification-status)).

```sh
RUST_LOG=info ./target/release/relay-server-example --port 41234 --secret-key-seed 7
```

Output observed, trimmed to the loopback and one interface, with the test host's interface address replaced by a documentation address. You get one pair of lines per interface of your machine:

```
INFO libp2p_swarm: local_peer_id=12D3KooWLnZUpcaBwbz9uD1XsyyHnbXUrJRmxnsMiRnuCmvPix67
Listening on /ip4/127.0.0.1/tcp/41234
Listening on /ip4/192.0.2.10/tcp/41234
Listening on /ip4/127.0.0.1/udp/41234/quic-v1
Listening on /ip4/192.0.2.10/udp/41234/quic-v1
```

`ss -ltnu` showed `tcp LISTEN 0.0.0.0:41234` and `udp UNCONN 0.0.0.0:41234`. Stop it with Ctrl-C, then continue.

## 4. Firewall

Open one port number for both protocols, for example 4001. Ports above 1023 need no privileges.

| Protocol | Port | Why |
| --- | --- | --- |
| TCP | 4001 | TCP transport (Noise + Yamux) |
| UDP | 4001 | QUIC (`/udp/4001/quic-v1`) |

Allow both **in two places**: the provider's network firewall or security group, if it has one, and the host firewall, if one is enabled. A rule for TCP alone leaves QUIC dead while TCP works, so test both (Health, below). The relay accepts the connection from clients, so no outbound rule beyond the default is needed.

Host firewall, if you use `ufw` (not run for this runbook):

```sh
sudo ufw allow 4001/tcp
sudo ufw allow 4001/udp
```

## 5. Install and run under systemd

Copy the binary, create the settings file, install the unit. Not run on a VPS; the unit text itself passed `systemd-analyze verify`.

```sh
sudo install -m 0755 target/release/relay-server-example /usr/local/bin/relay-server-example
sudo install -d -m 0755 /etc/ofl-relay
printf 'RELAY_PORT=4001\nRELAY_SEED=7\n' | sudo tee /etc/ofl-relay/relay.env >/dev/null   # replace 7 with your seed
```

`/etc/systemd/system/ofl-relay.service`:

```ini
[Unit]
Description=libp2p circuit relay v2 and bootstrap peer (Open Frontier Labs)
Documentation=https://github.com/OpenFrontierLabs/openfrontier/blob/main/docs/runbooks/bootstrap-relay.md
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
DynamicUser=yes
EnvironmentFile=/etc/ofl-relay/relay.env
Environment=RUST_LOG=info
ExecStart=/usr/local/bin/relay-server-example --port ${RELAY_PORT} --secret-key-seed ${RELAY_SEED}
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
RestrictAddressFamilies=AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
```

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now ofl-relay
```

Notes:

- `RUST_LOG=info` is what makes the peer id appear in the journal. Without it the process prints no peer id.
- The relay writes one line per ping and identify event for every connected peer, so the journal grows with the number of peers. Cap it with `SystemMaxUse=` in `/etc/systemd/journald.conf`.
- `DynamicUser=yes` and the sandboxing options are untested against this binary. If the service fails to start, remove the sandboxing options one at a time, starting from the bottom of the unit, and report which one broke it here.
- To verify the unit text before installing: place the unit under a scratch root with stub `*.target` units and the binary at its path, then `systemd-analyze --root=<dir> verify /etc/systemd/system/ofl-relay.service`. That is how it was checked; without the binary present `verify` fails with "not executable".

## 6. Print and publish the multiaddr

Get the peer id:

```sh
journalctl -u ofl-relay -b | grep local_peer_id
```

The relay does not know its public address and does not print it. The `Listening on` lines show addresses of the machine's own interfaces, which on providers that NAT a public address to the VM are private. Take the public address from the provider console. The two multiaddrs to publish, with documentation addresses:

```
/ip4/192.0.2.10/tcp/4001/p2p/<peer id>
/ip4/192.0.2.10/udp/4001/quic-v1/p2p/<peer id>
```

For IPv6, use `/ip6/2001:db8::10/...` the same way, and only if you started the relay with `--use-ipv6 true`.

Publish both. Nothing in this repo reads a bootstrap list yet (`ofl-node` uses libp2p only for its key type), so there is no config key to edit. Until the peer binary defines one, give the addresses to the Rung 1 participants directly; RFC 0001 says they are known contributors. When the peer binary adds its bootstrap list, publish them there and update this section. Whether the address of a production relay belongs in the public repo is a decision for that change; this document must keep using documentation addresses.

## How to tell it is healthy

From the relay host:

1. `systemctl is-active ofl-relay` prints `active`, and `systemctl show -p NRestarts ofl-relay` stays `0`.
2. `journalctl -u ofl-relay -b` shows `local_peer_id=` equal to the id you published, then `Listening on` for both `tcp/<port>` and `udp/<port>/quic-v1`.
3. `ss -ltnu | grep :4001` shows a TCP `LISTEN` and a UDP `UNCONN` socket.

From another machine, which is the check that matters, because it crosses the firewall. Use a checkout of the same rust-libp2p tag.

TCP, with the upstream ping example. It takes the address as its one argument and prints the remote peer id and a ping result:

```sh
cargo run --release -p ping-example -- /ip4/192.0.2.10/tcp/4001
# Event { peer: PeerId("12D3Koo..."), connection: ConnectionId(1), result: Ok(...) }
```

The printed `peer` must equal the published peer id. The relay's journal logs an `Error ... NegotiationFailed` line for this client: the ping example does not speak identify. That line is expected.

QUIC. The upstream ping example is TCP only. Copy `examples/ping` to a new package, add `"quic"` to the libp2p features in its `Cargo.toml`, add `.with_quic()` before `.with_behaviour(...)` in `main.rs`, listen on `/ip4/0.0.0.0/udp/0/quic-v1` instead of the TCP address, and dial `/ip4/192.0.2.10/udp/4001/quic-v1`. That variant was built and gave `result: Ok(...)` against the relay on loopback. Run the QUIC check separately: it is the one a UDP firewall rule breaks silently.

Relay v2 itself: run the upstream dcutr example as a listener with a seed different from the relay's. It dials the relay, asks for a reservation, and logs the result:

```sh
RUST_LOG=info cargo run --release -p dcutr-example -- --mode listen --secret-key-seed 31 \
  --relay-address /ip4/192.0.2.10/tcp/4001/p2p/<peer id>
# ... Relay accepted our reservation request
# ... Listening on address address=<relay multiaddr>/p2p-circuit/p2p/<listener id>
```

The relay's journal then shows `ReservationReqAccepted`. A second `dcutr-example --mode dial --remote-peer-id <listener id>` against the same relay address made the relay log `CircuitReqAccepted`. The dcutr example also accepts a `/udp/<port>/quic-v1/...` relay address, but the one attempt made that way panicked on an unrelated event during its start-up (`examples/dcutr/src/main.rs`, the first `panic!` loop), so QUIC was checked with the ping variant instead.

Unhealthy signs: TCP ping works and QUIC hangs (UDP blocked at the provider or host); the printed peer id differs from the published one (seed changed; republish); `NRestarts` growing (read the journal).

## Upgrades

The example is at version 0.1.0 with no release notes of its own, and its flags can change between tags.

1. Pick the new tag, then record the tag and its commit (`git rev-parse HEAD`) in this runbook and in whatever pin file the peer binary adopts. Change both together.
2. Check out the tag, rebuild with the same command as in [Build](#2-build) (`cargo build --release --locked -p relay-server-example`; checking out a tag does not rebuild `target/release/`), then run `--help` and compare it with the block above. If you build on another host, copy the new `relay-server-example` to the VPS and use that path in place of `target/release/relay-server-example` in step 3. If the flags changed, only note the new flags here; step 3 applies them after the old files are saved.
3. Save what is running before changing anything. A rollback needs the old binary, the old unit and the old settings together, so save all three: `sudo cp -p /usr/local/bin/relay-server-example /usr/local/bin/relay-server-example.prev`, `sudo cp -p /etc/systemd/system/ofl-relay.service /etc/ofl-relay/ofl-relay.service.prev`, `sudo cp -p /etc/ofl-relay/relay.env /etc/ofl-relay/relay.env.prev`. Then replace the binary atomically: `sudo install -m 0755 target/release/relay-server-example /usr/local/bin/relay-server-example.new`, `sudo mv /usr/local/bin/relay-server-example.new /usr/local/bin/relay-server-example`. Only now, if step 2 found changed flags, edit the unit file and the `RELAY_*` values to match.
4. `sudo systemctl daemon-reload` (needed whenever the unit file changed; harmless otherwise), `sudo systemctl restart ofl-relay`, then run the health checks, including the peer id comparison and a check that the running binary is the new one (`systemctl show -p ExecMainStartTimestamp ofl-relay` is after the replace).

The peer id survives the restart as long as `RELAY_SEED` is unchanged (verified: a scripted run restarted the relay and got the same id). Connected peers lose their connections; clients are expected to re-dial and renew reservations, but the example's clients were not tested across a relay restart. An upgrade is not zero downtime.

Roll back by restoring the three saved files, then reloading and restarting: `sudo mv /usr/local/bin/relay-server-example.prev /usr/local/bin/relay-server-example`, `sudo cp -p /etc/ofl-relay/ofl-relay.service.prev /etc/systemd/system/ofl-relay.service`, `sudo cp -p /etc/ofl-relay/relay.env.prev /etc/ofl-relay/relay.env`, `sudo systemctl daemon-reload`, `sudo systemctl restart ofl-relay`, then run the health checks again. Restoring only the binary leaves the old binary paired with the new flags and values; a changed `RELAY_SEED` also changes the peer id and breaks the published multiaddr.

## Not covered

- Running two relays, and how clients choose between them.
- Authentication, an allow-list, abuse handling beyond the default rate limits, and per-IP caps at the firewall.
- Monitoring and alerting. The only signals the example emits are the stdout lines above.
- Replacing the example with the peer binary. When it lands, this runbook should be rewritten around it, and the identity section removed if it persists a random key.
