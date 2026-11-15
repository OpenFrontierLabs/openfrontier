# ofl-node

The device identity of an Open Frontier Node: an Ed25519 keypair that is generated on first run,
kept on disk, and used to sign a small hardware report. This is the first piece of the node
([docs/ARCHITECTURE.md §2](../docs/ARCHITECTURE.md#2-open-frontier-node)); it does not run workers.
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

`keygen` is idempotent: if the key file exists it is left untouched and its peer id is printed.
`sign` creates the key too if this is the first run.

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
