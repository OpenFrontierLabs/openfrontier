"""`ofl doctor`: machine, toolchain and network checks. Works without a GPU.

Status meanings:
  ok    as expected
  info  informational, never affects the exit code
  warn  something a run would want fixed; exit code stays 0
  FAIL  hard failure, exit code 1: Python older than 3.11, torch not importable, less than 2 GiB
        free next to the repo, or github.com / huggingface.co unreachable.

Nothing here measures bandwidth. The network rows report DNS time and HTTPS round-trip time only.
"""

from __future__ import annotations

import os
import shutil
import socket
import ssl
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http.client import HTTPSConnection
from importlib import metadata
from pathlib import Path

HOSTS = ("huggingface.co", "github.com")
MIN_PYTHON = (3, 11)
FAIL_FREE_GIB = 2.0
WARN_FREE_GIB = 20.0  # an ofl heuristic for "room for data and checkpoints", not a plan requirement
CLOCK_WARN_SECONDS = 30.0
HTTPS_SAMPLES = 3


@dataclass(frozen=True)
class Row:
    status: str  # ok | info | warn | FAIL
    name: str
    detail: str


@dataclass(frozen=True)
class HttpsProbe:
    ok: bool
    seconds: float
    server_date: datetime | None
    error: str = ""


@dataclass(frozen=True)
class TorchInfo:
    version: str
    cuda_build: str | None  # torch.version.cuda
    cuda_available: bool
    devices: tuple[str, ...]


@dataclass
class Host:
    """Everything doctor reads from the machine, so tests can substitute fakes."""

    python: tuple[int, int, int] = field(default_factory=lambda: sys.version_info[:3])
    run: Callable[[list[str]], tuple[int, str]] = field(default=lambda argv: _run(argv))
    disk_free_bytes: Callable[[Path], int] = field(default=lambda p: shutil.disk_usage(p).free)
    dns: Callable[[str], float | None] = field(default=lambda h: _dns(h))
    https: Callable[[str], HttpsProbe] = field(default=lambda h: _https(h))
    torch: Callable[[], TorchInfo | None] = field(default=lambda: _torch())
    version: Callable[[str], str | None] = field(default=lambda n: _version(n))
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


def _run(argv: list[str]) -> tuple[int, str]:
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, ""
    return res.returncode, res.stdout.strip()


def _dns(host: str) -> float | None:
    start = time.perf_counter()
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        return None
    return time.perf_counter() - start


def _tls_context() -> ssl.SSLContext:
    """Default context, using certifi's bundle when the Python build has no system CA path."""
    if ssl.get_default_verify_paths().cafile is None and not os.environ.get("SSL_CERT_FILE"):
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _https(host: str) -> HttpsProbe:
    start = time.perf_counter()
    try:
        conn = HTTPSConnection(host, timeout=8, context=_tls_context())
        conn.request("HEAD", "/")
        resp = conn.getresponse()
        header = resp.getheader("Date")
        conn.close()
    except (OSError, ValueError) as exc:
        return HttpsProbe(False, time.perf_counter() - start, None, str(exc))
    elapsed = time.perf_counter() - start
    date = None
    if header:
        try:
            date = parsedate_to_datetime(header)
        except (TypeError, ValueError):
            date = None
    return HttpsProbe(True, elapsed, date)


def _torch() -> TorchInfo | None:
    try:
        import torch
    except ImportError:
        return None
    available = bool(torch.cuda.is_available())
    names = tuple(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    return TorchInfo(torch.__version__, torch.version.cuda, available, names)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _gpu_rows(host: Host, torch_info: TorchInfo | None) -> list[Row]:
    rc, out = host.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    if rc != 0 or not out:
        return [
            Row("warn", "gpu", "no GPU: CPU smoke only (nvidia-smi not found or reported none)"),
            Row("info", "cuda", "n/a without a GPU"),
        ]
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    rows = [Row("ok", "gpu", f"{len(lines)} x {lines[0].split(',')[0].strip()}")]
    rows.extend(Row("info", f"gpu {i}", ln) for i, ln in enumerate(lines))
    driver = lines[0].split(",")[-1].strip()
    if torch_info is None:
        rows.append(Row("info", "cuda", f"driver {driver}; torch not importable"))
    elif torch_info.cuda_build is None:
        rows.append(
            Row(
                "info",
                "cuda",
                f"driver {driver}; this environment's torch {torch_info.version} is a CPU build "
                "(expected: GPU runs use their own environment, see docs/run0.md)",
            )
        )
    else:
        state = "usable" if torch_info.cuda_available else "NOT usable from torch"
        status = "ok" if torch_info.cuda_available else "warn"
        rows.append(
            Row(
                status,
                "cuda",
                f"driver {driver}; torch {torch_info.version} built for CUDA "
                f"{torch_info.cuda_build}, {state}",
            )
        )
    return rows


def _network_rows(host: Host) -> list[Row]:
    rows: list[Row] = []
    skews: list[float] = []
    skew_hosts: list[str] = []
    for name in HOSTS:
        dns = host.dns(name)
        if dns is None:
            rows.append(Row("FAIL", f"net {name}", "DNS lookup failed"))
            continue
        probes = [host.https(name) for _ in range(HTTPS_SAMPLES)]
        good = [p for p in probes if p.ok]
        if not good:
            rows.append(Row("FAIL", f"net {name}", f"HTTPS failed: {probes[-1].error}"))
            continue
        median_ms = statistics.median(p.seconds for p in good) * 1000
        rows.append(
            Row(
                "ok",
                f"net {name}",
                f"DNS {dns * 1000:.0f} ms; HTTPS round trip incl. connect, median of "
                f"{len(good)}: {median_ms:.0f} ms (latency only, bandwidth not measured)",
            )
        )
        dates = [p.server_date for p in good if p.server_date is not None]
        if dates:
            skews.append((host.now() - dates[0]).total_seconds())
            skew_hosts.append(name)
    if skews:
        # A cached CDN response carries an old Date, which inflates one host's skew. A wrong local
        # clock shows on every host, so judge by the host that agrees best.
        best = min(skews, key=abs)
        status = "warn" if abs(best) > CLOCK_WARN_SECONDS else "ok"
        per_host = ", ".join(f"{h} {s:+.1f} s" for h, s in zip(skew_hosts, skews, strict=True))
        rows.append(Row(status, "clock", f"{best:+.1f} s (smallest of: {per_host})"))
    else:
        rows.append(Row("info", "clock", "not checked: no server Date header received"))
    return rows


def collect(host: Host, repo: Path) -> list[Row]:
    rows: list[Row] = []
    py = ".".join(map(str, host.python))
    rows.append(
        Row(
            "ok" if host.python >= MIN_PYTHON else "FAIL",
            "python",
            f"{py} (need >= {'.'.join(map(str, MIN_PYTHON))})",
        )
    )
    rc, out = host.run(["uv", "--version"])
    rows.append(Row("ok", "uv", out) if rc == 0 else Row("warn", "uv", "not found on PATH"))

    torch_info = host.torch()
    if torch_info is None:
        rows.append(Row("FAIL", "torch", "not importable"))
    else:
        rows.append(Row("ok", "torch", torch_info.version))
    for pkg in ("ai2-olmo-core", "jsonschema"):
        ver = host.version(pkg)
        rows.append(Row("ok", pkg, ver) if ver else Row("warn", pkg, "not installed"))

    rows.extend(_gpu_rows(host, torch_info))

    free = host.disk_free_bytes(repo) / 2**30
    if free < FAIL_FREE_GIB:
        status = "FAIL"
    elif free < WARN_FREE_GIB:
        status = "warn"
    else:
        status = "ok"
    rows.append(Row(status, "disk", f"{free:.1f} GiB free at {repo}"))

    rows.extend(_network_rows(host))
    rows.append(
        Row(
            "info",
            "run fit",
            "not computed: no run is open to join, so there are no thresholds to compare against",
        )
    )
    return rows


def render(rows: list[Row]) -> str:
    width = max(len(r.name) for r in rows)
    lines = [f"{'STATUS':<6}  {'CHECK':<{width}}  DETAIL"]
    lines.extend(f"{r.status:<6}  {r.name:<{width}}  {r.detail}" for r in rows)
    return "\n".join(lines)


def exit_code(rows: list[Row]) -> int:
    return 1 if any(r.status == "FAIL" for r in rows) else 0


def doctor(repo: Path, host: Host | None = None) -> int:
    rows = collect(host or Host(), repo)
    print(render(rows))
    failed = [r.name for r in rows if r.status == "FAIL"]
    warned = [r.name for r in rows if r.status == "warn"]
    if failed:
        print(f"\ndoctor: {len(failed)} hard failure(s): {', '.join(failed)}")
    else:
        print(f"\ndoctor: no hard failures; {len(warned)} warning(s)")
    return exit_code(rows)
