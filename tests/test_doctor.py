from datetime import UTC, datetime, timedelta
from pathlib import Path

from ofl.doctor import Host, HttpsProbe, Row, TorchInfo, collect, exit_code

NOW = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)
GIB = 2**30


def fake_host(**overrides: object) -> Host:
    def run(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "uv":
            return 0, "uv 0.0.0"
        return 127, ""  # nvidia-smi absent

    host = Host(
        python=(3, 12, 0),
        run=run,
        disk_free_bytes=lambda _p: 500 * GIB,
        dns=lambda _h: 0.004,
        https=lambda _h: HttpsProbe(True, 0.12, NOW),
        torch=lambda: TorchInfo("2.13.0+cpu", None, False, ()),
        version=lambda _n: "1.0",
        now=lambda: NOW,
    )
    for key, value in overrides.items():
        setattr(host, key, value)
    return host


def by_name(rows: list[Row]) -> dict[str, Row]:
    return {r.name: r for r in rows}


def test_no_gpu_machine_warns_and_exits_zero() -> None:
    rows = collect(fake_host(), Path("."))
    table = by_name(rows)
    assert table["gpu"].status == "warn"
    assert "CPU smoke only" in table["gpu"].detail
    assert exit_code(rows) == 0


def test_gpu_with_cpu_only_torch_is_informational() -> None:
    def run(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "nvidia-smi":
            return 0, "NVIDIA H100 80GB HBM3, 81559 MiB, 560.35.03"
        return 0, "uv 0.0.0"

    rows = collect(fake_host(run=run), Path("."))
    table = by_name(rows)
    assert table["gpu"].status == "ok"
    assert table["gpu"].detail.startswith("1 x NVIDIA H100")
    assert table["cuda"].status == "info"
    assert "560.35.03" in table["cuda"].detail
    assert "CPU build" in table["cuda"].detail


def test_gpu_present_but_torch_cannot_use_it_warns() -> None:
    def run(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "nvidia-smi":
            return 0, "NVIDIA H100 80GB HBM3, 81559 MiB, 560.35.03"
        return 0, "uv 0.0.0"

    gpu_torch = TorchInfo("2.13.0+cu130", "13.0", False, ())
    rows = collect(fake_host(run=run, torch=lambda: gpu_torch), Path("."))
    cuda = by_name(rows)["cuda"]
    assert cuda.status == "warn"
    assert "13.0" in cuda.detail
    assert exit_code(rows) == 0


def test_old_python_is_a_hard_failure() -> None:
    rows = collect(fake_host(python=(3, 10, 9)), Path("."))
    assert by_name(rows)["python"].status == "FAIL"
    assert exit_code(rows) == 1


def test_missing_torch_is_a_hard_failure() -> None:
    rows = collect(fake_host(torch=lambda: None), Path("."))
    assert by_name(rows)["torch"].status == "FAIL"
    assert exit_code(rows) == 1


def test_disk_thresholds() -> None:
    assert (
        by_name(collect(fake_host(disk_free_bytes=lambda _p: GIB), Path(".")))["disk"].status
        == "FAIL"
    )
    assert (
        by_name(collect(fake_host(disk_free_bytes=lambda _p: 10 * GIB), Path(".")))["disk"].status
        == "warn"
    )


def test_unreachable_network_is_a_hard_failure_per_host() -> None:
    down = fake_host(dns=lambda _h: None)
    rows = collect(down, Path("."))
    table = by_name(rows)
    assert table["net github.com"].status == "FAIL"
    assert table["net huggingface.co"].status == "FAIL"
    assert exit_code(rows) == 1


def test_https_failure_reports_the_error() -> None:
    bad = fake_host(https=lambda _h: HttpsProbe(False, 8.0, None, "timed out"))
    row = by_name(collect(bad, Path(".")))["net github.com"]
    assert row.status == "FAIL"
    assert "timed out" in row.detail


def test_latency_is_the_median_of_samples_and_bandwidth_is_not_claimed() -> None:
    samples = iter([0.5, 0.1, 0.2] * 2)
    host = fake_host(https=lambda _h: HttpsProbe(True, next(samples), NOW))
    row = by_name(collect(host, Path(".")))["net huggingface.co"]
    assert "200 ms" in row.detail
    assert "bandwidth not measured" in row.detail


def test_clock_skew_warns() -> None:
    skewed = fake_host(https=lambda _h: HttpsProbe(True, 0.1, NOW - timedelta(minutes=5)))
    clock = by_name(collect(skewed, Path(".")))["clock"]
    assert clock.status == "warn"
    assert "+300.0 s" in clock.detail
    assert by_name(collect(fake_host(), Path(".")))["clock"].status == "ok"
