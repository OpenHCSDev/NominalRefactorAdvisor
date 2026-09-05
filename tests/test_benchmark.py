from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import cast

from nominal_refactor_advisor.benchmark import (
    FocusedScanBenchmark,
    LinuxProcessTreeRssSampler,
    main,
)


def test_linux_process_tree_rss_sampler_observes_current_process() -> None:
    rss_bytes = LinuxProcessTreeRssSampler.rss_bytes(os.getpid())

    if Path("/proc").is_dir():
        assert rss_bytes is not None
        assert rss_bytes > 0
    else:
        assert rss_bytes is None


def test_focused_scan_benchmark_records_cold_and_warm_payloads(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    focused_path = package_root / "alpha.py"
    focused_path.write_text("VALUE = 'alpha'\n", encoding="utf-8", newline="")
    (package_root / "beta.py").write_text("VALUE = 'beta'\n", encoding="utf-8", newline="")
    benchmark = FocusedScanBenchmark(
        python_executable=Path(sys.executable),
        advisor_root=repo_root,
        targets=(focused_path,),
        cache_root=tmp_path / "cache",
        timeout_seconds=30.0,
    )

    report = benchmark.run()

    assert report.cold.return_code == 0
    assert report.warm.return_code == 0
    assert report.cold.scan_mode == "focused_local_partial"
    assert report.warm.scan_mode == "focused_local_partial"
    assert report.cold.finding_count == report.warm.finding_count
    assert report.cold.analysis_cache_status == "miss"
    assert report.warm.analysis_cache_status == "hit"
    assert report.cold.peak_rss_mb is None or report.cold.peak_rss_mb > 0


def test_benchmark_cli_emits_json_and_enforces_budget(
    tmp_path: Path,
    capsys,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    focused_path = package_root / "alpha.py"
    focused_path.write_text("VALUE = 'alpha'\n", encoding="utf-8", newline="")
    (package_root / "beta.py").write_text("VALUE = 'beta'\n", encoding="utf-8", newline="")

    exit_code = main(
        [
            "--advisor-root",
            repo_root.as_posix(),
            "--work-dir",
            (tmp_path / "runs").as_posix(),
            "--max-cold-seconds",
            "0",
            focused_path.as_posix(),
        ]
    )
    payload = cast(dict[str, object], json.loads(capsys.readouterr().out))

    assert exit_code == 1
    assert "cold" in payload
    assert "warm" in payload
