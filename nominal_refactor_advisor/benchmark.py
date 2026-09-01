"""Reproducible cold/warm focused-scan benchmark runner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any


@dataclass(frozen=True)
class FocusedScanBenchmarkRun:
    """Observed process and payload metrics for one benchmark pass."""

    label: str
    wall_seconds: float
    peak_rss_mb: float | None
    return_code: int
    timed_out: bool
    finding_count: int | None
    scan_mode: str | None
    scan_complete: bool | None
    parse_seconds: float | None
    analysis_seconds: float | None
    analysis_cache_status: str | None
    payload_error: str | None = None


@dataclass(frozen=True)
class FocusedScanBenchmarkReport:
    """Cold/warm measurements produced from one isolated cache directory."""

    targets: tuple[str, ...]
    cache_root: str
    cold: FocusedScanBenchmarkRun
    warm: FocusedScanBenchmarkRun

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LinuxProcessTreeRssSampler:
    """Sample aggregate resident memory for one Linux process tree."""

    @classmethod
    def rss_bytes(cls, root_pid: int) -> int | None:
        if not Path("/proc").is_dir():
            return None
        pending = [root_pid]
        seen: set[int] = set()
        total_bytes = 0
        sampled = False
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            status_path = Path(f"/proc/{pid}/status")
            try:
                status_text = status_path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in status_text.splitlines():
                if line.startswith("VmRSS:"):
                    total_bytes += int(line.split()[1]) * 1024
                    sampled = True
                    break
            children_path = Path(f"/proc/{pid}/task/{pid}/children")
            try:
                pending.extend(
                    int(child_pid) for child_pid in children_path.read_text().split()
                )
            except OSError:
                continue
        return total_bytes if sampled else None


@dataclass(frozen=True, kw_only=True)
class FocusedScanBenchmark:
    """Run NRA twice against the same targets and isolated cache."""

    python_executable: Path
    advisor_root: Path
    targets: tuple[Path, ...]
    cache_root: Path
    timeout_seconds: float = 60.0
    scan_budget_seconds: float = 45.0
    sample_interval_seconds: float = 0.01

    def run(self) -> FocusedScanBenchmarkReport:
        return FocusedScanBenchmarkReport(
            targets=tuple(path.as_posix() for path in self.targets),
            cache_root=self.cache_root.as_posix(),
            cold=self._run_once("cold"),
            warm=self._run_once("warm"),
        )

    def _run_once(self, label: str) -> FocusedScanBenchmarkRun:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_executable.as_posix(),
            "-m",
            "nominal_refactor_advisor",
            "--json",
            "--json-payload",
            "loop",
            "--no-structural-overlap",
            "--cache-dir",
            (self.cache_root / "ast").as_posix(),
            "--scan-budget-seconds",
            str(self.scan_budget_seconds),
            "--parse-workers",
            "1",
            "--analysis-workers",
            "1",
            *(path.as_posix() for path in self.targets),
        ]
        environment = os.environ.copy()
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                self.advisor_root.as_posix(),
                *((existing_python_path,) if existing_python_path else ()),
            )
        )
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file:
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
                started = monotonic()
                process = subprocess.Popen(
                    command,
                    cwd=self.advisor_root,
                    env=environment,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    start_new_session=True,
                )
                peak_rss_bytes: int | None = None
                timed_out = False
                while process.poll() is None:
                    sampled_rss = LinuxProcessTreeRssSampler.rss_bytes(process.pid)
                    if sampled_rss is not None:
                        peak_rss_bytes = max(peak_rss_bytes or 0, sampled_rss)
                    if monotonic() - started >= self.timeout_seconds:
                        timed_out = True
                        self._terminate_process_group(process)
                        break
                    sleep(self.sample_interval_seconds)
                return_code = process.wait()
                wall_seconds = round(monotonic() - started, 3)
                stdout_file.seek(0)
                stdout_text = stdout_file.read()
                stderr_file.seek(0)
                stderr_text = stderr_file.read()
        payload, payload_error = self._payload(stdout_text, stderr_text)
        timing = payload.get("timing") if isinstance(payload, dict) else None
        scan_status = payload.get("scan_status") if isinstance(payload, dict) else None
        return FocusedScanBenchmarkRun(
            label=label,
            wall_seconds=wall_seconds,
            peak_rss_mb=(
                None if peak_rss_bytes is None else round(peak_rss_bytes / 1024**2, 1)
            ),
            return_code=return_code,
            timed_out=timed_out,
            finding_count=self._optional_int(payload, "finding_count"),
            scan_mode=self._optional_str(scan_status, "mode"),
            scan_complete=self._optional_bool(scan_status, "complete"),
            parse_seconds=self._optional_float(timing, "parse_seconds"),
            analysis_seconds=self._optional_float(timing, "analysis_seconds"),
            analysis_cache_status=self._optional_str(
                timing,
                "analysis_cache_status",
            ),
            payload_error=payload_error,
        )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    @staticmethod
    def _payload(
        stdout_text: str,
        stderr_text: str,
    ) -> tuple[dict[str, Any], str | None]:
        try:
            payload = json.loads(stdout_text)
        except json.JSONDecodeError as error:
            diagnostic = stderr_text.strip() or stdout_text.strip() or str(error)
            return {}, diagnostic[-2000:]
        if not isinstance(payload, dict):
            return {}, "benchmark subprocess emitted a non-object JSON payload"
        return payload, None

    @staticmethod
    def _optional_int(payload: object, key: str) -> int | None:
        if not isinstance(payload, dict):
            return None
        value = payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _optional_str(payload: object, key: str) -> str | None:
        if not isinstance(payload, dict):
            return None
        value = payload.get(key)
        return value if isinstance(value, str) else None

    @staticmethod
    def _optional_bool(payload: object, key: str) -> bool | None:
        if not isinstance(payload, dict):
            return None
        value = payload.get(key)
        return value if isinstance(value, bool) else None

    @staticmethod
    def _optional_float(payload: object, key: str) -> float | None:
        if not isinstance(payload, dict):
            return None
        value = payload.get(key)
        return float(value) if isinstance(value, (int, float)) else None


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure cold and warm focused NRA loop scans.",
    )
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--advisor-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--work-dir", type=Path, default=Path("build/benchmarks"))
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--scan-budget-seconds", type=float, default=45.0)
    parser.add_argument("--max-cold-seconds", type=float)
    parser.add_argument("--max-cold-rss-mb", type=float)
    parser.add_argument("--max-warm-seconds", type=float)
    parser.add_argument("--max-warm-rss-mb", type=float)
    return parser


def _exceeds(value: float | None, maximum: float | None) -> bool:
    if maximum is None:
        return False
    return value is None or value > maximum


def _report_failed(
    report: FocusedScanBenchmarkReport,
    args: argparse.Namespace,
) -> bool:
    runs = (report.cold, report.warm)
    if any(
        run.return_code != 0
        or run.timed_out
        or run.payload_error is not None
        or run.scan_mode != "focused_local_partial"
        or run.scan_complete is not False
        or run.finding_count is None
        for run in runs
    ):
        return True
    if report.cold.finding_count != report.warm.finding_count:
        return True
    return any(
        (
            _exceeds(report.cold.wall_seconds, args.max_cold_seconds),
            _exceeds(report.cold.peak_rss_mb, args.max_cold_rss_mb),
            _exceeds(report.warm.wall_seconds, args.max_warm_seconds),
            _exceeds(report.warm.peak_rss_mb, args.max_warm_rss_mb),
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI and return a budget-aware process status."""

    args = _argument_parser().parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="focused-scan-",
        dir=args.work_dir,
    ) as cache_root_text:
        report = FocusedScanBenchmark(
            python_executable=args.python,
            advisor_root=args.advisor_root,
            targets=tuple(args.targets),
            cache_root=Path(cache_root_text),
            timeout_seconds=args.timeout_seconds,
            scan_budget_seconds=args.scan_budget_seconds,
        ).run()
        print(json.dumps(report.to_dict(), indent=2))
        return 1 if _report_failed(report, args) else 0


def process_main() -> None:
    """Console-script entrypoint."""

    raise SystemExit(main())


if __name__ == "__main__":
    process_main()
