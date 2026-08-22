#!/usr/bin/env python3
"""Run every module's built-in self-test, plus a smoke test of each tool.

    python run_selftests.py

Exits 0 if everything passes, 1 otherwise.  Nothing here needs a network
connection or any third-party package.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# Importing the package fixes this process's own console encoding too, so the
# runner can print its own output safely when piped on Windows.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpi  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent
PCAP = ROOT / "test_dpi.pcap"

MODULES = [
    "platform", "types", "pcap_reader", "packet_parser", "sni_extractor",
    "thread_safe_queue", "rule_manager", "connection_tracker",
    "load_balancer", "fast_path",
]


def run(label: str, args: list[str]) -> bool:
    """Run one check quietly; print a pass/fail line."""
    # Decode the child's output as UTF-8 explicitly.  Without this, `text=True`
    # falls back to the locale encoding (cp1252 on Windows), which mangles or
    # rejects the box-drawing characters the reports are framed in.
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    ok = proc.returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        for line in tail:
            print(f"        {line}")
    return ok


def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print(f"stdout encoding: {getattr(sys.stdout, 'encoding', '?')}\n")

    if not PCAP.exists():
        print(f"Missing test capture: {PCAP}")
        print("Generate one with:  python generate_test_pcap.py")
        return 1

    results: list[bool] = []

    print("Module self-tests")
    for name in MODULES:
        results.append(run(f"dpi/{name}.py", ["-m", f"dpi.{name}", str(PCAP)]))

    print("\nCommand-line tools")
    # Write tool output to a temp directory, so nothing is left behind and no
    # cleanup can fail on a read-only or permission-restricted folder.
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "selftest_out.pcap")
        results.append(run("main.py", ["main.py", str(PCAP), "3"]))
        results.append(run("main_simple.py", ["main_simple.py", str(PCAP)]))
        results.append(run("main_working.py", ["main_working.py", str(PCAP), out]))
        results.append(run("main_dpi.py", ["main_dpi.py", str(PCAP), out]))
        results.append(run("dpi_mt.py", ["dpi_mt.py", str(PCAP), out]))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
