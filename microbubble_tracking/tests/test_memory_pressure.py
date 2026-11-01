"""内存报告不能混淆计数器重置、缺失数据和零值；超时须收尾子进程。"""

import subprocess
import sys

import pytest

from customer_workspace.microbubble_tracking.memory_pressure_check import (
    linux_sample,
    run_replay,
)
from customer_workspace.microbubble_tracking.memory_pressure_report import (
    counter_delta,
    distribution,
    pagefile_reads,
)


def test_empty_distribution_does_not_claim_zero_latency():
    result = distribution([])
    assert result["count"] == 0
    assert result["max"] is None


def test_distribution_retains_outlier():
    result = distribution([1, 1, 2, 110])
    assert result["count"] == 4
    assert result["median"] == 1.5
    assert result["max"] == 110


def test_counter_delta_rejects_reset_and_missing_interval():
    assert counter_delta([{"pages": 3}, {"pages": 9}], "pages") == 6
    assert counter_delta([{"pages": 3}], "pages") is None
    with pytest.raises(ValueError, match="Counter reset"):
        counter_delta([{"pages": 9}, {"pages": 3}], "pages")


def test_pagefile_totals_exclude_other_processes_and_mapped_files(tmp_path):
    path = tmp_path / "faults.txt"
    path.write_text(
        "NumFaults, TotalBytes, Process Name (PID), FileName\n"
        '2, 8192, vmmemWSL (123), "C:\\pagefile.sys"\n'
        '5, 20480, chrome.exe (456), "C:\\pagefile.sys"\n'
        '3, 12288, vmmemWSL (123), "C:\\example.dll"\n'
    )
    assert pagefile_reads(path) == {"faults": 2, "bytes": 8192}


def test_invalid_pagefile_report_is_not_zero_paging(tmp_path):
    path = tmp_path / "failed.txt"
    path.write_text("Trace export failed\n")
    with pytest.raises(ValueError, match="Expected an xperf"):
        pagefile_reads(path)


def test_linux_snapshot_has_units_and_no_process_command_lines():
    row = linux_sample()
    assert row["total_bytes"] >= row["available_bytes"] > 0
    assert "pswpin" in row["vmstat"]
    assert all("cmdline" not in process for process in row["processes"])


def test_replay_exit_code_is_not_hidden(tmp_path):
    with (tmp_path / "run.log").open("w") as log:
        assert run_replay([sys.executable, "-c", "raise SystemExit(7)"], log, 5) == 7


def test_timeout_interrupts_replay_and_allows_cleanup(tmp_path):
    marker = tmp_path / "cleaned"
    code = (
        "import time\n"
        "from pathlib import Path\n"
        "try:\n"
        "    time.sleep(30)\n"
        "finally:\n"
        f"    Path({str(marker)!r}).write_text('closed')\n"
    )
    with (
        (tmp_path / "run.log").open("w") as log,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        run_replay([sys.executable, "-c", code], log, 1)
    assert marker.read_text() == "closed"
