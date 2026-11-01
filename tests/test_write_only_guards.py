"""检查真实驱动未连接时的本地保护；不模拟 AFG，也不访问硬件。"""

import pytest

from customer_workspace.afg_timing import RoundUnreliable, TimedRig
from microbubble_experiment.afg import AFGError, load_config


@pytest.mark.parametrize("method", ["send_amplitude", "send_frequency"])
def test_unprepared_driver_rejects_write_without_connecting(method):
    rig = TimedRig(load_config())
    with pytest.raises(AFGError, match="轮前核查"):
        getattr(rig, method)("PZT1", 0.1)
    assert rig.sessions == {} and rig.rm is None
    assert rig.last_amplitude_timing is None
    assert rig.last_frequency_timing is None


@pytest.mark.parametrize("method", ["send_amplitude", "send_frequency"])
def test_stop_invalidates_write_only_permission_without_devices(method):
    rig = TimedRig(load_config())
    assert rig.all_off() == []
    with pytest.raises(AFGError, match="轮前核查"):
        getattr(rig, method)("PZT1", 0.1)
    assert rig.stop_requested.is_set()
    assert rig.sessions == {} and rig.rm is None


def test_unreliable_error_preserves_audit_evidence():
    report = {
        "passed": False,
        "problems": ["A 错误队列：-113,Undefined header"],
        "error_queue_replies": {"A": ['-113,"Undefined header"', '0,"No error"']},
    }
    error = RoundUnreliable(report)
    assert error.report is report
    assert "-113" in str(error)
