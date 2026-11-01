"""避免温度预检把缺测、失败或高温样本误判为可开始。"""

from customer_workspace.tools.gcu_thermal_report import assess


def good_samples():
    return [
        {
            "cpu_c": 60,
            "gpu_c": 50,
            "cpu_fan_rpm": 3000,
            "gpu_fan_rpm": 3000,
            "read_ms": 550,
            "ok": True,
        }
        for _ in range(10)
    ]


def test_hot_sample_blocks_start_even_if_other_samples_are_cool():
    samples = good_samples()
    samples[5]["cpu_c"] = 95
    assert not assess(samples)["recorded_start_conditions_pass"]


def test_missing_gpu_temperature_cannot_pass():
    samples = good_samples()
    del samples[3]["gpu_c"]
    assert not assess(samples)["recorded_start_conditions_pass"]


def test_stopped_fan_and_failed_read_cannot_pass():
    samples = good_samples()
    samples[0]["gpu_fan_rpm"] = 0
    samples[1]["ok"] = False
    assert not assess(samples)["recorded_start_conditions_pass"]


def test_old_records_never_authorize_a_pause():
    result = assess(good_samples())
    assert result["recorded_start_conditions_pass"]
    assert result["live_recheck_required"]
    assert "No service-pause authorization" in result["authorization"]


def test_empty_samples_cannot_pass():
    assert not assess([])["recorded_start_conditions_pass"]


def test_nonfinite_temperature_cannot_pass():
    samples = good_samples()
    samples[2]["cpu_c"] = float("nan")
    assert not assess(samples)["recorded_start_conditions_pass"]
