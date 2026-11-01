"""确认诊断按同一原画面配对，不混用运行内帧号或把重叠区间重复放大。"""

import pytest

from customer_workspace.tools.queue_delay_report import matched_comparison, overlap_ms


def test_same_original_frame_is_matched_despite_different_receiver_numbers():
    left = [{"seq": 10, "feedback_total_ms": 20}, {"seq": 11, "feedback_total_ms": 30}]
    right = [
        {"seq": 20, "feedback_total_ms": 35},
        {"seq": 21, "feedback_total_ms": 25},
        {"seq": 22, "feedback_total_ms": 90},
    ]
    result, paired = matched_comparison(
        left, right, {10: 100, 11: 101}, {20: 101, 21: 100, 22: 102}
    )
    assert result["common_count"] == 2
    assert result["q2_only"] == [102]
    assert result["q2_common"]["max"] == 35
    assert result["q2_all"]["max"] == 90
    assert [row["difference_feedback_total_ms"] for row in paired] == [5, 5]


def test_waiting_overlap_clips_to_the_actual_interval():
    assert overlap_ms(10, 10.02, 9.9, 10.01) == pytest.approx(10)
    assert overlap_ms(10, 10.02, 10.01, 10.03) == pytest.approx(10)
    assert overlap_ms(10, 10.02, 10.03, 10.04) == 0
