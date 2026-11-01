"""只检查模拟相机的验收逻辑，不启动 Windows 服务或访问硬件。"""

import pytest

from customer_workspace.tools.windows_camera_sim import consecutive_cycle_start


def test_continuous_loops_can_start_mid_cycle():
    assert consecutive_cycle_start(list("cabcabca"), list("abc")) == 2


@pytest.mark.parametrize("observed", ["", "abacabc", "abbcabc", "abxcabc"])
def test_gap_duplicate_and_corruption_fail_even_if_later_cycle_is_complete(observed):
    assert consecutive_cycle_start(list(observed), list("abc")) is None


def test_repeated_source_frame_does_not_choose_wrong_start():
    assert consecutive_cycle_start(list("acaba"), list("abac")) == 2
