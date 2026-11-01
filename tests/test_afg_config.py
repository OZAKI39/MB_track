"""核对 AFG 配置入口及幅值边界；在参数解析阶段截停，不连接设备。"""

import argparse
import importlib
from dataclasses import replace

import pytest

from customer_workspace.afg_video_check import AFGControl, load_video_plan
from customer_workspace.experiment_runner import load_plan
from microbubble_experiment.afg import DEFAULT_CONFIG, ChannelSettings, load_config


@pytest.mark.parametrize(
    "module_name",
    [
        "microbubble_experiment.afg",
        "customer_workspace.experiment_runner",
        "customer_workspace.timing_benchmark",
        "customer_workspace.four_pzt_validation",
        "customer_workspace.amplitude_pair_diagnostic",
        "customer_workspace.visa_diagnostics",
    ],
)
def test_cli_defaults_use_one_config(module_name, monkeypatch):
    class ParsingChecked(Exception):
        pass

    def check_config(parser, *args, **kwargs):
        assert parser.get_default("config") == DEFAULT_CONFIG
        raise ParsingChecked

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", check_config)
    with pytest.raises(ParsingChecked):
        importlib.import_module(module_name).main()


def test_default_config_is_independent_of_working_directory(tmp_path, monkeypatch):
    expected = load_config(DEFAULT_CONFIG)
    monkeypatch.chdir(tmp_path)
    assert load_config() == expected
    assert AFGControl().config == DEFAULT_CONFIG


@pytest.mark.parametrize("name", ["PZT1", "PZT2", "PZT3", "PZT4"])
def test_shared_amplitude_boundary(name):
    limit = load_config()["channels"][name]
    ChannelSettings(amplitude_vpp=0.8).validate(limit)
    ChannelSettings(amplitude_vpp=0.001).validate(limit)
    with pytest.raises(ValueError, match="amplitude_vpp"):
        ChannelSettings(amplitude_vpp=0.801).validate(limit)


@pytest.mark.parametrize(
    "filename", ["four_pzt_video_20s.json", "two_action_video_20s.json"]
)
def test_both_experiment_entries_validate_the_same_plan(filename):
    path = DEFAULT_CONFIG.with_name(filename)
    plan = load_plan(path, load_config())
    assert load_video_plan(path) == plan
    assert all(s.amplitude_vpp == 0.8 for s in plan.settings.values())
    assert all(s.frequency_hz == 2_500_000 for s in plan.settings.values())
    invalid = dict(plan.settings, PZT4=ChannelSettings(amplitude_vpp=0.801))
    with pytest.raises(ValueError, match="amplitude_vpp"):
        replace(plan, settings=invalid).validate(load_config())
