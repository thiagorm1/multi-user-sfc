import ast
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from muar_sfc.experiment_config import ExperimentConfig, command_plan, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_file_generates_only_qwen_evaluation_with_33_seeds_and_300_requests():
    config = load_config(ROOT / "config/mappo_qwen.toml")
    plan = command_plan(config, ROOT, "/python")
    assert len(plan) == 2
    evaluation = plan[1]
    assert evaluation[evaluation.index("--requests") + 1] == "300"
    seeds = evaluation[evaluation.index("--seeds") + 1 : evaluation.index("--variants")]
    assert seeds == [str(n) for n in range(42, 75)]
    assert evaluation[evaluation.index("--variants") + 1] == "mappo_qwen"
    assert "mappo" not in evaluation


def test_hyperparameters_are_forwarded_and_stage_controls_execution():
    config = ExperimentConfig.model_validate(
        {
            "execution": {"stage": "train"},
            "training": {
                "lr_actor": 0.0003,
                "gamma": 0.97,
                "update_every": 10,
                "cloning_batch_size": 64,
            },
        }
    )
    plan = command_plan(config, ROOT, "/python")
    assert len(plan) == 1
    for flag, value in [
        ("--lr-actor", "0.0003"),
        ("--gamma", "0.97"),
        ("--update-every", "10"),
        ("--cloning-batch-size", "64"),
    ]:
        assert plan[0][plan[0].index(flag) + 1] == value
    config.execution.stage = "evaluate"
    assert len(command_plan(config, ROOT, "/python")) == 1
    assert "benchmark_mappo_qwen_muvr.py" in command_plan(config, ROOT, "/python")[0][1]


@pytest.mark.parametrize(
    "data",
    [
        {"training": {"lr_actor": -1}},
        {"training": {"gamma": 2}},
        {"training": {"cloning_batch_size": 0}},
        {"training": {"unknown_parameter": 4}},
        {"evaluation": {"seed_count": 0}},
        {"qwen": {"model": "mock"}},
        {"evaluation": {"seed_start": 900000}},
    ],
)
def test_invalid_or_leaking_configuration_is_rejected(data):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "experiment_runner", ROOT / "scripts/run_mappo_qwen.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_does_not_launch_process_or_write_files(tmp_path, monkeypatch):
    runner = load_runner()
    config = tmp_path / "settings.toml"
    config.write_text('[execution]\noutput_dir="results/new"\n')
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.sys, "argv", ["run", "--config", str(config), "--check"])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("Launched process"))
    monkeypatch.setattr(runner, "check_qwen_available", lambda *a: pytest.fail("Contacted Qwen"))
    runner.main()
    assert not (tmp_path / "results").exists()


def test_existing_model_is_protected_before_any_process_is_launched(tmp_path, monkeypatch):
    runner = load_runner()
    model = tmp_path / "model.pt"
    model.write_bytes(b"existing")
    config = tmp_path / "settings.toml"
    config.write_text('[execution]\nstage="train"\nmodel_path="model.pt"\n')
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.sys, "argv", ["run", "--config", str(config)])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("Launched process"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert model.read_bytes() == b"existing"


def test_all_generated_flags_are_supported_by_the_real_script_parsers():
    plan = command_plan(ExperimentConfig(), ROOT, "/python")
    for command in plan:
        tree = ast.parse(Path(command[1]).read_text())
        supported = set()
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
            ):
                supported.update(arg.value for arg in call.args if isinstance(arg, ast.Constant))
        assert {arg for arg in command[2:] if arg.startswith("--")} <= supported
