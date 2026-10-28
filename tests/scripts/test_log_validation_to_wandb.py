import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def validation_logging_module():
    script_path = Path(__file__).parents[2] / "scripts" / "log_validation_to_wandb.py"
    spec = importlib.util.spec_from_file_location("log_validation_to_wandb", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_goal_metrics(tmp_path, validation_logging_module):
    summary = tmp_path / "goal.json"
    summary.write_text(json.dumps({"success_rate": 0.84, "total_episodes": 100}), encoding="utf-8")

    assert validation_logging_module.load_goal_metrics(summary) == {
        "validation/libero_goal_success_rate": 0.84,
        "validation/libero_goal_episodes": 100.0,
    }


def test_load_official_libero_para_metrics(tmp_path, validation_logging_module):
    summary = tmp_path / "para.json"
    summary.write_text(
        json.dumps(
            {
                "overall_success_rate": 0.63,
                "total_episodes": 4092,
                "per_category": {
                    "act_lexical_synonym": {"total": 100, "successes": 70, "success_rate": 0.7},
                },
            }
        ),
        encoding="utf-8",
    )

    assert validation_logging_module.load_para_metrics(summary) == {
        "validation/libero_para_success_rate": 0.63,
        "validation/libero_para_episodes": 4092.0,
        "validation/libero_para_category/act_lexical_synonym": 0.7,
    }
