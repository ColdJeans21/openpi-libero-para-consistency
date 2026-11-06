import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def module():
    script_path = Path(__file__).parents[2] / "examples/libero_plus/eval_language_openpi.py"
    spec = importlib.util.spec_from_file_location("eval_language_openpi", script_path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


def test_parse_official_task_name_accepts_neutral_language_entry(module):
    assert module.parse_official_task_name("put_the_bowl_on_the_plate_language_17_view_0_0_100_0_0_initstate_0") == (
        "put_the_bowl_on_the_plate_language_17",
        "put_the_bowl_on_the_plate",
        17,
    )


def test_parse_official_task_name_rejects_non_neutral_physical_setting(module):
    with pytest.raises(ValueError, match="neutral physical setting"):
        module.parse_official_task_name("put_the_bowl_on_the_plate_language_17_view_30_0_100_0_0_initstate_0")


def test_load_language_tasks_filters_category_and_difficulty(tmp_path, module):
    benchmark_dir = tmp_path / "libero/libero/benchmark"
    bddl_dir = tmp_path / "libero/libero/bddl_files/libero_goal"
    benchmark_dir.mkdir(parents=True)
    bddl_dir.mkdir(parents=True)
    classification = {
        "libero_goal": [
            {
                "id": 1,
                "name": "turn_on_the_stove_language_2_view_0_0_100_0_0_initstate_0",
                "category": "Language Instructions",
                "difficulty_level": 4,
            },
            {
                "id": 2,
                "name": "turn_on_the_stove_view_20_0_100_0_0_initstate_0",
                "category": "Camera Viewpoints",
                "difficulty_level": 1,
            },
        ]
    }
    (benchmark_dir / "task_classification.json").write_text(json.dumps(classification), encoding="utf-8")
    (bddl_dir / "turn_on_the_stove_language_2.bddl").write_text(
        "(define (problem test)\n  (:language Activate the cooktop)\n)\n", encoding="utf-8"
    )

    tasks, _ = module.load_language_tasks(tmp_path, "libero_goal", difficulty_levels={4})

    assert len(tasks) == 1
    assert tasks[0]["base_task"] == "turn_on_the_stove"
    assert tasks[0]["instruction"] == "Activate the cooktop"


def test_write_summary_groups_difficulty_and_base_task(tmp_path, module):
    classification = tmp_path / "classification.json"
    classification.write_text("{}", encoding="utf-8")
    records = [
        {"success": True, "difficulty_level": 1, "base_task": "a"},
        {"success": False, "difficulty_level": 1, "base_task": "a"},
        {"success": True, "difficulty_level": 5, "base_task": "b"},
    ]

    summary = module.write_summary(
        tmp_path,
        records,
        model_name="model",
        suite="libero_goal",
        seed=7,
        expected_tasks=3,
        classification_path=classification,
    )

    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["per_difficulty"]["1"]["success_rate"] == 0.5
    assert summary["per_difficulty"]["5"]["success_rate"] == 1.0
    assert summary["complete"] is True
