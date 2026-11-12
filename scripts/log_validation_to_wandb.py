#!/usr/bin/env python3
"""Attach closed-loop LIBERO validation metrics to an existing training run.

Evaluation is intentionally decoupled from the two-GPU FSDP training process.
The checkpoint step is logged as a custom W&B x-axis, so results can be added
after training without violating W&B's monotonically increasing internal step.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import wandb


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _first_number(data: dict[str, Any], names: tuple[str, ...], *, source: Path) -> float:
    for name in names:
        value = data.get(name)
        if isinstance(value, int | float):
            return float(value)
    raise KeyError(f"None of {names} was found as a number in {source}")


def load_goal_metrics(path: Path) -> dict[str, float]:
    data = _read_json(path)
    rate = _first_number(data, ("success_rate", "overall_success_rate"), source=path)
    metrics = {"validation/libero_goal_success_rate": rate}
    if isinstance(data.get("total_episodes"), int | float):
        metrics["validation/libero_goal_episodes"] = float(data["total_episodes"])
    return metrics


def load_para_metrics(path: Path) -> dict[str, float]:
    data = _read_json(path)
    rate = _first_number(data, ("overall_success_rate", "success_rate"), source=path)
    metrics = {"validation/libero_para_success_rate": rate}
    for key in ("total_episodes", "num_tasks", "total_tasks"):
        if isinstance(data.get(key), int | float):
            metrics["validation/libero_para_episodes"] = float(data[key])
            break

    category_results = data.get("per_category", data.get("category_results", data.get("categories")))
    if isinstance(category_results, dict):
        for category, result in category_results.items():
            if isinstance(result, int | float):
                category_rate = float(result)
            elif isinstance(result, dict):
                try:
                    category_rate = _first_number(result, ("success_rate", "overall_success_rate", "mean"), source=path)
                except KeyError:
                    continue
            else:
                continue
            safe_category = str(category).strip().lower().replace(" ", "_").replace("/", "_")
            metrics[f"validation/libero_para_category/{safe_category}"] = category_rate
    return metrics


def load_plus_metrics(path: Path) -> dict[str, float]:
    data = _read_json(path)
    if data.get("complete") is not True or data.get("total_episodes") != data.get("expected_episodes"):
        raise ValueError(f"Refusing incomplete LIBERO-Plus evaluation: {path}")
    metrics = {
        "validation/libero_plus_language_success_rate": _first_number(data, ("success_rate",), source=path),
        "validation/libero_plus_language_episodes": float(data["total_episodes"]),
    }
    for level, result in data.get("per_difficulty", {}).items():
        metrics[f"validation/libero_plus_language_difficulty/{level}"] = float(result["success_rate"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--libero-goal-summary", type=Path, required=True)
    parser.add_argument("--libero-para-summary", type=Path, required=True)
    parser.add_argument("--libero-plus-summary", type=Path)
    parser.add_argument("--project", default="openpi")
    parser.add_argument("--entity", default=None)
    args = parser.parse_args()

    run_id_path = args.checkpoint_dir / "wandb_id.txt"
    if not run_id_path.is_file():
        raise FileNotFoundError(f"Training W&B run id not found: {run_id_path}")

    metrics = {
        "validation/checkpoint_step": args.checkpoint_step,
        **load_goal_metrics(args.libero_goal_summary),
        **load_para_metrics(args.libero_para_summary),
    }
    metrics["validation/paraphrase_gap"] = (
        metrics["validation/libero_goal_success_rate"] - metrics["validation/libero_para_success_rate"]
    )
    if args.libero_plus_summary is not None:
        metrics.update(load_plus_metrics(args.libero_plus_summary))

    run = wandb.init(
        id=run_id_path.read_text(encoding="utf-8").strip(),
        resume="must",
        project=args.project,
        entity=args.entity,
    )
    run.define_metric("validation/checkpoint_step")
    run.define_metric("validation/*", step_metric="validation/checkpoint_step")
    run.log(metrics)
    run.summary["latest_validation_checkpoint_step"] = args.checkpoint_step
    run.summary["latest_libero_goal_success_rate"] = metrics["validation/libero_goal_success_rate"]
    run.summary["latest_libero_para_success_rate"] = metrics["validation/libero_para_success_rate"]
    run.summary["latest_paraphrase_gap"] = metrics["validation/paraphrase_gap"]
    if args.libero_plus_summary is not None:
        run.summary["latest_libero_plus_language_success_rate"] = metrics[
            "validation/libero_plus_language_success_rate"
        ]
    run.finish()

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
