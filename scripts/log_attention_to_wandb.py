"""Upload attention evaluations into their own W&B run, leaving training runs intact."""

import json
from pathlib import Path

import wandb


def main():
    root = Path("reports/attention/v1")
    selection = json.loads((root / "selection.json").read_text())
    id_path = root / "wandb_id.txt"
    run_id = id_path.read_text().strip() if id_path.exists() else wandb.util.generate_id()
    id_path.write_text(run_id + "\n")
    run = wandb.init(
        entity="xzzsryouxiwang-shenzhen-university",
        project="openpi",
        id=run_id,
        resume="allow",
        name="attention_fm_first_v1",
        job_type="evaluation",
        config={
            "protocol": "attention_v1",
            "selected_beta": selection["selected_beta"],
            "selection_model": "fm",
            "manifest_sha256": selection["manifest_sha256"],
            "training": False,
            "seed": 7,
        },
    )
    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text())
        for key, value in data.items():
            if isinstance(value, float | int | bool):
                run.summary[f"counterfactual/{path.parent.name}/{key}"] = value
    if selection["selected_beta"] > 0:
        for model in ("fm", "teacher"):
            name = f"attention_{model}_selected_v1"
            validation = Path("reports/validation") / name / "step_29999"
            paths = {
                "libero_goal": validation / "libero_goal_seed7.json",
                "libero_para": validation / "libero_para_seed7/summary.json",
                "libero_plus_language": Path("reports/libero_plus_language") / name / "seed7/summary.json",
            }
            for label, path in paths.items():
                data = json.loads(path.read_text())
                run.summary[f"validation/{model}/{label}_success_rate"] = (
                    data["total_successes"] / data["total_episodes"]
                )
    run.summary["selection/attention_supported_on_dev"] = selection["attention_supported_on_dev"]
    run.finish()


if __name__ == "__main__":
    main()
