#!/usr/bin/env python3
"""Evaluate an OpenPI policy on LIBERO-Plus language perturbations.

This evaluator deliberately isolates the language factor.  It reads the
official LIBERO-Plus task classification and rewritten instructions, while
using the unmodified LIBERO-Goal environments and initial state zero.  The
official language entries encode the neutral camera and robot settings, so the
only changed policy input is the instruction string.

Results are appended to ``episodes.jsonl`` after every rollout, which makes a
long evaluation safe to resume after interruption.  ``summary.json`` contains
overall, per-difficulty, and per-base-task success rates.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LANGUAGE_CATEGORY = "Language Instructions"
NEUTRAL_ENV_SUFFIX = "0_0_100_0_0_initstate_0"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bddl_instruction(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("(:language"):
                return stripped[len("(:language") :].rstrip(")").strip()
    raise ValueError(f"No (:language ...) instruction found in {path}")


def parse_official_task_name(name: str) -> tuple[str, str, int]:
    """Return language BDDL stem, original task stem, and language variant."""
    if "_view_" not in name:
        raise ValueError(f"Expected LIBERO-Plus composite task name, got: {name}")
    bddl_stem, environment_suffix = name.split("_view_", 1)
    if environment_suffix != NEUTRAL_ENV_SUFFIX:
        raise ValueError(
            f"Language-only evaluation requires the neutral physical setting; got {environment_suffix!r} for {name}"
        )
    match = re.fullmatch(r"(.+)_language_(\d+)", bddl_stem)
    if match is None:
        raise ValueError(f"Cannot parse language variant from: {name}")
    return bddl_stem, match.group(1), int(match.group(2))


def load_language_tasks(
    libero_plus_root: Path,
    suite: str,
    *,
    difficulty_levels: set[int] | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    classification_path = libero_plus_root / "libero/libero/benchmark/task_classification.json"
    classification = _read_json(classification_path)
    rows = classification.get(suite)
    if not isinstance(rows, list):
        raise KeyError(f"Suite {suite!r} is missing from {classification_path}")

    bddl_dir = libero_plus_root / "libero/libero/bddl_files" / suite
    tasks: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("category") != LANGUAGE_CATEGORY:
            continue
        difficulty = row.get("difficulty_level")
        if not isinstance(difficulty, int):
            raise ValueError(f"Invalid difficulty level in classification row: {row}")
        if difficulty_levels is not None and difficulty not in difficulty_levels:
            continue

        official_name = row.get("name")
        official_id = row.get("id")
        if not isinstance(official_name, str) or not isinstance(official_id, int):
            raise ValueError(f"Invalid classification row: {row}")
        bddl_stem, base_task, variant = parse_official_task_name(official_name)
        language_bddl = bddl_dir / f"{bddl_stem}.bddl"
        if not language_bddl.is_file():
            raise FileNotFoundError(f"Official language BDDL is missing: {language_bddl}")
        tasks.append(
            {
                "official_id": official_id,
                "official_task_name": official_name,
                "base_task": base_task,
                "language_variant": variant,
                "difficulty_level": difficulty,
                "language_bddl": str(language_bddl),
                "instruction": parse_bddl_instruction(language_bddl),
            }
        )

    if not tasks:
        raise ValueError(f"No {LANGUAGE_CATEGORY!r} tasks found for suite {suite!r}")
    official_names = [task["official_task_name"] for task in tasks]
    if len(official_names) != len(set(official_names)):
        raise ValueError("LIBERO-Plus classification contains duplicate task names")
    return tasks, classification_path


def _torch_load_compat(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("official_task_name")
            if not isinstance(name, str):
                raise ValueError(f"Missing official_task_name at {path}:{line_number}")
            if name in completed:
                raise ValueError(f"Duplicate completed task {name!r} in {path}")
            completed[name] = record
    return completed


def _rate_summary(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for record in records:
        group = str(record[key])
        totals[group][0] += int(bool(record["success"]))
        totals[group][1] += 1
    return {
        group: {
            "successes": values[0],
            "total": values[1],
            "success_rate": values[0] / values[1],
        }
        for group, values in sorted(totals.items())
    }


def write_summary(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    model_name: str,
    suite: str,
    seed: int,
    expected_tasks: int,
    classification_path: Path,
) -> dict[str, Any]:
    successes = sum(int(bool(record["success"])) for record in records)
    summary = {
        "benchmark": "LIBERO-Plus",
        "factor": LANGUAGE_CATEGORY,
        "suite": suite,
        "model_name": model_name,
        "seed": seed,
        "initial_state_index": 0,
        "num_trials_per_task": 1,
        "total_episodes": len(records),
        "expected_episodes": expected_tasks,
        "total_successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "complete": len(records) == expected_tasks,
        "per_difficulty": _rate_summary(records, "difficulty_level"),
        "per_base_task": _rate_summary(records, "base_task"),
        "classification_path": str(classification_path),
        "classification_sha256": _sha256(classification_path),
        "protocol": "official language classification; neutral camera/robot; one rollout; init state 0",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--difficulty-level", type=int, action="append", choices=range(1, 6))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks, classification_path = load_language_tasks(
        args.libero_plus_root.resolve(),
        args.suite,
        difficulty_levels=set(args.difficulty_level) if args.difficulty_level else None,
    )
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    base_bddl_dir = Path(get_libero_path("bddl_files")) / args.suite
    base_init_dir = Path(get_libero_path("init_states")) / args.suite
    for task in tasks:
        base_task = task["base_task"]
        base_bddl = base_bddl_dir / f"{base_task}.bddl"
        base_init = base_init_dir / f"{base_task}.pruned_init"
        if not base_bddl.is_file() or not base_init.is_file():
            raise FileNotFoundError(f"Missing base environment for {base_task}: {base_bddl}, {base_init}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    completed = _load_completed(episodes_path)
    expected_names = {task["official_task_name"] for task in tasks}
    unexpected = set(completed) - expected_names
    if unexpected:
        raise ValueError(f"Output contains tasks outside this evaluation selection: {sorted(unexpected)[:3]}")
    remaining = [task for task in tasks if task["official_task_name"] not in completed]
    logging.info(
        "LIBERO-Plus language tasks: %d total, %d completed, %d remaining",
        len(tasks),
        len(completed),
        len(remaining),
    )

    if remaining:
        client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        envs: dict[str, Any] = {}
        initial_states: dict[str, Any] = {}
        try:
            with episodes_path.open("a", encoding="utf-8") as output_file:
                for task in tqdm(remaining, desc="LIBERO-Plus Language"):
                    base_task = task["base_task"]
                    if base_task not in envs:
                        env = OffScreenRenderEnv(
                            bddl_file_name=base_bddl_dir / f"{base_task}.bddl",
                            camera_heights=LIBERO_ENV_RESOLUTION,
                            camera_widths=LIBERO_ENV_RESOLUTION,
                        )
                        env.seed(args.seed)
                        envs[base_task] = env
                        initial_states[base_task] = _torch_load_compat(base_init_dir / f"{base_task}.pruned_init")

                    env = envs[base_task]
                    env.reset()
                    obs = env.set_init_state(initial_states[base_task][0])
                    for _ in range(args.num_steps_wait):
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                    action_plan: collections.deque[Any] = collections.deque()
                    success = False
                    action_queries = 0
                    for _step in range(args.max_steps):
                        if not action_plan:
                            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            image = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(image, args.resize_size, args.resize_size)
                            )
                            wrist_image = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_image, args.resize_size, args.resize_size)
                            )
                            proprio = np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"].copy()),
                                    obs["robot0_gripper_qpos"],
                                )
                            )
                            element = {
                                "observation/image": image,
                                "observation/wrist_image": wrist_image,
                                "observation/state": proprio,
                                "prompt": task["instruction"],
                            }
                            action_chunk = client.infer(element)["actions"]
                            if len(action_chunk) < args.replan_steps:
                                raise RuntimeError(
                                    f"Policy returned {len(action_chunk)} actions; need {args.replan_steps}"
                                )
                            action_plan.extend(action_chunk[: args.replan_steps])
                            action_queries += 1

                        action = np.asarray(action_plan.popleft())
                        obs, _, done, _ = env.step(action.tolist())
                        if env.check_success() or done:
                            success = True
                            break

                    record = {
                        **task,
                        "language_bddl": Path(task["language_bddl"]).name,
                        "success": success,
                        "num_steps": _step + 1,
                        "action_queries": action_queries,
                        "seed": args.seed,
                        "initial_state_index": 0,
                    }
                    output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                    output_file.flush()
                    os.fsync(output_file.fileno())
                    completed[task["official_task_name"]] = record
                    current_successes = sum(int(bool(item["success"])) for item in completed.values())
                    print(
                        f"[{len(completed)}/{len(tasks)}] {'OK' if success else 'FAIL'} "
                        f"L{task['difficulty_level']} {task['instruction'][:70]} | "
                        f"SR={current_successes / len(completed) * 100:.1f}%"
                    )
        finally:
            for env in envs.values():
                env.close()

    records = [completed[task["official_task_name"]] for task in tasks if task["official_task_name"] in completed]
    summary = write_summary(
        output_dir,
        records,
        model_name=args.model_name,
        suite=args.suite,
        seed=args.seed,
        expected_tasks=len(tasks),
        classification_path=classification_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
