"""Matched scene/goal intervention evaluation; run inside libero:latest."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from openpi.evaluation.attention_protocol import array_digest
from openpi.evaluation.attention_protocol import bddl_sections
from openpi.evaluation.attention_protocol import file_digest
from openpi.evaluation.attention_protocol import goal_values
from openpi.evaluation.attention_protocol import matched_noise
from openpi.evaluation.attention_protocol import summarize_cases


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def model_digest(env):
    model = env.sim.model
    names = [model.body_id2name(i) for i in range(model.nbody)]
    data = {
        name: array_digest(getattr(model, name))
        for name in (
            "body_pos",
            "body_quat",
            "body_mass",
            "body_inertia",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_type",
            "geom_bodyid",
            "geom_friction",
            "cam_pos",
            "cam_quat",
            "cam_fovy",
            "site_pos",
        )
    }
    data["names"] = names
    data["dims"] = [model.nq, model.nv, model.nu]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def observation_digest(obs):
    return hashlib.sha256(
        "".join(
            array_digest(obs[k])
            for k in (
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            )
        ).encode()
    ).hexdigest()


def quat_axis(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1 - quat[3] ** 2)
    return np.zeros(3) if math.isclose(float(den), 0) else quat[:3] * 2 * math.acos(float(quat[3])) / den


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--model", choices=("fm", "teacher"), required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    manifest = json.loads(args.manifest.read_text())
    manifest_sha = file_digest(args.manifest)
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=args.port)
    metadata = client.get_server_metadata()
    if metadata.get("attention_model") != args.model or not math.isclose(metadata.get("attention_beta", -1), args.beta):
        raise ValueError("Policy server model/beta do not match requested experiment")
    if not metadata.get("matched_evaluation_noise"):
        raise ValueError("Policy server cannot accept controlled noise")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": args.model,
        "beta": args.beta,
        "split": args.split,
        "manifest_sha256": manifest_sha,
        "server": metadata,
        "max_steps": args.max_steps,
        "wait_steps": 10,
        "replan_steps": 5,
    }
    meta_path = args.output_dir / "protocol.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != protocol:
        raise ValueError("Cannot resume with a different evaluation protocol")
    write_json(meta_path, protocol)
    records_path = args.output_dir / "episodes.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    completed = {r["case_id"] for r in records}
    if len(completed) != len(records):
        raise ValueError("Duplicate completed case")
    expected = [
        f"{p['pair_id']}:{idx}:{target}:{wording}"
        for p in manifest["pairs"]
        for idx in p[f"{args.split}_state_indices"]
        for target in range(2)
        for wording in ("canonical", "paraphrase")
    ]
    bddl_dir = Path(get_libero_path("bddl_files")) / "libero_goal"
    initial_dir = Path(get_libero_path("init_states")) / "libero_goal"
    dummy = [0.0] * 6 + [-1.0]
    with records_path.open("a") as output:
        for pair in manifest["pairs"]:
            envs = []
            try:
                for target in pair["targets"]:
                    path = bddl_dir / f"{target['task']}.bddl"
                    if (
                        file_digest(path) != target["bddl_sha256"]
                        or bddl_sections(path.read_text())[":goal"] != target["goal"]
                    ):
                        raise ValueError("BDDL or goal differs from frozen manifest")
                    np.random.seed(manifest["seed"])
                    env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256, camera_widths=256)
                    env.seed(manifest["seed"])
                    env.reset()
                    envs.append(env)
                shared_xml = envs[0].sim.model.get_xml()
                frozen = {
                    name: getattr(envs[0].sim.model, name).copy()
                    for name in ("body_pos", "body_quat", "geom_pos", "geom_quat", "cam_pos", "cam_quat", "site_pos")
                }
                # Establish the fingerprint AFTER compiling the common XML.
                # Original construction vs XML roundtrip can differ in floating
                # serialization, while all evaluated conditions must match the
                # same compiled common model exactly.
                envs[0].reset_from_xml_string(shared_xml)
                for name, array in frozen.items():
                    getattr(envs[0].sim.model, name)[:] = array
                reference_model = model_digest(envs[0])
                initial_file = initial_dir / f"{pair['initial_state_source']}.pruned_init"
                if file_digest(initial_file) != pair["initial_states_sha256"]:
                    raise ValueError("Initial-state source changed")
                try:
                    states = torch.load(initial_file, map_location="cpu", weights_only=False)
                except TypeError:
                    states = torch.load(initial_file, map_location="cpu")
                for state_index in pair[f"{args.split}_state_indices"]:
                    for target_index, env in enumerate(envs):
                        for wording in ("canonical", "paraphrase"):
                            case_id = f"{pair['pair_id']}:{state_index}:{target_index}:{wording}"
                            if case_id in completed:
                                continue
                            # Restore the SAME physical XML, fixture positions and
                            # dynamic state. Target goal parsing stays environment-specific.
                            env.reset_from_xml_string(shared_xml)
                            for name, array in frozen.items():
                                getattr(env.sim.model, name)[:] = array
                            if model_digest(env) != reference_model:
                                raise ValueError("The two goals do not share an identical physics model")
                            obs = env.set_init_state(np.asarray(states[state_index]))
                            for _ in range(10):
                                obs, _, _, _ = env.step(dummy)
                            obs_sha = observation_digest(obs)
                            audit_path = args.audit_dir / f"{pair['pair_id']}_{state_index}.json"
                            audit = {
                                "model_sha256": reference_model,
                                "observation_sha256": obs_sha,
                                "manifest_sha256": manifest_sha,
                            }
                            if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
                                raise ValueError("Initial physics/images/state changed across instructions or methods")
                            write_json(audit_path, audit)

                            if any(goal_values(env, pair["targets"])):
                                raise ValueError("Initial state already satisfies a target; refuse trivial evaluation")
                            plan = collections.deque()
                            queries = 0
                            wrong_ever = False
                            success = False
                            for _step in range(args.max_steps):
                                if not plan:
                                    noise = matched_noise(
                                        pair["pair_id"],
                                        state_index,
                                        queries,
                                        seed=manifest["seed"],
                                        horizon=metadata["action_horizon"],
                                        action_dim=metadata["action_dim"],
                                    )
                                    request = {
                                        "observation/image": image_tools.convert_to_uint8(
                                            image_tools.resize_with_pad(
                                                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224
                                            )
                                        ),
                                        "observation/wrist_image": image_tools.convert_to_uint8(
                                            image_tools.resize_with_pad(
                                                np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                                                224,
                                                224,
                                            )
                                        ),
                                        "observation/state": np.concatenate(
                                            (
                                                obs["robot0_eef_pos"],
                                                quat_axis(obs["robot0_eef_quat"]),
                                                obs["robot0_gripper_qpos"],
                                            )
                                        ),
                                        "prompt": pair["targets"][target_index][wording],
                                        "_evaluation_noise": noise,
                                    }
                                    prediction = client.infer(request)
                                    if prediction.get("evaluation_noise_sha256") != array_digest(noise):
                                        raise ValueError("Server did not use supplied evaluation noise")
                                    actions = np.asarray(prediction["actions"])
                                    if actions.shape[0] < 5 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
                                        raise ValueError("Invalid policy action chunk")
                                    plan.extend(actions[:5])
                                    queries += 1
                                obs, _, _, _ = env.step(np.asarray(plan.popleft()).tolist())
                                values = goal_values(env, pair["targets"])
                                wrong_ever |= values[1 - target_index]
                                if values[target_index]:
                                    success = True
                                    break
                            values = goal_values(env, pair["targets"])
                            record = {
                                "case_id": case_id,
                                "pair_id": pair["pair_id"],
                                "state_index": state_index,
                                "target_index": target_index,
                                "wording": wording,
                                "success": success,
                                "instruction": pair["targets"][target_index][wording],
                                "wrong_goal_at_end": bool(values[1 - target_index]),
                                "wrong_goal_ever": bool(wrong_ever),
                                "steps": _step + 1,
                                "action_queries": queries,
                                "initial_observation_sha256": obs_sha,
                                "manifest_sha256": manifest_sha,
                            }
                            output.write(json.dumps(record) + "\n")
                            output.flush()
                            os.fsync(output.fileno())
                            records.append(record)
                            completed.add(case_id)
                            summary = {**summarize_cases(records, expected, manifest_sha), **protocol}
                            write_json(args.output_dir / "summary.json", summary)
                            print(f"[{len(records)}/{len(expected)}] {case_id}: success={success}", flush=True)
            finally:
                for env in envs:
                    env.close()
    write_json(args.output_dir / "summary.json", {**summarize_cases(records, expected, manifest_sha), **protocol})


if __name__ == "__main__":
    main()
