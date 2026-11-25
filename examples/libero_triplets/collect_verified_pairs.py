"""Collect initial-state counterfactual pairs with BOTH branches executed successfully.

This is supplemental data, never a replacement for the Goal demonstrations.
Labels are the first H executed controls of each successful receding-horizon
rollout, not an unexecuted alternative prediction or aligned full trajectories.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from examples.libero_triplets.collect_teacher_triplets import atomic_json
from examples.libero_triplets.collect_teacher_triplets import digest
from examples.libero_triplets.collect_teacher_triplets import request_from_obs
from openpi.evaluation.attention_protocol import array_digest
from openpi.evaluation.attention_protocol import bddl_sections
from openpi.evaluation.attention_protocol import file_digest
from openpi.evaluation.attention_protocol import goal_values
from openpi.evaluation.attention_protocol import matched_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    source = json.loads(args.manifest.read_text())
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=args.port)
    teacher = client.get_server_metadata()
    if teacher.get("triplet_teacher") != "official_pi05_libero" or not teacher.get("controlled_noise"):
        raise ValueError("Expected frozen official teacher with controlled noise")
    path = args.output_dir / "manifest.json"
    identity = {
        "schema_version": 2,
        "name": "verified_initial_pairs_v2",
        "teacher": teacher,
        "source_manifest_sha256": file_digest(args.manifest),
        "max_steps": args.max_steps,
        "label_type": "both_successful_executed_prefix",
        "horizon": 10,
    }
    manifest = (
        json.loads(path.read_text()) if path.exists() else {**identity, "records": [], "shards": [], "shard_sha256": {}}
    )
    if any(manifest.get(k) != v for k, v in identity.items()):
        raise ValueError("Resume identity changed")
    completed = {r["id"] for r in manifest["records"]}
    for pair in source["pairs"]:
        held_out = set(pair["dev_state_indices"]) | set(pair["test_state_indices"])
        envs = []
        try:
            for target in pair["targets"]:
                bddl = Path(get_libero_path("bddl_files")) / "libero_goal" / (target["task"] + ".bddl")
                if (
                    file_digest(bddl) != target["bddl_sha256"]
                    or bddl_sections(bddl.read_text())[":goal"] != target["goal"]
                ):
                    raise ValueError("Task definition changed")
                np.random.seed(source["seed"])
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                env.seed(source["seed"])
                env.reset()
                envs.append(env)
            xml = envs[0].sim.model.get_xml()
            geometry = {
                k: getattr(envs[0].sim.model, k).copy()
                for k in ("body_pos", "body_quat", "geom_pos", "geom_quat", "cam_pos", "cam_quat", "site_pos")
            }
            initial_path = (
                Path(get_libero_path("init_states")) / "libero_goal" / (pair["initial_state_source"] + ".pruned_init")
            )
            try:
                states = torch.load(initial_path, map_location="cpu", weights_only=False)
            except TypeError:
                states = torch.load(initial_path, map_location="cpu")
            for index in range(50):
                key = f"{pair['pair_id']}_state{index:02d}"
                if index in held_out or key in completed:
                    continue
                branches = []
                for target_index, env in enumerate(envs):
                    env.reset_from_xml_string(xml)
                    for name, value in geometry.items():
                        getattr(env.sim.model, name)[:] = value
                    obs = env.set_init_state(np.asarray(states[index]))
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    physics_hash = array_digest(env.sim.get_state().flatten())
                    prompt = pair["targets"][target_index]["canonical"]
                    noise = matched_noise(pair["pair_id"], index, 0, seed=source["seed"], horizon=10, action_dim=32)
                    _, image, wrist, state = request_from_obs(obs, prompt, noise, image_tools)
                    input_hash = [array_digest(x) for x in (image, wrist, state)]
                    actions, queue, query, wrong = [], collections.deque(), 0, False
                    initial_goals = goal_values(env, pair["targets"])
                    for _ in range(args.max_steps):
                        if any(initial_goals):
                            break  # Do not train on a goal already achieved at the common start.
                        if not queue:
                            noise = matched_noise(
                                pair["pair_id"], index, query, seed=source["seed"], horizon=10, action_dim=32
                            )
                            request, *_ = request_from_obs(obs, prompt, noise, image_tools)
                            reply = client.infer(request)
                            chunk = np.asarray(reply["actions"], np.float32)
                            if (
                                chunk.shape != (10, 7)
                                or not np.isfinite(chunk).all()
                                or reply.get("evaluation_noise_sha256") != array_digest(noise)
                            ):
                                raise ValueError("Invalid teacher response")
                            queue.extend(chunk[:5])
                            query += 1
                        action = np.asarray(queue.popleft())
                        actions.append(action)
                        obs, _, _, _ = env.step(action.tolist())
                        goals = goal_values(env, pair["targets"])
                        wrong |= bool(goals[1 - target_index])
                        if goals[target_index]:
                            break
                    success = (
                        bool(goal_values(env, pair["targets"])[target_index])
                        and not wrong
                        and len(actions) >= 10
                        and not any(initial_goals)
                    )
                    branches.append(
                        {
                            "success": success,
                            "wrong": wrong,
                            "actions": np.asarray(actions, np.float32),
                            "image": image,
                            "wrist": wrist,
                            "state": state,
                            "input_hash": input_hash,
                            "physics_hash": physics_hash,
                        }
                    )
                same_start = (
                    branches[0]["input_hash"] == branches[1]["input_hash"]
                    and branches[0]["physics_hash"] == branches[1]["physics_hash"]
                )
                # Reject the record rather than relaxing exact-state equality.
                keep = same_start and all(b["success"] for b in branches)
                record = {
                    "id": key,
                    "pair_id": pair["pair_id"],
                    "state_index": index,
                    "accepted": keep,
                    "same_start": same_start,
                    "branch_initial_input_sha256": [b["input_hash"] for b in branches],
                    "branch_initial_physics_sha256": [b["physics_hash"] for b in branches],
                    "branch_success": [b["success"] for b in branches],
                    "wrong_goal": [b["wrong"] for b in branches],
                    "lengths": [len(b["actions"]) for b in branches],
                    "initial_input_sha256": branches[0]["input_hash"],
                    "initial_physics_sha256": branches[0]["physics_hash"],
                }
                if keep:
                    relative = f"shards/{key}.npz"
                    out = args.output_dir / relative
                    out.parent.mkdir(parents=True, exist_ok=True)
                    a, b = branches
                    np.savez_compressed(
                        out,
                        image=a["image"],
                        wrist_image=a["wrist"],
                        state=a["state"],
                        actions=a["actions"][:10],
                        actions_alt=b["actions"][:10],
                        rollout0=a["actions"],
                        rollout1=b["actions"],
                        prompt=pair["targets"][0]["canonical"],
                        prompt_alt=pair["targets"][1]["canonical"],
                    )
                    manifest["shards"].append(relative)
                    manifest["shard_sha256"][relative] = digest(out)
                    record["shard"] = relative
                manifest["records"].append(record)
                atomic_json(path, manifest)
                print(
                    f"{key}: both_success={keep}, accepted={len(manifest['shards'])}/{len(manifest['records'])}",
                    flush=True,
                )
        finally:
            for env in envs:
                env.close()
    manifest["completed"] = True
    atomic_json(path, manifest)


if __name__ == "__main__":
    main()
