"""Collect same-state language/action triplets from successful teacher rollouts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from openpi.evaluation.attention_protocol import array_digest
from openpi.evaluation.attention_protocol import bddl_sections
from openpi.evaluation.attention_protocol import file_digest
from openpi.evaluation.attention_protocol import goal_values
from openpi.evaluation.attention_protocol import matched_noise


def quat_axis(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1 - quat[3] ** 2)
    return np.zeros(3) if np.isclose(float(den), 0) else quat[:3] * 2 * np.arccos(float(quat[3])) / den


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def request_from_obs(obs, prompt, noise, image_tools):
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224
        )
    )
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat_axis(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
        "_evaluation_noise": noise,
    }, image, wrist, state


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

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_sha = file_digest(args.manifest)
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=args.port)
    server = client.get_server_metadata()
    if server.get("triplet_teacher") != "official_pi05_libero" or not server.get("controlled_noise"):
        raise ValueError("Triplet collection requires the frozen official teacher server")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = args.output_dir / "manifest.json"
    output_manifest = (
        json.loads(output_manifest_path.read_text(encoding="utf-8"))
        if output_manifest_path.exists()
        else {
            "schema_version": 1,
            "name": "libero_same_state_cfd_pilot_v1",
            "label_type": "successful_official_pi05_pseudo_action_chunk",
            "source_manifest_sha256": source_sha,
            "teacher": server,
            "held_out": "all manifest dev/test state indices",
            "shards": [],
            "shard_sha256": {},
            "attempted_episodes": 0,
            "successful_episodes": 0,
        }
    )
    if output_manifest["source_manifest_sha256"] != source_sha or output_manifest["teacher"] != server:
        raise ValueError("Refusing to resume with a changed manifest or teacher")
    completed = {Path(item).stem for item in output_manifest["shards"]}
    bddl_dir = Path(get_libero_path("bddl_files")) / "libero_goal"
    initial_dir = Path(get_libero_path("init_states")) / "libero_goal"
    dummy = [0.0] * 6 + [-1.0]

    for pair in source["pairs"]:
        held_out = set(pair["dev_state_indices"]) | set(pair["test_state_indices"])
        train_indices = [index for index in range(50) if index not in held_out]
        envs = []
        try:
            for target in pair["targets"]:
                path = bddl_dir / f"{target['task']}.bddl"
                if file_digest(path) != target["bddl_sha256"] or bddl_sections(path.read_text())[":goal"] != target["goal"]:
                    raise ValueError("Frozen BDDL task changed")
                np.random.seed(source["seed"])
                env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256, camera_widths=256)
                env.seed(source["seed"])
                env.reset()
                envs.append(env)
            shared_xml = envs[0].sim.model.get_xml()
            frozen = {
                name: getattr(envs[0].sim.model, name).copy()
                for name in ("body_pos", "body_quat", "geom_pos", "geom_quat", "cam_pos", "cam_quat", "site_pos")
            }
            initial_file = initial_dir / f"{pair['initial_state_source']}.pruned_init"
            try:
                states = torch.load(initial_file, map_location="cpu", weights_only=False)
            except TypeError:
                states = torch.load(initial_file, map_location="cpu")

            for state_index in train_indices:
                for target_index, env in enumerate(envs):
                    stem = f"{pair['pair_id']}_state{state_index:02d}_target{target_index}"
                    if stem in completed:
                        continue
                    output_manifest["attempted_episodes"] += 1
                    env.reset_from_xml_string(shared_xml)
                    for name, array in frozen.items():
                        getattr(env.sim.model, name)[:] = array
                    obs = env.set_init_state(np.asarray(states[state_index]))
                    for _ in range(10):
                        obs, _, _, _ = env.step(dummy)
                    plan = collections.deque()
                    queries = 0
                    wrong_ever = False
                    rows = {key: [] for key in ("image", "wrist_image", "state", "action_anchor", "action_counterfactual")}
                    for _step in range(args.max_steps):
                        if not plan:
                            noise = matched_noise(
                                pair["pair_id"],
                                state_index,
                                queries,
                                seed=source["seed"],
                                horizon=server["action_horizon"],
                                action_dim=server["action_dim"],
                            )
                            anchor_prompt = pair["targets"][target_index]["canonical"]
                            negative_prompt = pair["targets"][1 - target_index]["canonical"]
                            request, image, wrist, state = request_from_obs(obs, anchor_prompt, noise, image_tools)
                            anchor = client.infer(request)
                            negative_request = dict(request)
                            negative_request["prompt"] = negative_prompt
                            negative = client.infer(negative_request)
                            expected = array_digest(noise)
                            if anchor.get("evaluation_noise_sha256") != expected or negative.get("evaluation_noise_sha256") != expected:
                                raise ValueError("Teacher ignored controlled noise")
                            anchor_actions = np.asarray(anchor["actions"], np.float32)
                            negative_actions = np.asarray(negative["actions"], np.float32)
                            if anchor_actions.shape != negative_actions.shape or anchor_actions.shape[-1] != 7:
                                raise ValueError("Teacher returned malformed actions")
                            rows["image"].append(image)
                            rows["wrist_image"].append(wrist)
                            rows["state"].append(state)
                            rows["action_anchor"].append(anchor_actions)
                            rows["action_counterfactual"].append(negative_actions)
                            plan.extend(anchor_actions[:5])
                            queries += 1
                        obs, _, _, _ = env.step(np.asarray(plan.popleft()).tolist())
                        values = goal_values(env, pair["targets"])
                        wrong_ever |= values[1 - target_index]
                        if values[target_index]:
                            break
                    success = bool(goal_values(env, pair["targets"])[target_index]) and not wrong_ever
                    if success:
                        relative = f"shards/{stem}.npz"
                        path = args.output_dir / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        tmp = path.with_suffix(".tmp.npz")
                        np.savez_compressed(
                            tmp,
                            **{key: np.asarray(value) for key, value in rows.items()},
                            canonical=np.asarray(pair["targets"][target_index]["canonical"]),
                            paraphrase=np.asarray(pair["targets"][target_index]["paraphrase"]),
                            counterfactual=np.asarray(pair["targets"][1 - target_index]["canonical"]),
                            pair_id=np.asarray(pair["pair_id"]),
                            state_index=np.asarray(state_index),
                            target_index=np.asarray(target_index),
                        )
                        tmp.replace(path)
                        output_manifest["shards"].append(relative)
                        output_manifest["shard_sha256"][relative] = digest(path)
                        output_manifest["successful_episodes"] += 1
                        completed.add(stem)
                    atomic_json(output_manifest_path, output_manifest)
                    print(
                        f"{stem}: success={success}, queries={queries}, "
                        f"kept={output_manifest['successful_episodes']}/{output_manifest['attempted_episodes']}",
                        flush=True,
                    )
        finally:
            for env in envs:
                env.close()

    output_manifest["completed"] = True
    atomic_json(output_manifest_path, output_manifest)


if __name__ == "__main__":
    main()
