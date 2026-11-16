"""Serve a frozen Pi0 attention intervention with optional matched evaluation noise."""

import argparse
import dataclasses
import logging

import numpy as np

from openpi.evaluation.attention_protocol import array_digest
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


class EvaluationNoisePolicy:
    def __init__(self, policy, *, horizon, action_dim):
        self.policy, self.horizon, self.action_dim = policy, horizon, action_dim

    def infer(self, observation):
        data = dict(observation)
        noise = data.pop("_evaluation_noise", None)
        if noise is None:
            return self.policy.infer(data)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (self.horizon, self.action_dim) or not np.isfinite(noise).all():
            raise ValueError("Malformed matched evaluation noise")
        result = self.policy.infer(data, noise=noise)
        result["evaluation_noise_sha256"] = array_digest(noise)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fm", "teacher"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    name = "pi0_libero_goal_lora_canonical" if args.model == "fm" else "pi0_libero_goal_lora_consistency"
    cfg = config.get_config(name)
    model = dataclasses.replace(cfg.model, instruction_attention_bias=args.beta)
    if args.model == "teacher":
        model = dataclasses.replace(model, paired_loss_mode="teacher_consistency", consistency_loss_weight=1.0)
    cfg = dataclasses.replace(cfg, model=model)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    wrapped = EvaluationNoisePolicy(policy, horizon=model.action_horizon, action_dim=model.action_dim)
    metadata = {
        **policy.metadata,
        "attention_model": args.model,
        "attention_beta": args.beta,
        "attention_scope": "all_layers_all_heads_action_queries_to_instruction_content",
        "action_horizon": model.action_horizon,
        "action_dim": model.action_dim,
        "checkpoint": args.checkpoint,
        "matched_evaluation_noise": True,
    }
    # A counterfactual episode can spend a long time in MuJoCo between policy
    # queries. Disable protocol keepalives for this local, matched-noise study
    # so a slow environment step cannot invalidate an otherwise healthy run.
    WebsocketPolicyServer(
        wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
        ping_interval=None,
        ping_timeout=None,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
