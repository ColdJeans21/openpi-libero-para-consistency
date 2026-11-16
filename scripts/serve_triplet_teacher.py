"""Serve the official high-success LIBERO policy for offline triplet labeling."""

import argparse
import logging

import numpy as np

from openpi.evaluation.attention_protocol import array_digest
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


class ControlledNoisePolicy:
    def __init__(self, policy, *, horizon: int, action_dim: int):
        self.policy = policy
        self.horizon = horizon
        self.action_dim = action_dim

    def infer(self, observation):
        data = dict(observation)
        noise = np.asarray(data.pop("_evaluation_noise"), dtype=np.float32)
        if noise.shape != (self.horizon, self.action_dim) or not np.isfinite(noise).all():
            raise ValueError("Malformed controlled noise")
        result = self.policy.infer(data, noise=noise)
        result["evaluation_noise_sha256"] = array_digest(noise)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    cfg = config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    wrapped = ControlledNoisePolicy(
        policy,
        horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
    )
    metadata = {
        **policy.metadata,
        "triplet_teacher": "official_pi05_libero",
        "teacher_checkpoint": args.checkpoint,
        "action_horizon": cfg.model.action_horizon,
        "action_dim": cfg.model.action_dim,
        "controlled_noise": True,
    }
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
