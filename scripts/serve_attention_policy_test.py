from types import SimpleNamespace

import numpy as np
import pytest

from scripts.serve_attention_policy import EvaluationNoisePolicy


def test_matched_noise_is_forwarded_and_removed_from_model_observation():
    seen = []

    def infer(obs, **kwargs):
        seen.append((obs, kwargs))
        return {"actions": np.zeros((10, 7))}

    policy = EvaluationNoisePolicy(SimpleNamespace(infer=infer), horizon=10, action_dim=32)
    noise = np.ones((10, 32), np.float32)
    observation = {"prompt": "put bowl on plate", "_evaluation_noise": noise}
    result = policy.infer(observation)
    assert len(result["evaluation_noise_sha256"]) == 64
    np.testing.assert_array_equal(seen[0][1]["noise"], noise)
    assert "_evaluation_noise" not in seen[0][0]
    assert "_evaluation_noise" in observation
    assert "evaluation_noise_sha256" not in policy.infer({"prompt": "original"})
    assert seen[1][1] == {}
    for malformed in (np.zeros((1, 10, 32)), np.full((10, 32), np.nan)):
        with pytest.raises(ValueError, match="Malformed"):
            policy.infer({"_evaluation_noise": malformed})
