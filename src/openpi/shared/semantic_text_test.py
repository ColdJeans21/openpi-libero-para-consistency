import json

import numpy as np
import pytest

from openpi import transforms
from openpi.shared import semantic_text


def test_training_cache_is_strict_and_features_survive_tokenization(tmp_path):
    path = tmp_path / "text.npz"
    np.savez(
        path, texts=np.array(["push left", "move left"]), features=np.arange(16).reshape(2, 8), metadata=json.dumps({})
    )
    transform = transforms.SemanticPromptFeatures(str(path), feature_dim=8)
    result = transform({"prompt": " push left ", "prompt_alt": "move left"})
    np.testing.assert_array_equal(result["semantic_features"], np.arange(8))
    np.testing.assert_array_equal(result["semantic_features_alt"], np.arange(8, 16))
    assert result["prompt"] == "perform the task"
    assert result["prompt_alt"] == "perform the task"
    with pytest.raises(ValueError, match="Training prompt missing"):
        transform({"prompt": "push right"})


def test_inference_can_encode_unseen_text_without_altering_training_cache(tmp_path, monkeypatch):
    path = tmp_path / "text.npz"
    np.savez(path, texts=np.array(["known"]), features=np.zeros((1, 8)), metadata=json.dumps({}))
    before = path.read_bytes()
    monkeypatch.setattr(semantic_text, "encode_inference_prompt", lambda prompt, metadata: np.ones(8))
    transform = transforms.SemanticPromptFeatures(str(path), feature_dim=8, allow_online=True)
    np.testing.assert_array_equal(transform({"prompt": "unseen"})["semantic_features"], np.ones(8))
    assert before == path.read_bytes()
