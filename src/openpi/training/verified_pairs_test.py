import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.training.verified_pairs import MixedVerifiedPairs


def fixture_data(tmp_path):
    source = Path(__file__).resolve().parents[3] / "examples/libero_attention/manifest_v1.json"
    split = json.loads(source.read_text())
    records, shards, hashes = [], [], {}
    for pair in split["pairs"]:
        indices = [i for i in range(50) if i not in pair["dev_state_indices"] + pair["test_state_indices"]][:10]
        for i in indices:
            name = f"{pair['pair_id']}_{i}.npz"
            p = tmp_path / name
            np.savez(
                p,
                image=np.zeros((2, 2, 3), np.uint8),
                wrist_image=np.zeros((2, 2, 3), np.uint8),
                state=np.ones(8),
                actions=np.ones((10, 7)),
                actions_alt=np.full((10, 7), 3.0),
                prompt="original",
                prompt_alt="alternate",
            )
            records.append(
                {
                    "accepted": True,
                    "branch_success": [True, True],
                    "wrong_goal": [False, False],
                    "pair_id": pair["pair_id"],
                    "state_index": i,
                    "shard": name,
                }
            )
            shards.append(name)
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    m = {
        "schema_version": 2,
        "completed": True,
        "label_type": "both_successful_executed_prefix",
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "records": records,
        "shards": shards,
        "shard_sha256": hashes,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    return m


def test_mixture_preserves_both_transformed_labels_and_cache(tmp_path, monkeypatch):
    from openpi.training import data_loader

    fixture_data(tmp_path)

    # A nonidentity action transform detects raw/normalized-label mixing.
    def transform(rows, *_args, **_kwargs):
        return [{**r, "actions": np.pad(2 * r["actions"] - 1, ((0, 0), (0, 25)))} for r in rows]

    monkeypatch.setattr(data_loader, "transform_dataset", transform)
    base = [{"actions": np.zeros((10, 32)), "prompt": str(i)} for i in range(14)]
    dataset = MixedVerifiedPairs(base, tmp_path, None)
    assert len(dataset) == 16
    assert sum(bool(dataset[i]["pair_is_negative"]) for i in range(16)) == 2
    np.testing.assert_allclose(dataset[7]["actions"][:, :7], 1.0)
    np.testing.assert_allclose(dataset[7]["paired_actions"][:, :7], 5.0)
    np.testing.assert_allclose(dataset[15]["actions"][:, :7], 5.0)
    np.testing.assert_allclose(dataset[15]["paired_actions"][:, :7], 1.0)
    dataset[7]["actions"][:] = 99.0
    np.testing.assert_allclose(dataset[7]["actions"][:, :7], 1.0)
    assert [dataset[i]["prompt"] for i in range(16) if i % 8 != 7] == [str(i) for i in range(14)]


def test_rejects_failed_branch_and_changed_shard(tmp_path):
    m = fixture_data(tmp_path)
    m["records"][0]["branch_success"][1] = False
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="failed"):
        MixedVerifiedPairs([], tmp_path, None)
    m["records"][0]["branch_success"][1] = True
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    (tmp_path / m["shards"][0]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        MixedVerifiedPairs([], tmp_path, None)
