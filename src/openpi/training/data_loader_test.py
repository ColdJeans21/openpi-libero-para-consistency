import dataclasses
import json

import jax
import numpy as np
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_expand_episode_data_index_for_global_ids():
    compact = {
        "from": torch.tensor([0, 5, 12]),
        "to": torch.tensor([5, 12, 20]),
    }

    expanded = _data_loader.expand_episode_data_index_for_global_ids(compact, episodes=[10, 42, 99])

    assert expanded["from"][10].item() == 0
    assert expanded["to"][42].item() == 12
    assert expanded["from"][99].item() == 12
    assert expanded["to"][99].item() == 20


def test_triplet_npz_dataset_flattens_three_supervised_members(tmp_path):
    np.savez(
        tmp_path / "episode.npz",
        image=np.zeros((2, 4, 4, 3), np.uint8),
        wrist_image=np.ones((2, 4, 4, 3), np.uint8),
        state=np.zeros((2, 8), np.float32),
        action_anchor=np.ones((2, 3, 7), np.float32),
        action_counterfactual=np.full((2, 3, 7), 2, np.float32),
        canonical=np.asarray("put the bowl on the plate"),
        paraphrase=np.asarray("place the bowl onto the plate"),
        counterfactual=np.asarray("put the bowl on the stove"),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "shards": ["episode.npz"]}), encoding="utf-8"
    )

    dataset = _data_loader.TripletNpzDataset(tmp_path)

    assert len(dataset) == 6
    assert dataset[0]["prompt"] == "put the bowl on the plate"
    assert dataset[1]["prompt"] == "place the bowl onto the plate"
    assert dataset[2]["prompt"] == "put the bowl on the stove"
    np.testing.assert_array_equal(dataset[0]["actions"], dataset[1]["actions"])
    assert np.all(dataset[2]["actions"] == 2)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
