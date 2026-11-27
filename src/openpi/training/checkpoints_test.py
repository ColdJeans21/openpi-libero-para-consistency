from types import SimpleNamespace
from unittest import mock

from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training import checkpoints
from openpi.training import utils


def test_bounded_checkpoint_handlers(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_CHECKPOINT_IO_GB", "4")
    handler = checkpoints.ocp.PyTreeCheckpointHandler
    with mock.patch.object(checkpoints.ocp, "PyTreeCheckpointHandler", wraps=handler) as factory:
        manager, _ = checkpoints.initialize_checkpoint_dir(tmp_path / "run", keep_period=5, overwrite=False, resume=False)
    try:
        assert len(factory.call_args_list) == 2
        for call in factory.call_args_list:
            assert call.kwargs["save_concurrent_gb"] == 4
            assert call.kwargs["restore_concurrent_gb"] == 4
    finally:
        manager.close()


def test_waited_save_restores_optimizer_and_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_CHECKPOINT_IO_GB", "4")
    monkeypatch.setenv("OPENPI_CHECKPOINT_WAIT", "1")
    graph, params = nnx.split(nnx.Linear(2, 2, rngs=nnx.Rngs(0)))
    tx = optax.adam(1e-3)
    state = utils.TrainState(
        step=jnp.asarray(21), params=params, model_def=graph,
        opt_state=tx.init(params), tx=tx, ema_decay=None,
    )
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, semantic_cache_path=None))
    manager, _ = checkpoints.initialize_checkpoint_dir(tmp_path / "run", keep_period=5, overwrite=False, resume=False)
    try:
        with mock.patch.object(manager, "wait_until_finished", wraps=manager.wait_until_finished) as wait:
            checkpoints.save_state(manager, state, loader, 20)
            assert wait.called, "Training must not overlap checkpoint commit in memory-constrained mode"
        assert (tmp_path / "run/20/_CHECKPOINT_METADATA").is_file()
        restored = checkpoints.restore_state(manager, state, loader)
        assert int(restored.step) == 21
        np.testing.assert_array_equal(restored.params["kernel"].value, state.params["kernel"].value)
        assert int(restored.opt_state[0].count) == 0
    finally:
        manager.close()
