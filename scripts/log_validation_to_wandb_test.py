import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "validation_logger", Path(__file__).with_name("log_validation_to_wandb.py")
)
logger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logger)
namespace_metrics = logger.namespace_metrics


def test_primary_metrics_keep_their_existing_names():
    metrics = {"validation/checkpoint_step": 29999, "validation/libero_goal_success_rate": 0.8}
    assert namespace_metrics(metrics, "validation") == metrics


def test_residual_off_cannot_overwrite_primary_metric_keys():
    metrics = {"validation/checkpoint_step": 29999, "validation/libero_goal_success_rate": 0.8}
    result = namespace_metrics(metrics, "validation_residual_off")
    assert not metrics.keys() & result.keys()
    assert result["validation_residual_off/libero_goal_success_rate"] == 0.8
    with pytest.raises(ValueError, match="namespace"):
        namespace_metrics(metrics, "validation/anything")
