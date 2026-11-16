import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.evaluation.attention_protocol import BETAS
from openpi.evaluation.attention_protocol import bddl_sections
from openpi.evaluation.attention_protocol import goal_values
from openpi.evaluation.attention_protocol import matched_noise
from openpi.evaluation.attention_protocol import physical_signature
from openpi.evaluation.attention_protocol import select_beta
from openpi.evaluation.attention_protocol import summarize_cases


def test_bddl_comparison_ignores_only_goal_language_and_interest():
    first = bddl_sections(
        "(define (problem X) (:objects bowl plate) (:init (On bowl table)) (:goal (And (On bowl plate))) (:language put bowl on plate))"
    )
    other = bddl_sections(
        "(define (problem X) (:objects bowl plate) (:init (On bowl table)) (:goal (And (On bowl table))) (:language move bowl))"
    )
    assert physical_signature(first) == physical_signature(other)
    other[":init"] = [["On", "bowl", "plate"]]
    assert physical_signature(first) != physical_signature(other)


def test_noise_is_reproducible_and_changes_by_state_and_query():
    noise = matched_noise("pair", 1, 2)
    np.testing.assert_array_equal(noise, matched_noise("pair", 1, 2))
    assert noise.shape == (10, 32)
    assert not np.array_equal(noise, matched_noise("pair", 2, 2))
    assert not np.array_equal(noise, matched_noise("pair", 1, 3))


def test_selector_rejects_incomplete_or_unmatched_and_protects_original_tasks():
    def row(beta, canonical, paired, para):
        return {
            "beta": beta,
            "complete": True,
            "expected_episodes": 80,
            "manifest_sha256": "same",
            "canonical_success_rate": canonical,
            "paired_target_success_rate": paired,
            "paraphrase_success_rate": para,
            "success_rate": (canonical + para) / 2,
        }

    rows = [row(BETAS[0], 0.8, 0.5, 0.4), row(BETAS[1], 0.8, 0.6, 0.5), row(BETAS[2], 0.7, 0.9, 0.9)]
    assert select_beta(rows)["selected_beta"] == math.log(2)
    tied = [row(beta, 0.8, 0.5, 0.4) for beta in BETAS]
    assert select_beta(tied)["selected_beta"] == 0
    rows[2]["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        select_beta(rows)


def test_joint_target_metric_penalizes_one_goal_and_checks_shared_scene():
    rows = [
        {
            "case_id": str(i),
            "pair_id": "p",
            "state_index": 0,
            "manifest_sha256": "m",
            "wording": "canonical" if i % 2 == 0 else "paraphrase",
            "success": i < 2,
            "initial_observation_sha256": "same",
            "wrong_goal_at_end": False,
        }
        for i in range(4)
    ]
    summary = summarize_cases(rows, [str(i) for i in range(4)], "m")
    assert summary["success_rate"] == 0.5
    assert summary["paired_target_success_rate"] == 0
    assert summary["paired_all_prompts_success_rate"] == 0
    rows[-1]["initial_observation_sha256"] = "different"
    with pytest.raises(ValueError, match="identical initial"):
        summarize_cases(rows, [str(i) for i in range(4)], "m")


def test_changed_instruction_uses_changed_success_checker():
    seen = []

    def evaluate(predicate):
        assert predicate[0] == "on", "LIBERO predicate registry is lowercase"
        seen.append(predicate)
        return predicate[-1] == "stove"

    env = SimpleNamespace(env=SimpleNamespace(_eval_predicate=evaluate))
    targets = [{"goal": [["And", ["On", "Bowl", name]]]} for name in ("Plate", "Stove")]
    assert goal_values(env, targets) == [False, True]
    assert seen == [["on", "bowl", "plate"], ["on", "bowl", "stove"]]
    assert targets[0]["goal"][0][1] == ["On", "Bowl", "Plate"]
    targets[1]["goal"] = [["Or", ["On", "bowl", "stove"]]]
    with pytest.raises(ValueError, match="conjunction"):
        goal_values(env, targets)


def test_frozen_manifest_has_disjoint_dev_test_states_and_matching_physics():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "examples/libero_attention/manifest_v1.json").read_text())
    for pair in manifest["pairs"]:
        assert len(pair["dev_state_indices"]) == 10
        assert len(pair["test_state_indices"]) == 20
        assert not set(pair["dev_state_indices"]) & set(pair["test_state_indices"])
        assert pair["targets"][0]["physical_sha256"] == pair["targets"][1]["physical_sha256"]
        assert pair["targets"][0]["goal"] != pair["targets"][1]["goal"]
