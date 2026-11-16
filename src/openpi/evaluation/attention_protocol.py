"""Frozen protocol for FM-first instruction-attention interventions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

BETAS = (0.0, math.log(2), math.log(4))
PAIRS = (
    ("bowl_plate_stove", "put_the_bowl_on_the_plate", "put_the_bowl_on_the_stove"),
    ("bottle_rack_cabinet", "put_the_wine_bottle_on_the_rack", "put_the_wine_bottle_on_top_of_the_cabinet"),
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bddl_sections(text: str) -> dict:
    """Parse the repository's simple BDDL without importing a simulator."""
    tokens = re.findall(r'\(|\)|"[^"\n]*"|[^\s()]+', re.sub(r";[^\n]*", "", text))
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced BDDL")
            stack.pop()
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError("Token outside BDDL expression")
    if stack or len(roots) != 1 or roots[0][0] != "define":
        raise ValueError("Expected one complete BDDL definition")
    return {node[0]: node[1:] for node in roots[0][1:] if isinstance(node, list)}


def physical_signature(sections: dict) -> str:
    return digest({k: v for k, v in sections.items() if k not in {":language", ":goal", ":obj_of_interest"}})


def matched_noise(pair_id: str, state_index: int, query_index: int, *, seed=7, horizon=10, action_dim=32):
    """Deliberately exclude model, beta, target and wording from the random key."""
    key = digest(["attention-v1", seed, pair_id, state_index, query_index])
    return np.random.default_rng(int(key[:16], 16)).standard_normal((horizon, action_dim)).astype(np.float32)


def array_digest(value) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


def goal_values(env, targets):
    """Evaluate the requested goal ASTs, not an environment's old done flag."""
    values = []
    for target in targets:
        root = target["goal"]
        if len(root) != 1 or root[0][0].lower() != "and":
            raise ValueError("This frozen protocol requires an explicit conjunction goal")
        # LIBERO's normal BDDL parser lowercases every token before dispatch.
        # The frozen manifest preserves file spelling (e.g. "On"). Apply the
        # same normalization here without changing the frozen goal/source hash.
        predicates = [[str(token).lower() for token in predicate] for predicate in root[0][1:]]
        values.append(all(env.env._eval_predicate(predicate) for predicate in predicates))  # noqa: SLF001
    return values


def summarize_cases(records: list[dict], expected_ids: list[str], manifest_sha: str) -> dict:
    ids = [r["case_id"] for r in records]
    if len(set(ids)) != len(ids) or set(ids) - set(expected_ids):
        raise ValueError("Duplicate or unexpected counterfactual case")
    if any(r["manifest_sha256"] != manifest_sha for r in records):
        raise ValueError("Counterfactual manifest mismatch")
    groups = {}
    for r in records:
        key = (r["pair_id"], r["state_index"])
        groups.setdefault(key, []).append(r)
    for group in groups.values():
        if len({r["initial_observation_sha256"] for r in group}) != 1:
            raise ValueError("Instructions were not evaluated on identical initial observations")
    complete_groups = [g for g in groups.values() if len(g) == 4]
    canonical = [r for r in records if r["wording"] == "canonical"]
    paraphrase = [r for r in records if r["wording"] == "paraphrase"]

    def rate(rows):
        return sum(bool(r["success"]) for r in rows) / len(rows) if rows else None

    return {
        "complete": set(ids) == set(expected_ids),
        "total_episodes": len(records),
        "expected_episodes": len(expected_ids),
        "total_successes": sum(bool(r["success"]) for r in records),
        "success_rate": rate(records),
        "canonical_success_rate": rate(canonical),
        "paraphrase_success_rate": rate(paraphrase),
        "paired_target_success_rate": (
            sum(all(r["success"] for r in g if r["wording"] == "canonical") for g in complete_groups)
            / len(complete_groups)
            if complete_groups
            else None
        ),
        "paired_all_prompts_success_rate": (
            sum(all(r["success"] for r in g) for g in complete_groups) / len(complete_groups)
            if complete_groups
            else None
        ),
        "wrong_goal_end_rate": (sum(bool(r["wrong_goal_at_end"]) for r in records) / len(records) if records else None),
        "manifest_sha256": manifest_sha,
    }


def select_beta(summaries: list[dict]) -> dict:
    """Choose once on FM development cases; external benchmarks never select beta."""
    if len(summaries) != 3 or any(not s["complete"] for s in summaries):
        raise ValueError("Need three complete FM development evaluations")
    if {s["beta"] for s in summaries} != set(BETAS):
        raise ValueError("Unexpected beta grid")
    if len({(s["manifest_sha256"], s["expected_episodes"]) for s in summaries}) != 1:
        raise ValueError("Development evaluations are not matched")
    base = next(s for s in summaries if s["beta"] == 0)
    eligible = [s for s in summaries if s["canonical_success_rate"] >= base["canonical_success_rate"] - 0.02]
    best = max(
        eligible,
        key=lambda s: (s["paired_target_success_rate"], s["paraphrase_success_rate"], s["success_rate"], -s["beta"]),
    )
    return {
        "selected_beta": best["beta"],
        "development_model": "fm",
        "manifest_sha256": best["manifest_sha256"],
        "rule": "canonical >= baseline-2pp; maximize paired target, then paraphrase, then overall; smaller beta on ties",
        "attention_supported_on_dev": best["beta"] > 0,
        "baseline": base,
        "selected": best,
    }
