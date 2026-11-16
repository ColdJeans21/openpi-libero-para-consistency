# ruff: noqa: RUF001
"""Audit complete residual-v2 results and matched benchmark outcomes, locally."""

import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BASELINE = "teacher_consistency_lam1_30k"
RESIDUAL = "teacher_prefix_residual_r16_30k"
OFF = RESIDUAL + "_residual_off"


def load(path):
    return json.loads(path.read_text())


def exact_mcnemar(fixed, lost):
    n = fixed + lost
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(fixed, lost) + 1)) / (2**n)) if n else 1.0


def matched_rows(run, benchmark):
    if benchmark == "plus":
        path = REPORTS / "libero_plus_language" / run / "seed7/episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        keys = [row["official_id"] for row in rows]
        identity_fields = ("instruction", "seed", "initial_state_index", "base_task")
    else:
        base = REPORTS / "validation" / run / "step_29999/libero_para_seed7"
        rows, keys = [], []
        for eval_id in range(10):
            records = load(base / f"eval{eval_id}.json")["episodes"]
            for row in records:
                rows.append(row)
                keys.append((eval_id, row["bddl_file"], row["initial_state_idx"], row["episode_idx"]))
        identity_fields = ("paraphrased_instruction", "initial_state_idx", "episode_idx")
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate evaluation IDs: {run}/{benchmark}")
    return {
        key: (tuple(row[field] for field in identity_fields), bool(row["success"]))
        for key, row in zip(keys, rows, strict=True)
    }


def paired(first, second, benchmark):
    a, b = matched_rows(first, benchmark), matched_rows(second, benchmark)
    if a.keys() != b.keys() or any(a[key][0] != b[key][0] for key in a):
        raise ValueError("Evaluation cases or instruction/state labels differ")
    fixed = sum(not a[key][1] and b[key][1] for key in a)
    lost = sum(a[key][1] and not b[key][1] for key in a)
    return {
        "first": first,
        "second": second,
        "benchmark": benchmark,
        "n": len(a),
        "fixed": fixed,
        "lost": lost,
        "net_successes": fixed - lost,
        "exact_mcnemar_two_sided": exact_mcnemar(fixed, lost),
    }


def analyze():
    data = {
        "updated_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "checkpoint_step": 29999,
        "train_seed": 42,
        "evaluation_seed": 7,
        "runs": {},
        "paired": [],
    }
    for run in (BASELINE, RESIDUAL, OFF):
        base = REPORTS / "validation" / run / "step_29999"
        values = {
            "goal": load(base / "libero_goal_seed7.json"),
            "para": load(base / "libero_para_seed7/summary.json"),
            "plus": load(REPORTS / "libero_plus_language" / run / "seed7/summary.json"),
        }
        assert values["goal"]["total_episodes"] == values["para"]["total_episodes"] == 100
        assert values["plus"]["complete"]
        assert values["plus"]["total_episodes"] == 410
        data["runs"][run] = values
    for first, second in ((BASELINE, RESIDUAL), (OFF, RESIDUAL)):
        for benchmark in ("para", "plus"):
            result = paired(first, second, benchmark)
            summaries = data["runs"]
            expected = summaries[second][benchmark]["total_successes"] - summaries[first][benchmark]["total_successes"]
            assert result["net_successes"] == expected
            data["paired"].append(result)
    data["learning_curve"] = []
    for step in (10000, 20000, 29999):
        base = REPORTS / "validation" / RESIDUAL / f"step_{step}"
        data["learning_curve"].append(
            {
                "step": step,
                "goal": load(base / "libero_goal_seed7.json")["total_successes"],
                "para": load(base / "libero_para_seed7/summary.json")["total_successes"],
            }
        )
    return data


def write_analysis():
    data = analyze()
    rows = data["runs"]
    scores = ["| 方法 | Goal /100 | Para /100 | Plus Language /410 |", "| --- | ---: | ---: | ---: |"]
    for name, label in ((BASELINE, "原Teacher λ1"), (RESIDUAL, "保留通路残差v2 ON"), (OFF, "同一v2权重残差OFF")):
        values = rows[name]
        cells = [
            f"{values[k]['total_successes']}/{values[k]['total_episodes']} ({100 * values[k]['total_successes'] / values[k]['total_episodes']:.2f}%)"
            for k in ("goal", "para", "plus")
        ]
        scores.append("| " + label + " | " + " | ".join(cells) + " |")
    comparisons = [
        "| 前者 → 后者 | 集合 | 修复 | 退化 | 净成功变化 | 精确McNemar双侧p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in data["paired"]:
        label = "原Teacher → v2 ON" if item["first"] == BASELINE else "v2 OFF → ON"
        comparisons.append(
            f"| {label} | {item['benchmark']} {item['n']} | {item['fixed']} | {item['lost']} | "
            f"{item['net_successes']:+d} | {item['exact_mcnemar_two_sided']:.5g} |"
        )
    plus_base = rows[BASELINE]["plus"]
    plus_v2 = rows[RESIDUAL]["plus"]
    tasks = ["| 基础任务 | 原Teacher成功数 | v2成功数 | 总数 | 净成功变化 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, old in plus_base["per_base_task"].items():
        new = plus_v2["per_base_task"][name]
        assert old["total"] == new["total"]
        tasks.append(
            f"| {name} | {old['successes']} | {new['successes']} | {new['total']} | {new['successes'] - old['successes']:+d} |"
        )
    text = "\n".join(
        [
            "# 2026-11-16：保留语言通路的语义残差v2，最终结果与开关对照",
            "",
            f"本地核验：{data['updated_at']}。用户要求报告仅本地保存，不同步远程。",
            "",
            "## 核心结论",
            "",
            "**残差分支被训练后的模型实际依赖，但没有带来相对原Teacher的稳定净增益。** 保留原PaliGemma语言通路显著缓解v1整套设计的退化，但当前v2仍不适合直接加入主方法。",
            "",
            *scores,
            "",
            "v2相对原Teacher：Goal−3pp、Para−12pp、Plus+2.20pp（仅净增9/410）。Goal86%未达到预先约定的87%门槛；Teacher λ10的Plus57.07%也仍高于v2。不能只截取Plus一列宣称整体改善。",
            "",
            "## 这组架构实验回答了什么",
            "",
            "1. **v1→v2是修复退化的证据。** 同为Teacher λ1，v1为73/19/27.56%，v2为86/45/50.49%；分别+13、+26、+22.93pp。主要实现差别为保留原PaliGemma真实语言前缀，使动作隐状态重新具备原预训练语言条件。当前只有单训练种子，这不证明低秩接口本身有独立增益。",
            "",
            "2. **OFF不是原Teacher。** 同一v2权重关闭分支变为24/9/17.80%，说明联合训练使主路径与残差共同适配。推理时突然去掉其中一项，是对已训练模型的消融；它不等于从头不带残差训练的基线，也不能说明新增分支相对原Teacher增加62pp。",
            "",
            "3. **小输出残差也可能影响闭环成败。** 最后打印窗口step29900：总loss0.0217、canonical FM0.0216、同义一致性约0.0001；残差RMS0.0612、base RMS1.0800，比例约5.67%。这只能描述训练窗口里的幅度。闭环状态转移和成功判据非线性，幅度小不代表能随意删除，也不能据此断言语言更重要或模型只走旁路。",
            "",
            "4. **同义loss下降与行为泛化必须分开。** v2训练一致性已很低，Para仍比原Teacher低12pp，未见指令的成功率不能由训练一致性MSE代替。不同语义能否正确切换目标仍需合法反事实行为评测。",
            "",
            "## 固定案例配对复核",
            "",
            *comparisons,
            "",
            "Plus按official_id匹配，并核验instruction/seed/initial_state/base_task；Para按eval_id/BDDL/initial_state/episode匹配并核验指令。这里是同benchmark案例配对，原标准评测没有显式保证不同模型每次查询使用相同噪声。p值为事后描述、未作多重比较校正；语言样本聚集于10个基础任务和固定状态，不能当作跨任务或跨训练种子结论。",
            "",
            "## Plus逐基础任务变化",
            "",
            *tasks,
            "",
            "整体净增9条可能由不同任务间的收益与退化抵消而来；应该同时看任务分布与未见语言表现，不能只看总体均值。",
            "",
            "## 学习曲线",
            "",
            "| Checkpoint | Goal /100 | Para /100 |",
            "| ---: | ---: | ---: |",
            *[f"| {r['step']} | {r['goal']} | {r['para']} |" for r in data["learning_curve"]],
            "",
            "只用最终29999做跨方法比较；不按不同评测集分别挑选checkpoint。",
            "",
            "## 对下一步的含义",
            "",
            "目前保留原多模态语言通路、训练改写与Teacher作为开发参照；不默认叠加v1/v2残差。下一步先独立检验FM上的Attention干预，再将同一β迁移至Teacher；若无独立收益，就保留负结果，避免靠组合数量替代机制证据。若以后重做残差，应控制其接入位置和训练适配方式，检验能否在不破坏已有canonical映射的情况下获得新增泛化能力，这只是待验证假设。",
            "",
            "Attention在本次核验时没有完成评测：真实权重冒烟通过，但首个episode前在BDDL谓词大小写转换处报错；这不是方法负结果。接口修复与续跑状态见总报告及本地队列日志。",
            "",
            "原始证据：`reports/validation/<run>/step_29999/`、`reports/libero_plus_language/<run>/seed7/`；可重算脚本：`scripts/analyze_semantic_residual_results.py`。",
            "",
        ]
    )
    (REPORTS / "semantic_residual_v2_analysis_20261116.json").write_text(json.dumps(data, indent=2) + "\n")
    (REPORTS / "semantic_residual_v2_analysis_20261116.md").write_text(text)
    print(json.dumps(data["paired"], indent=2))


if __name__ == "__main__":
    write_analysis()
