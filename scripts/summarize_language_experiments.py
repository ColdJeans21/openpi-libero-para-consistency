# ruff: noqa: RUF001
"""Regenerate the research ledger from raw final-checkpoint evaluation summaries."""

import csv
import datetime
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT = "SigLIP全量 / Gemma LoRA / Expert LoRA"
# Run metadata only; evaluation scores must always come from raw JSON files.
RUNS = [
    ("canonical_30k", "原架构 Canonical FM", "FM", 0, DEFAULT, 468.04),
    ("para_aug_lam0_30k", "原架构 双指令FM增强", "Paired FM", 0, DEFAULT, 468.04),
    ("consistency_lam01_30k", "原架构 对称一致性", "Symmetric", 0.1, DEFAULT, 468.04),
    ("consistency_lam1_30k", "原架构 对称一致性", "Symmetric", 1, DEFAULT, 468.04),
    ("consistency_lam10_30k", "原架构 对称一致性", "Symmetric", 10, DEFAULT, 468.04),
    ("teacher_consistency_lam01_30k", "原架构 Teacher", "Teacher", 0.1, DEFAULT, 468.04),
    ("teacher_consistency_lam1_30k", "原架构 Teacher（Para开发参照）", "Teacher", 1, DEFAULT, 468.04),
    ("teacher_consistency_lam10_30k", "原架构 Teacher", "Teacher", 10, DEFAULT, 468.04),
    (
        "teacher_consistency_lam1_expert_full_30k",
        "原架构 Expert完整更新",
        "Teacher",
        1,
        "SigLIP冻结 / Gemma冻结 / Expert全量",
        314.71,
    ),
    (
        "teacher_consistency_lam1_vlm_lora_expert_full_30k",
        "原架构 Gemma LoRA＋Expert完整更新",
        "Teacher",
        1,
        "SigLIP冻结 / Gemma LoRA / Expert全量",
        342.58,
    ),
    (
        "teacher_consistency_lam1_strict_lora_30k",
        "原架构 严格LoRA",
        "Teacher",
        1,
        "SigLIP冻结 / Gemma LoRA / Expert LoRA",
        53.24,
    ),
    ("semantic_interface_r16_fm_30k", "替代语言通路 rank16 v1", "FM", 0, DEFAULT + " / 语义头", 468.58),
    ("semantic_interface_r16_teacher_lam1_30k", "替代语言通路 rank16 v1", "Teacher", 1, DEFAULT + " / 语义头", 468.58),
    ("teacher_prefix_residual_r16_30k", "保留原语言通路＋rank16残差 v2", "Teacher", 1, DEFAULT + " / 语义头", 468.58),
    (
        "teacher_prefix_residual_r16_30k_residual_off",
        "v2同一权重，推理残差关闭",
        "Teacher",
        1,
        "复用v2权重，仅关闭残差",
        468.58,
    ),
    ("attention_fm_selected_v1", "原架构 FM＋Attention干预", "FM / 推理干预", 0, "复用canonical_30k；新增训练参数0", 0),
    (
        "attention_teacher_selected_v1",
        "原架构 Teacher＋同一Attention干预",
        "Teacher / 推理干预",
        1,
        "复用teacher λ1；新增训练参数0",
        0,
    ),
    ("triplet_fm_control_30k", "三元组pilot 原架构", "FM（无差分）", 0, DEFAULT, 468.04),
    ("triplet_language_fixed_30k", "三元组pilot 固定语言头", "FM（无差分）", 0, DEFAULT + " / 独立语言头", 505.788176),
    ("triplet_language_dynamic_30k", "三元组pilot 动态语言头", "FM（无差分）", 0, DEFAULT + " / 动态语言头", 505.806626),
    ("paired_fm_language_fixed_30k", "Goal同义配对 固定语言头", "Paired FM", 0, DEFAULT + " / 独立语言头", 505.788176),
    ("verified_r0_30k", "R0 完整Goal＋验证异义", "Paired FM", 0, DEFAULT, 468.04),
    ("verified_r1_30k", "R1 同数据＋同义Teacher", "Paired FM+Teacher", 1, DEFAULT, 468.04),
    ("verified_r2_30k", "R2 同数据＋异义差分", "Paired FM+Teacher+Diff(0.1)", 1, DEFAULT, 468.04),
]


def param_cell(value):
    return "待精确核对" if value is None else f"{value:.2f}"


def training_data(run):
    if run.startswith("verified_"):
        return "Goal428演示+60训练改写；7:1混合双分支成功验证的初始状态异义对"
    if run.startswith("triplet_"):
        return "两任务族四目标/78条成功anchor rollout；pi05伪标签；替代原Goal演示"
    if run in {"canonical_30k", "attention_fm_selected_v1"}:
        return "Goal10任务428演示/仅canonical；attention复用权重"
    if run == "semantic_interface_r16_fm_30k":
        return "Goal428演示/配对输入但仅canonical FM"
    return "Goal10任务428演示+60训练改写；评测开关组复用权重"


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def attention_status():
    path = ROOT / "logs/attention_v1/STATUS"
    return path.read_text().strip() if path.exists() else "NOT_QUEUED"


def attention_dev_text():
    rows = []
    for label, beta in (("β=0", "b0"), ("β=ln2", "bln2"), ("β=ln4", "bln4")):
        data = read(REPORTS / f"attention/v1/fm_dev_{beta}/summary.json")
        if not data:
            rows.append(f"| {label} | 0/80 | — | — | — | — | — | 未运行 |")
            continue
        complete = data.get("complete") is True and data.get("total_episodes") == data.get("expected_episodes") == 80

        def value(key, summary=data):
            return "—" if summary.get(key) is None else f"{100 * summary[key]:.2f}%"

        rows.append(
            f"| {label} | {data.get('total_episodes', 0)}/80 | {value('success_rate')} | "
            f"{value('canonical_success_rate')} | {value('paraphrase_success_rate')} | "
            f"{value('paired_target_success_rate')} | {value('paired_all_prompts_success_rate')} | "
            f"{'完整' if complete else '不完整，不参与选择'} |"
        )
    return "\n".join(rows)


def metric(path, expected, *, require_complete=False):
    data = read(path)
    if data is None:
        return None
    count, successes = data["total_episodes"], data["total_successes"]
    if count != expected or (require_complete and not data.get("complete")):
        return None  # A partial evaluation is never presented as a completed score.
    if not 0 <= successes <= count:
        raise ValueError(f"Invalid successes in {path}")
    return {"successes": successes, "episodes": count, "rate": successes / count, "source": str(path.relative_to(ROOT))}


def collect():
    rows = []
    selection = read(REPORTS / "attention/v1/selection.json")
    for run, method, objective, weight, scope, params in RUNS:
        base = REPORTS / "validation" / run / "step_29999"
        if run == "consistency_lam01_30k":
            base = REPORTS / "validation/step_29999"  # Historical run, mapped in controlled_experiment_v1.md.
        goal = metric(base / "libero_goal_seed7.json", 100)
        para = metric(base / "libero_para_seed7/summary.json", 100)
        plus = metric(REPORTS / "libero_plus_language" / run / "seed7/summary.json", 410, require_complete=True)
        attention_model = "fm" if run == "attention_fm_selected_v1" else "teacher"
        cf = None
        if run.startswith("attention_"):
            suffix = "selected" if selection and selection["selected_beta"] > 0 else "b0"
            cf = read(REPORTS / f"attention/v1/{attention_model}_test_{suffix}/summary.json")
            if cf and not cf.get("complete"):
                cf = None
        if goal and para:
            status = "完成" if plus else "Goal/Para完成；Plus未测"
        elif run.startswith("attention_"):
            status = "排队（FM先行）"
            current = attention_status()
            if "RUNNING" in current:
                status = "评测中；尚无完整结果"
            elif "BLOCKED_OR_FAILED" in current:
                status = "dev两条件完整；选择/测试未完成"
            if selection:
                status = "评测中" if selection["selected_beta"] > 0 else "dev选择β=0；无非零干预收益"
        elif run.endswith("residual_off"):
            status = "等待v2最终权重"
        else:
            status = "训练/评测中（未出最终结果）"
        if run.startswith("verified_") and not (goal and para and plus):
            status_path = ROOT / "logs/verified_cfd_v2/STATUS"
            stage = status_path.read_text().strip() if status_path.exists() else "待数据验证/冒烟"
            if "TRAINING verified_" in stage and "smoke" not in stage and run not in stage:
                status = "等待队列；本组尚无完整结果"
            else:
                status = "新实验；" + stage
        rows.append(
            {
                "run": run,
                "method": method,
                "objective": objective,
                "lambda": weight,
                "trainable_scope": scope,
                "trainable_params_M": params,
                "training_data": training_data(run),
                "status": status,
                "goal": goal,
                "para": para,
                "plus": plus,
                "cf": cf,
            }
        )
    return rows


def cell(value):
    if value is None:
        return "—"
    return f"{100 * value['rate']:.2f}%（{value['successes']}/{value['episodes']}）"


def source_link(value, label):
    if value is None:
        return f"{label}未测/未完成"
    return f"[{label}]({value['source'].removeprefix('reports/')})"


def write_report():
    REPORTS.mkdir(exist_ok=True)
    rows = collect()
    by_run = {r["run"]: r for r in rows}
    table = []
    out = io.StringIO()
    fields = [
        "run",
        "method",
        "objective",
        "lambda",
        "trainable_scope",
        "trainable_params_M",
        "status",
        "training_data",
    ]
    for name in ("goal", "para", "plus"):
        fields.extend([f"{name}_successes", f"{name}_episodes", f"{name}_rate", f"{name}_source"])
    fields.extend(["cf_paired_target_rate", "cf_all_four_rate"])
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for i, row in enumerate(rows, 1):
        record = {k: row[k] for k in fields if k in row}
        for name in ("goal", "para", "plus"):
            for key in ("successes", "episodes", "rate", "source"):
                record[f"{name}_{key}"] = row[name][key] if row[name] else ""
        for field, key in (
            ("cf_paired_target_rate", "paired_target_success_rate"),
            ("cf_all_four_rate", "paired_all_prompts_success_rate"),
        ):
            record[field] = row["cf"][key] if row["cf"] else ""
        writer.writerow(record)
        cf = (
            (
                f"{100 * row['cf']['paired_target_success_rate']:.2f}% / "
                f"{100 * row['cf']['paired_all_prompts_success_rate']:.2f}%"
            )
            if row["cf"]
            else "—"
        )
        table.append(
            f"| E{i:02d} | {row['method']} | {row['objective']}；λ={row['lambda']} | "
            f"{param_cell(row['trainable_params_M'])} | {cell(row['goal'])} | {cell(row['para'])} | "
            f"{cell(row['plus'])} | — | {cf} | {row['status']} |"
        )

    def delta(first, second, benchmark):
        a, b = by_run[first][benchmark], by_run[second][benchmark]
        return f"{100 * (b['rate'] - a['rate']):+.2f}pp" if a and b else "待评测"

    residual = by_run["teacher_prefix_residual_r16_30k"]
    off = by_run["teacher_prefix_residual_r16_30k_residual_off"]
    if all(residual[name] for name in ("goal", "para", "plus")):
        residual_text = "v2最终结果（Goal/Para/Plus）：" + " / ".join(
            cell(residual[name]) for name in ("goal", "para", "plus")
        )
        residual_text += (
            "；相对原Teacher λ1分别为"
            + " / ".join(
                delta("teacher_consistency_lam1_30k", residual["run"], name) for name in ("goal", "para", "plus")
            )
            + "。"
        )
        if all(off[name] for name in ("goal", "para", "plus")):
            residual_text += (
                "同权重OFF→ON的变化分别为"
                + " / ".join(delta(off["run"], residual["run"], name) for name in ("goal", "para", "plus"))
                + "；分支的推理贡献与整个训练配方的收益仍需分开解释。"
            )
        else:
            residual_text += "残差OFF对照尚未完整，不作分支依赖性结论。"
    else:
        residual_text = "保留原通路v2尚需最终结果，不写作已有效模块。"

    residual_analysis_path = REPORTS / "semantic_residual_v2_analysis_20261116.json"
    residual_analysis = read(residual_analysis_path)
    residual_details = ""
    if residual_analysis:
        residual_details = "\n\n### 残差v2：已完成开关对照后的解释\n\n"
        residual_details += (
            "v2同权重ON/OFF分别为86/24（Goal）、45/9（Para）、207/73（Plus成功数）。"
            "**模型依赖残差分支，不等于新增分支优于原Teacher。** OFF去掉的是与主干联合适配过的路径，"
            "并不是从头不带残差训练的原Teacher；真正方法级对照仍是原Teacher89/57/198。"
            "v2低于Goal≥87%的预定门槛，Para下降12pp，当前不加入主方法。\n\n"
            "v1→v2在Goal/Para/Plus分别+13/+26/+22.93pp，支持保留原多模态语言通路有助于修复替代接口的退化；"
            "尚不支持该残差接口有独立净增益。\n\n"
            "| 对照 | 集合 | 修复 | 退化 | 净成功变化 | 精确McNemar双侧p |\n"
            "| --- | --- | ---: | ---: | ---: | ---: |\n"
        )
        for item in residual_analysis["paired"]:
            label = "原Teacher→v2 ON" if item["first"] == "teacher_consistency_lam1_30k" else "v2 OFF→ON"
            residual_details += (
                f"| {label} | {item['benchmark']} {item['n']} | {item['fixed']} | {item['lost']} | "
                f"{item['net_successes']:+d} | {item['exact_mcnemar_two_sided']:.5g} |\n"
            )
        residual_details += (
            "\n上述是固定benchmark案例配对，标准评测未显式保证跨模型每次查询同噪声；"
            "p值为事后描述，未校正多重比较，不能替代跨任务/训练种子证据。"
            "Plus净增9条须结合较大的任务间收益/退化交换解释，不能直接称为稳定泛化提升。"
            "逐任务变化、学习曲线、loss与残差RMS讨论见[本轮详细分析](semantic_residual_v2_analysis_20261116.md)。\n"
        )

    links = [
        f"- **E{i:02d} `{r['run']}`**：{r['trainable_scope']}；"
        + " / ".join(source_link(r[k], label) for k, label in (("goal", "Goal"), ("para", "Para"), ("plus", "Plus")))
        for i, r in enumerate(rows, 1)
    ]
    selection = read(REPORTS / "attention/v1/selection.json")
    selection_text = (
        f"已锁定β={selection['selected_beta']}；仅由FM dev选择。"
        if selection
        else f"β尚未选择；本地状态：{attention_status()}。完整研究流程尚未结束，不能判断干预有效性。"
    )
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")
    text = r"""# OpenPI 语言条件研究：方法、实验结果与下一步组合依据

更新时间：__NOW__。本文件由 `scripts/summarize_language_experiments.py` 从本地原始JSON重新生成；机器可读版本：[all_language_experiments.csv](all_language_experiments.csv)。最后一节为所有方法的大表，缺失结果保留为空。

__LOCAL_VERIFIED_ANALYSIS__

## 1. 要解决的问题与当前 insight

**同一场景下，控制应对等价语言稳定，对改变目标的关键语义保持敏感。** 当前已得到的证据主要支持“语言增强＋canonical分支的单向Teacher监督改善改写泛化”；尚未验证“同场景换一个关键语义词，行为就正确切换目标”。模型更关注语言的Attention干预将首先在原始FM上检验，再与Teacher组合，分别判断干预效果和组合效果。

目前不能把“图像凌驾于语言”当作已经证明的机制。完整SigLIP训练有收益，可能意味着更好的视觉接地；它并不直接证明降低视觉关注、提高语言Attention就会改善行为。

## 2. 数据、输入输出与公平比较口径

- E01–E14及后续新训练从官方 `pi0_base` 初始化；E15–E17是已有权重上的推理对照。E01–E14使用LIBERO-Goal 10任务、428条演示；增强组使用60条独立训练改写。E18–E20改用两任务族三元组pilot，不能套用428条演示口径；E21回到E02配对数据。R0–R2为新一轮混合数据实验。训练seed=42、global batch=4、30k更新、双卡FSDP、EMA关闭。
- 输入：同一时刻的俯视图＋腕图、机器人状态、指令；训练目标为未来10步动作块。内部动作补齐32维，执行时取前7维。语言配对复用同观测、同专家动作、同噪声和flow时间。
- 全表主实验统一最终checkpoint **29999**；不按不同评测集分别挑最优checkpoint。已有10k/20k曲线保留在原始报告中，不能与最终值混用。
- Goal：10任务×10次，100 episodes，评测seed7。Para：固定100条语言/任务manifest，seed7。Plus：官方 **Language Instructions** 因素、410 cases、每条1次、initial state 0，seed7；不是完整LIBERO-Plus所有因素，也不是410个独立物理任务。
- 训练改写与Para评测manifest独立；新增Attention测试另冻结同物理场景的目标切换manifest。新诊断复用已训练任务，不能声称未见任务泛化。
- 默认所谓“LoRA”实际训练 **完整SigLIP＋Gemma-2B LoRA＋Expert LoRA＋投影层**，约468.04M参数；严格LoRA冻结SigLIP，约53.24M。PaliGemma由SigLIP视觉编码器和Gemma多模态Transformer组成，SigLIP不等于整个VLM。
- 主表为单训练种子探索性结果；多seed按用户要求暂缓。百分数是固定样本上的观测结果，不能替代跨种子、跨状态、跨任务统计证据。缺失“—”不代表0分。

## 3. 已实现的方法：损失与架构分别改变了什么

在同一flow坐标上，`x_t=t·ε+(1−t)·a`、`u=ε−a`，`v_c/v_p`分别代表canonical/改写分支预测。

1. **Canonical FM**：`MSE(v_c,u)`，仅原始指令。
2. **双指令FM增强**：`0.5·[MSE(v_c,u)+MSE(v_p,u)]`。它本身已让两种指令回归同一目标，存在隐含对齐，不能作为“没有语言一致性学习”的绝对空白组。
3. **对称一致性**：双指令FM增强 `+ λ·MSE(v_c,v_p)`；双分支都受到显式一致性梯度。
4. **Stop-gradient Teacher**：`MSE(v_c,u)+λ·MSE(sg(v_c),v_p)`。共享同一个模型，canonical分支对改写分支提供单向监督；不是独立冻结模型或EMA Teacher。`sg`只截断该项的canonical目标梯度，模型仍通过canonical FM与改写分支更新。
5. **参数范围消融**：固定Teacher λ1，对照SigLIP全量/冻结、Gemma冻结/LoRA、Expert LoRA/全量；均包含相应动作投影等附属层。
6. **替代通路 v1**（E12为canonical FM，不是Paired FM；与E13比较同时改变监督方式）：移除PaliGemma真实语言前缀，换中性文本；冻结T5-large的1024维mean-pool特征投影为归一化rank16意图`z(L)`，动作输出改成 `v_base+U·z(L)`；新增速度残差零初始化。该实验同时改变编码器、pooling、注入位置和原通路，不能单独归因于低秩。
7. **保留通路 v2**：保留原PaliGemma真实语言前缀，增加可开关的rank16语义残差；Teacher λ1；训练后同一checkpoint分别开启/关闭残差。开关对照能检验推理是否依赖分支，不能单独区分新增训练引起的主干适配收益。重复的prefix control早期停止，不当成完成的科学对照；主参照复用历史Teacher λ1。
8. **Attention干预 v1（状态见下）**：不训练新权重，原Attention logit增加 `β·1[action query]·1[instruction content key]`，只作用于动作token→有效指令内容token，全部18层/8头；不改视觉/文本prefix查询、不改state查询、不改可见性mask、不改prefix KV。排除padding/BOS/EOS/换行。它乘大softmax前的语言相对赔率`exp(β)`，不是强制最终语言Attention质量达到某个比例。该指标也不等于语言对行为的因果贡献。

## 4. 哪些有效：证据与尚未排除的解释

### 语言增强：当前最清楚的收益来源

Canonical → 双指令FM增强：Goal __AUG_GOAL__、Para __AUG_PARA__、Plus __AUG_PLUS__。因此Teacher相对canonical的大幅提升含有接触训练改写的收益，不能全部归因于新损失。

### Teacher：有方法级收益，但不宣称普遍最优

增强 → Teacher λ1：Goal __TEA_GOAL__、Para __TEA_PARA__、Plus __TEA_PLUS__。同λ1时Teacher相对对称一致性：Goal−1pp、Para+5pp、Plus+1.71pp。Teacher λ1的Para最高（57%），Teacher λ10的Plus最高（57.07%），并不存在跨集合一致的最佳λ。

按事先开发规则，Goal≥87%，再以Para选λ，仍以λ1为开发参照；不得事后根据Plus挑λ10，再把Plus称为未用于选择的外部证据。Teacher与增强/对称组还改变了改写分支是否有直接FM监督，尚未完成“只切换stop-gradient，其余目标完全相同”的消融；共享flow坐标必要性也未通过独立noise/time负对照验证。

### 模块贡献：完整SigLIP有直接对照，Expert LoRA没有普遍优势

- 保持Gemma/Expert LoRA不变，SigLIP冻结 → 全量：Goal+18pp、Para+18pp、Plus+14.63pp。支持当前数据与训练配方下视觉适配有贡献。
- 冻结SigLIP、保持Gemma LoRA，Expert LoRA → 完整更新：Goal+3pp、Para+5pp、Plus+7.56pp。不能称“完整Expert普遍更差”。
- 冻结SigLIP、完整Expert，Gemma冻结 → LoRA：Goal−4pp、Para+1pp、Plus持平，暂不支持语言侧LoRA单独有额外收益。
- 仍缺“完整SigLIP＋Gemma LoRA＋完整Expert”等全因子对照，不能把不匹配的训练范围当单变量消融。显存可行性须依据具体参数范围与实际峰值，不能由上述约468M更新推断约3B全部解冻也能跑。

### 替代语言通路 v1：当前实现退化

同为Teacher λ1，原架构89/57/48.29 → v1为73/19/27.56（Goal/Para/Plus，%）。FM下v1也退化。Teacher将v1的Plus从7.07%改善至27.56%，但未修复架构退化；训练一致性MSE降低不等于闭环成功率提高，也不能证明不同语义已塌缩。

__RESIDUAL_RESULT__ Attention是否有效须结合FM独立对照、锁定β后的测试及标准评测判断；未完成的条件不作收益结论。

__RESIDUAL_DETAILS__

### 固定样本配对证据（事后描述，不等于多seed结论）

原增强 → Teacher λ1，在Plus410中修复47条、退化22条，净+25；精确McNemar双侧p≈0.00354。Para100修复17、退化10，p≈0.248。Teacher λ1 → λ10在Plus净+36，但这是外部集合差异，不改既定开发选择。样本聚集于10个基础任务与固定状态，未作多重比较校正，不能把这些p值解释成跨任务/跨训练重复的有效性保证。逐条核验与更多讨论见[2026-11-14报告](semantic_architecture_loss_comparison_20261114.md)。

## 5. FM-first Attention实验：计划与真实执行状态

**__SELECTION__**

真实权重冒烟已通过：β0精确恢复动作、参数保持不变、非零β输出有限且有变化（最大动作差约0.00354）。2026-11-16首次仿真在第一个episode前因BDDL原文`On`与LIBERO小写谓词注册表不匹配而退出，未产出完整episode；这不是方法负结果。已对齐LIBERO解析器的小写规则并增加回归检查，后续续跑仍使用同一个冻结manifest、同一β网格和选择规则。原错误日志保留。GPU冒烟通过不代表闭环评测已完成。

| FM开发条件 | 已完成episode | 总成功率 | canonical | paraphrase | 双canonical目标均成功 | 四条指令均成功 | 完整性 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
__ATTENTION_DEV_TABLE__

β=0与β=ln2均已完成80/80；β=ln2相对β=0仅表现为总成功率+1.25pp、paraphrase+2.50pp、四条指令全成功+5pp，双canonical目标成功率持平，且出现1.25%的错误目标结束率。**这些只是同一个FM开发集上的选择数据，不是锁定β后的测试证据。** β=ln4在1/80后因websocket keepalive超时中断，所以不能完成β选择，也不能把β=ln2写成最终最佳。连接配置已改为仅对本地长时反事实评测禁用keepalive；冻结manifest、模型权重和选择规则均未改变。

执行顺序：等当前v2训练、10k/20k/29999 Goal/Para、最终Plus，以及同权重残差OFF的Goal/Para/Plus都完成后，取得同一个GPU队列锁；检查显卡空闲，再进行真实checkpoint冒烟及以下评测。

1. A/B先只用原Canonical FM最终权重：β∈{0,ln2,ln4}，同一数据、同一权重。开发集80 episodes/β，合计240。
2. 固定两个合法“同场景不同目标”任务对：碗→盘/炉灶，酒瓶→架子/柜顶。BDDL除目标、语言、interest字段外物理定义一致；共享XML、fixture位置、仿真状态；对初始图像/本体状态/模型指纹做严格相等检查。成功判据随请求目标变化，另记录错误目标完成率；若不一致，停止并报告，绝不当成换词测试成功。
3. 每个状态4条指令：目标A canonical/未参与训练改写，目标B canonical/未参与训练改写。每对10开发状态、20不重叠测试状态，冻结SHA256 manifest；测试160 episodes/条件。改写按目标与物理定义筛选、用固定哈希选择，不读取成功率。
4. 跨指令/β/模型共享同一`pair,state,query-index`派生的动作采样噪声；避免初始噪声差异冒充语言影响。主要行为指标：两个canonical目标都成功的状态比例、四条指令全成功的状态比例；不是只测动作距离增大。
5. 只在FM开发集选β：canonical成功率不低于β0超过2pp；依次最大化双目标成功率、改写成功率、总成功率，平局选较小β。选定后锁死，禁止用Teacher或Plus重新选。
6. 测试A=FM β0、B=FM β*、C=Teacher λ1 β0、D=Teacher λ1 β*。若β*=0，记录非零干预不受支持并跳过重复条件。若β*>0，为B/D补Goal100、Para100、Plus410；A/C原标准结果复用，新的反事实A/C仍需跑。
7. GPU冒烟先验证真实FM权重β0输出精确恢复、参数不变、非零干预输出有限且实际有变化。CPU单元测试不代替这一步。新评测独立W&B run，不覆盖已有训练指标。

队列入口：`scripts/queue_attention_after_residual.sh`；执行器：`scripts/run_attention_study.sh`；冻结数据：`examples/libero_attention/manifest_v1.json`。本轮是已有FM/Teacher权重上的推理干预，不重复30k训练、不新增checkpoint；复用原权重位置并记录路径，新增报告位于`reports/attention/v1/`。按用户最新要求，报告仅本地保存；`logs/attention_v1/AUTO_PUBLISH_DISABLED`与队列默认开关禁用自动提交/推送。

## 6. 后续组合应该依据什么

目前优先保留原多模态语言通路、训练改写增强，以及Teacher λ1开发参照；SigLIP适配是有收益但较昂贵的候选。v1和v2均不默认加入主方法：v2虽比v1恢复明显，但相对原Teacher仍有Goal/Para退化。Attention没有完整行为结果，暂不作有效性判断。

接下来先完成FM独立Attention对照，再判断是否与Teacher组合；只有新增模块单独有收益，才进入同预算组合比较。残差v2暂不作为应当叠加的模块。检查收益是否重复、是否保留原任务性能与关键语义切换。先证明效果与机制，再决定组合数量；增加实验数量本身不等于论文贡献。固定Attention重加权应作为有前人基础的诊断/基线，创新需落在验证出的机制与进一步必要的设计，而非简单命名。

尚未完成：合法不同语义任务的行为切换证据、负样本对比损失、独立noise/time消融、同目标下仅切换stop-gradient的消融、多训练seed、完整解冻全模型。不要将排队/讨论内容写成已完成成果。

## 7. 证据索引与历史项目的比较边界

__LINKS__

- 冻结训练设置与标记：`examples/libero_para/runs/`；历史参数范围和逐条配对核验见2026-11-14报告。
- 历史π0.5/DP/ACT结果来自[libero_benchmark_final.md](libero_benchmark_final.md)。它们的训练范围、分辨率、数据量、模型与评测episode数量均不同，下面放入独立“历史”分组，不能与π0受控消融作因果对比。π0.5 LoRA在Goal的97.2%曾被口头混用；可核实的97.2%属于官方π0.5检查点评估，LoRA Goal保留未测。
- 历史DP/ACT的Spatial与Goal来自分别训练的suite策略，不是同一个跨suite模型；目标不可辨识与性能差距有关，但跨架构差异不能只归因于语言。
- 大表反事实一列为“两个canonical目标均成功 / 四条指令均成功”的**状态级**比例，分母为40个测试状态；与160 episodes总成功率是不同指标。β0若胜出，所列为复用的baseline反事实结果，非干预收益。


## 8. 全方法评估结果大表（主实验最终29999；结果截至上述更新时间）

主实验训练范围详见E编号证据索引；所有数值均为本地实测。Plus列仅语言因素。M=百万可训练参数；Attention行M=0表示**新增训练参数**为0，并非模型参数总量为0。历史行不进入主实验排序。“—”为未测或尚未完成。

| 编号 | 方法/架构 | 训练目标 | 可训练M | Goal | Para | Plus Language | Spatial | 反事实双目标 / 全四条 | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
__TABLE__
| H01 | π0.5 官方微调检查点（评估复现） | 历史项目；不同训练配方 | — | 97.20%（486/500） | — | — | 98.80%（494/500） | — | 历史完成 |
| H02 | π0.5 LoRA | 历史项目；40任务 | — | — | — | — | 84.60%（423/500） | — | 历史Spatial完成 |
| H03 | Diffusion Policy | 历史；各suite分别训练 | 267 | 8.00%（40/500） | — | — | 84.20%（421/500） | — | 历史完成 |
| H04 | ACT | 历史；各suite分别训练 | 52 | 9.40%（47/500） | — | — | 78.00%（390/500） | — | 历史完成 |
"""
    replacements = {
        "__NOW__": now,
        "__LOCAL_VERIFIED_ANALYSIS__": (
            (REPORTS / "verified_cfd_analysis_20261125.md").read_text()
            if (REPORTS / "verified_cfd_analysis_20261125.md").exists() else ""
        ),
        "__LINKS__": "\n".join(links),
        "__TABLE__": "\n".join(table),
        "__SELECTION__": selection_text,
        "__RESIDUAL_RESULT__": residual_text,
        "__RESIDUAL_DETAILS__": residual_details,
        "__ATTENTION_DEV_TABLE__": attention_dev_text(),
    }
    for key, benchmark in (("GOAL", "goal"), ("PARA", "para"), ("PLUS", "plus")):
        replacements[f"__AUG_{key}__"] = delta("canonical_30k", "para_aug_lam0_30k", benchmark)
        replacements[f"__TEA_{key}__"] = delta("para_aug_lam0_30k", "teacher_consistency_lam1_30k", benchmark)
    for key, value in replacements.items():
        text = text.replace(key, value)
    (REPORTS / "ALL_EXPERIMENTS_METHODS_AND_RESULTS.md").write_text(text.lstrip(), encoding="utf-8")
    (REPORTS / "all_language_experiments.csv").write_text(out.getvalue(), encoding="utf-8")
    print(f"Generated ledger: {len(rows)} controlled/queued methods plus four historical references")


if __name__ == "__main__":
    write_report()
