# OpenPI 语言条件研究：方法、实验结果与下一步组合依据

更新时间：2026-11-15T17:57:49+08:00。本文件由 `scripts/summarize_language_experiments.py` 从本地原始JSON重新生成；机器可读版本：[all_language_experiments.csv](all_language_experiments.csv)。最后一节为所有方法的大表，缺失结果保留为空。

## 1. 要解决的问题与当前 insight

**同一场景下，控制应对等价语言稳定，对改变目标的关键语义保持敏感。** 当前已得到的证据主要支持“语言增强＋canonical分支的单向Teacher监督改善改写泛化”；尚未验证“同场景换一个关键语义词，行为就正确切换目标”。模型更关注语言的Attention干预将首先在原始FM上检验，再与Teacher组合，分别判断干预效果和组合效果。

目前不能把“图像凌驾于语言”当作已经证明的机制。完整SigLIP训练有收益，可能意味着更好的视觉接地；它并不直接证明降低视觉关注、提高语言Attention就会改善行为。

## 2. 数据、输入输出与公平比较口径

- 主实验均从官方 `pi0_base` 初始化；LIBERO-Goal 10任务、428条演示；60条独立训练改写。训练seed=42、global batch=4、30k更新、双卡FSDP、EMA关闭。
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
6. **替代通路 v1**：移除PaliGemma真实语言前缀，换中性文本；冻结T5-large的1024维mean-pool特征投影为归一化rank16意图`z(L)`，动作输出改成 `v_base+U·z(L)`；新增速度残差零初始化。该实验同时改变编码器、pooling、注入位置和原通路，不能单独归因于低秩。
7. **保留通路 v2（当前实验）**：保留原PaliGemma真实语言前缀，增加可开关的rank16语义残差；Teacher λ1；训练后同一checkpoint分别开启/关闭残差。开关对照能检验推理是否依赖分支，不能单独区分新增训练引起的主干适配收益。重复的prefix control早期停止，不当成完成的科学对照；主参照复用历史Teacher λ1。
8. **Attention干预 v1（排队）**：不训练新权重，原Attention logit增加 `β·1[action query]·1[instruction content key]`，只作用于动作token→有效指令内容token，全部18层/8头；不改视觉/文本prefix查询、不改state查询、不改可见性mask、不改prefix KV。排除padding/BOS/EOS/换行。它乘大softmax前的语言相对赔率`exp(β)`，不是强制最终语言Attention质量达到某个比例。该指标也不等于语言对行为的因果贡献。

## 4. 哪些有效：证据与尚未排除的解释

### 语言增强：当前最清楚的收益来源

Canonical → 双指令FM增强：Goal +5.00pp、Para +29.00pp、Plus +25.12pp。因此Teacher相对canonical的大幅提升含有接触训练改写的收益，不能全部归因于新损失。

### Teacher：有方法级收益，但不宣称普遍最优

增强 → Teacher λ1：Goal +0.00pp、Para +7.00pp、Plus +6.10pp。同λ1时Teacher相对对称一致性：Goal−1pp、Para+5pp、Plus+1.71pp。Teacher λ1的Para最高（57%），Teacher λ10的Plus最高（57.07%），并不存在跨集合一致的最佳λ。

按事先开发规则，Goal≥87%，再以Para选λ，仍以λ1为开发参照；不得事后根据Plus挑λ10，再把Plus称为未用于选择的外部证据。Teacher与增强/对称组还改变了改写分支是否有直接FM监督，尚未完成“只切换stop-gradient，其余目标完全相同”的消融；共享flow坐标必要性也未通过独立noise/time负对照验证。

### 模块贡献：完整SigLIP有直接对照，Expert LoRA没有普遍优势

- 保持Gemma/Expert LoRA不变，SigLIP冻结 → 全量：Goal+18pp、Para+18pp、Plus+14.63pp。支持当前数据与训练配方下视觉适配有贡献。
- 冻结SigLIP、保持Gemma LoRA，Expert LoRA → 完整更新：Goal+3pp、Para+5pp、Plus+7.56pp。不能称“完整Expert普遍更差”。
- 冻结SigLIP、完整Expert，Gemma冻结 → LoRA：Goal−4pp、Para+1pp、Plus持平，暂不支持语言侧LoRA单独有额外收益。
- 仍缺“完整SigLIP＋Gemma LoRA＋完整Expert”等全因子对照，不能把不匹配的训练范围当单变量消融。显存可行性须依据具体参数范围与实际峰值，不能由上述约468M更新推断约3B全部解冻也能跑。

### 替代语言通路 v1：当前实现退化

同为Teacher λ1，原架构89/57/48.29 → v1为73/19/27.56（Goal/Para/Plus，%）。FM下v1也退化。Teacher将v1的Plus从7.07%改善至27.56%，但未修复架构退化；训练一致性MSE降低不等于闭环成功率提高，也不能证明不同语义已塌缩。

保留原通路v2尚需最终结果，不写作已有效模块。 Attention是否有效须结合FM独立对照、锁定β后的测试及标准评测判断；未完成的条件不作收益结论。

### 固定样本配对证据（事后描述，不等于多seed结论）

原增强 → Teacher λ1，在Plus410中修复47条、退化22条，净+25；精确McNemar双侧p≈0.00354。Para100修复17、退化10，p≈0.248。Teacher λ1 → λ10在Plus净+36，但这是外部集合差异，不改既定开发选择。样本聚集于10个基础任务与固定状态，未作多重比较校正，不能把这些p值解释成跨任务/跨训练重复的有效性保证。逐条核验与更多讨论见[2026-11-14报告](semantic_architecture_loss_comparison_20261114.md)。

## 5. 排队的FM-first Attention实验

**β尚未选择；等待当前残差实验及其所有评测完成。**

执行顺序：等当前v2训练、10k/20k/29999 Goal/Para、最终Plus，以及同权重残差OFF的Goal/Para/Plus都完成后，取得同一个GPU队列锁；检查显卡空闲，再进行真实checkpoint冒烟及以下评测。

1. A/B先只用原Canonical FM最终权重：β∈{0,ln2,ln4}，同一数据、同一权重。开发集80 episodes/β，合计240。
2. 固定两个合法“同场景不同目标”任务对：碗→盘/炉灶，酒瓶→架子/柜顶。BDDL除目标、语言、interest字段外物理定义一致；共享XML、fixture位置、仿真状态；对初始图像/本体状态/模型指纹做严格相等检查。成功判据随请求目标变化，另记录错误目标完成率；若不一致，停止并报告，绝不当成换词测试成功。
3. 每个状态4条指令：目标A canonical/未参与训练改写，目标B canonical/未参与训练改写。每对10开发状态、20不重叠测试状态，冻结SHA256 manifest；测试160 episodes/条件。改写按目标与物理定义筛选、用固定哈希选择，不读取成功率。
4. 跨指令/β/模型共享同一`pair,state,query-index`派生的动作采样噪声；避免初始噪声差异冒充语言影响。主要行为指标：两个canonical目标都成功的状态比例、四条指令全成功的状态比例；不是只测动作距离增大。
5. 只在FM开发集选β：canonical成功率不低于β0超过2pp；依次最大化双目标成功率、改写成功率、总成功率，平局选较小β。选定后锁死，禁止用Teacher或Plus重新选。
6. 测试A=FM β0、B=FM β*、C=Teacher λ1 β0、D=Teacher λ1 β*。若β*=0，记录非零干预不受支持并跳过重复条件。若β*>0，为B/D补Goal100、Para100、Plus410；A/C原标准结果复用，新的反事实A/C仍需跑。
7. GPU冒烟先验证真实FM权重β0输出精确恢复、参数不变、非零干预输出有限且实际有变化。CPU单元测试不代替这一步。新评测独立W&B run，不覆盖已有训练指标。

队列入口：`scripts/queue_attention_after_residual.sh`；执行器：`scripts/run_attention_study.sh`；冻结数据：`examples/libero_attention/manifest_v1.json`。本轮是已有FM/Teacher权重上的推理干预，不重复30k训练、不新增checkpoint；复用原权重位置并记录路径，新增报告位于`reports/attention/v1/`。

## 6. 后续组合应该依据什么

目前优先保留原多模态语言通路、训练改写增强，以及Teacher λ1开发参照；SigLIP适配是有收益但较昂贵的候选。暂不把v1替代接口加入主方法，也不把尚无结果的v2/Attention视作有效。

待v2与Attention分别有证据，再比较“Teacher基础”“Teacher＋有效架构/Attention”“Teacher＋两者”的同预算结果，并检查收益是否重复、是否保留原任务性能与关键语义切换。先证明效果与机制，再决定组合数量；增加实验数量本身不等于论文贡献。固定Attention重加权应作为有前人基础的诊断/基线，创新需落在验证出的机制与进一步必要的设计，而非简单命名。

尚未完成：合法不同语义任务的行为切换证据、负样本对比损失、独立noise/time消融、同目标下仅切换stop-gradient的消融、多训练seed、完整解冻全模型。不要将排队/讨论内容写成已完成成果。

## 7. 证据索引与历史项目的比较边界

- **E01 `canonical_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA；[Goal](validation/canonical_30k/step_29999/libero_goal_seed7.json) / [Para](validation/canonical_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/canonical_30k/seed7/summary.json)
- **E02 `para_aug_lam0_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA；[Goal](validation/para_aug_lam0_30k/step_29999/libero_goal_seed7.json) / [Para](validation/para_aug_lam0_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/para_aug_lam0_30k/seed7/summary.json)
- **E03 `consistency_lam01_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA；[Goal](validation/step_29999/libero_goal_seed7.json) / [Para](validation/step_29999/libero_para_seed7/summary.json) / Plus未测/未完成
- **E04 `consistency_lam1_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA；[Goal](validation/consistency_lam1_30k/step_29999/libero_goal_seed7.json) / [Para](validation/consistency_lam1_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/consistency_lam1_30k/seed7/summary.json)
- **E05 `consistency_lam10_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA；[Goal](validation/consistency_lam10_30k/step_29999/libero_goal_seed7.json) / [Para](validation/consistency_lam10_30k/step_29999/libero_para_seed7/summary.json) / Plus未测/未完成
- **E06 `teacher_consistency_lam01_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA；[Goal](validation/teacher_consistency_lam01_30k/step_29999/libero_goal_seed7.json) / [Para](validation/teacher_consistency_lam01_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/teacher_consistency_lam01_30k/seed7/summary.json)
- **E07 `teacher_consistency_lam1_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA；[Goal](validation/teacher_consistency_lam1_30k/step_29999/libero_goal_seed7.json) / [Para](validation/teacher_consistency_lam1_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/teacher_consistency_lam1_30k/seed7/summary.json)
- **E08 `teacher_consistency_lam10_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA；[Goal](validation/teacher_consistency_lam10_30k/step_29999/libero_goal_seed7.json) / [Para](validation/teacher_consistency_lam10_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/teacher_consistency_lam10_30k/seed7/summary.json)
- **E09 `teacher_consistency_lam1_expert_full_30k`**：SigLIP冻结 / Gemma冻结 / Expert全量；[Goal](validation/teacher_consistency_lam1_expert_full_30k/step_29999/libero_goal_seed7.json) / [Para](validation/teacher_consistency_lam1_expert_full_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/teacher_consistency_lam1_expert_full_30k/seed7/summary.json)
- **E10 `teacher_consistency_lam1_vlm_lora_expert_full_30k`**：SigLIP冻结 / Gemma LoRA / Expert全量；[Goal](validation/teacher_consistency_lam1_vlm_lora_expert_full_30k/step_29999/libero_goal_seed7.json) / [Para](validation/teacher_consistency_lam1_vlm_lora_expert_full_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/teacher_consistency_lam1_vlm_lora_expert_full_30k/seed7/summary.json)
- **E11 `teacher_consistency_lam1_strict_lora_30k`**：SigLIP冻结 / Gemma LoRA / Expert LoRA；[Goal](validation/teacher_consistency_lam1_strict_lora_30k/step_29999/libero_goal_seed7.json) / [Para](validation/teacher_consistency_lam1_strict_lora_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/teacher_consistency_lam1_strict_lora_30k/seed7/summary.json)
- **E12 `semantic_interface_r16_fm_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA / 语义头；[Goal](validation/semantic_interface_r16_fm_30k/step_29999/libero_goal_seed7.json) / [Para](validation/semantic_interface_r16_fm_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/semantic_interface_r16_fm_30k/seed7/summary.json)
- **E13 `semantic_interface_r16_teacher_lam1_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA / 语义头；[Goal](validation/semantic_interface_r16_teacher_lam1_30k/step_29999/libero_goal_seed7.json) / [Para](validation/semantic_interface_r16_teacher_lam1_30k/step_29999/libero_para_seed7/summary.json) / [Plus](libero_plus_language/semantic_interface_r16_teacher_lam1_30k/seed7/summary.json)
- **E14 `teacher_prefix_residual_r16_30k`**：SigLIP全量 / Gemma LoRA / Expert LoRA / 语义头；Goal未测/未完成 / Para未测/未完成 / Plus未测/未完成
- **E15 `teacher_prefix_residual_r16_30k_residual_off`**：复用v2权重，仅关闭残差；Goal未测/未完成 / Para未测/未完成 / Plus未测/未完成
- **E16 `attention_fm_selected_v1`**：复用canonical_30k；新增训练参数0；Goal未测/未完成 / Para未测/未完成 / Plus未测/未完成
- **E17 `attention_teacher_selected_v1`**：复用teacher λ1；新增训练参数0；Goal未测/未完成 / Para未测/未完成 / Plus未测/未完成

- 冻结训练设置与标记：`examples/libero_para/runs/`；历史参数范围和逐条配对核验见2026-11-14报告。
- 历史π0.5/DP/ACT结果来自[libero_benchmark_final.md](libero_benchmark_final.md)。它们的训练范围、分辨率、数据量、模型与评测episode数量均不同，下面放入独立“历史”分组，不能与π0受控消融作因果对比。π0.5 LoRA在Goal的97.2%曾被口头混用；可核实的97.2%属于官方π0.5检查点评估，LoRA Goal保留未测。
- 历史DP/ACT的Spatial与Goal来自分别训练的suite策略，不是同一个跨suite模型；目标不可辨识与性能差距有关，但跨架构差异不能只归因于语言。
- 大表反事实一列为“两个canonical目标均成功 / 四条指令均成功”的**状态级**比例，分母为40个测试状态；与160 episodes总成功率是不同指标。β0若胜出，所列为复用的baseline反事实结果，非干预收益。

## 8. 全方法评估结果大表（主实验最终29999；结果截至上述更新时间）

主实验训练范围详见E编号证据索引；所有数值均为本地实测。Plus列仅语言因素。M=百万可训练参数；Attention行M=0表示**新增训练参数**为0，并非模型参数总量为0。历史行不进入主实验排序。“—”为未测或尚未完成。

| 编号 | 方法/架构 | 训练目标 | 可训练M | Goal | Para | Plus Language | Spatial | 反事实双目标 / 全四条 | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E01 | 原架构 Canonical FM | FM；λ=0 | 468.04 | 84.00%（84/100） | 21.00%（21/100） | 17.07%（70/410） | — | — | 完成 |
| E02 | 原架构 双指令FM增强 | Paired FM；λ=0 | 468.04 | 89.00%（89/100） | 50.00%（50/100） | 42.20%（173/410） | — | — | 完成 |
| E03 | 原架构 对称一致性 | Symmetric；λ=0.1 | 468.04 | 84.00%（84/100） | 43.00%（43/100） | — | — | — | Goal/Para完成；Plus未测 |
| E04 | 原架构 对称一致性 | Symmetric；λ=1 | 468.04 | 90.00%（90/100） | 52.00%（52/100） | 46.59%（191/410） | — | — | 完成 |
| E05 | 原架构 对称一致性 | Symmetric；λ=10 | 468.04 | 79.00%（79/100） | 41.00%（41/100） | — | — | — | Goal/Para完成；Plus未测 |
| E06 | 原架构 Teacher | Teacher；λ=0.1 | 468.04 | 90.00%（90/100） | 39.00%（39/100） | 38.54%（158/410） | — | — | 完成 |
| E07 | 原架构 Teacher（Para开发参照） | Teacher；λ=1 | 468.04 | 89.00%（89/100） | 57.00%（57/100） | 48.29%（198/410） | — | — | 完成 |
| E08 | 原架构 Teacher | Teacher；λ=10 | 468.04 | 88.00%（88/100） | 52.00%（52/100） | 57.07%（234/410） | — | — | 完成 |
| E09 | 原架构 Expert完整更新 | Teacher；λ=1 | 314.71 | 78.00%（78/100） | 43.00%（43/100） | 41.22%（169/410） | — | — | 完成 |
| E10 | 原架构 Gemma LoRA＋Expert完整更新 | Teacher；λ=1 | 342.58 | 74.00%（74/100） | 44.00%（44/100） | 41.22%（169/410） | — | — | 完成 |
| E11 | 原架构 严格LoRA | Teacher；λ=1 | 53.24 | 71.00%（71/100） | 39.00%（39/100） | 33.66%（138/410） | — | — | 完成 |
| E12 | 替代语言通路 rank16 v1 | FM；λ=0 | 468.58 | 78.00%（78/100） | 16.00%（16/100） | 7.07%（29/410） | — | — | 完成 |
| E13 | 替代语言通路 rank16 v1 | Teacher；λ=1 | 468.58 | 73.00%（73/100） | 19.00%（19/100） | 27.56%（113/410） | — | — | 完成 |
| E14 | 保留原语言通路＋rank16残差 v2 | Teacher；λ=1 | 468.58 | — | — | — | — | — | 训练/评测中（未出最终结果） |
| E15 | v2同一权重，推理残差关闭 | Teacher；λ=1 | 468.58 | — | — | — | — | — | 等待v2最终权重 |
| E16 | 原架构 FM＋Attention干预 | FM / 推理干预；λ=0 | 0.00 | — | — | — | — | — | 排队（FM先行） |
| E17 | 原架构 Teacher＋同一Attention干预 | Teacher / 推理干预；λ=1 | 0.00 | — | — | — | — | — | 排队（FM先行） |
| H01 | π0.5 官方微调检查点（评估复现） | 历史项目；不同训练配方 | — | 97.20%（486/500） | — | — | 98.80%（494/500） | — | 历史完成 |
| H02 | π0.5 LoRA | 历史项目；40任务 | — | — | — | — | 84.60%（423/500） | — | 历史Spatial完成 |
| H03 | Diffusion Policy | 历史；各suite分别训练 | 267 | 8.00%（40/500） | — | — | 84.20%（421/500） | — | 历史完成 |
| H04 | ACT | 历史；各suite分别训练 | 52 | 9.40%（47/500） | — | — | 78.00%（390/500） | — | 历史完成 |
