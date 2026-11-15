# 2026-11-14：OpenPI 语义一致性损失与架构实验汇总

## 结论

1. 原架构上的语言增强是最大、最清楚的收益来源：Para 21%→50%，Plus 17.07%→42.20%。
2. Stop-gradient Teacher 在当前单种子实验中提供额外收益，尤其在 Plus 上；但不是所有权重、所有评测集都提升，也尚未证明一定优于对称一致性。
3. 当前 rank-16 语义接口实现没有提升原架构效果：同为 Teacher λ=1，Goal/Para/Plus 从 89/57/48.29 降至 73/19/27.56。
4. Teacher 在新接口内部仍改善 Plus（7.07%→27.56%），但 Para 仅 +3pp、Goal -5pp，无法弥补架构退化。
5. 模块消融最清楚地支持完整 SigLIP 适配的作用；不支持“Expert LoRA 普遍优于完整 Expert 更新”的结论。

所有百分数以最终 checkpoint 29999 汇总，避免按评测集挑选不同检查点。

## 数据与比较口径

- π0 base 初始化；LIBERO-Goal 10任务、428条演示；60条独立训练改写。
- 训练 seed=42，global batch=4，30k updates，双卡FSDP，EMA关闭。
- Goal：seed7，10任务×10次=100 episodes。
- Para：固定100条语言/任务manifest，seed7。
- Plus：官方 Language Instructions 分类，410 cases，每条1次、initial state 0，seed7；并非完整Plus所有扰动因素，也不是410个独立物理任务。
- 已核对新旧Plus classification SHA256一致：`faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6`。
- 均为单训练种子；固定样本配对检验不能替代跨训练种子、跨状态及跨任务泛化证据。Goal/Para与Plus采用不同协议，不直接比较三列的绝对难度。
- 早期默认LoRA配方实际训练完整SigLIP、Gemma-2B LoRA、Expert LoRA和投影层，约468.04M可训练参数，不能称为严格纯LoRA。

## 一、原架构：损失与语言增强

原架构保留语言token进入PaliGemma、动作专家读取多模态前缀的通路。以下均采用上述默认训练范围。

| 设置 | Goal 成功/100 | Para 成功/100 | Plus 成功/410（%） |
| --- | ---: | ---: | ---: |
| Canonical FM | 84 | 21 | 70（17.07） |
| 双指令FM增强，λ=0 | 89 | 50 | 173（42.20） |
| 对称一致性 λ=0.1 | 84 | 43 | 未测 |
| 对称一致性 λ=1 | 90 | 52 | 191（46.59） |
| 对称一致性 λ=10 | 79 | 41 | 未测 |
| Teacher λ=0.1 | 90 | 39 | 158（38.54） |
| Teacher λ=1 | 89 | 57 | 198（48.29） |
| Teacher λ=10 | 88 | 52 | 234（57.07） |

目标定义：

- Canonical：`FM(v_c, u)`。
- 增强：`0.5 * (FM(v_c,u) + FM(v_p,u))`。
- 对称一致性：增强项 `+ λ * MSE(v_c,v_p)`。
- Teacher：`FM(v_c,u) + λ * MSE(stop_gradient(v_c),v_p)`；teacher为当前共享模型的canonical分支，不是另一个冻结或EMA模型。

Teacher 相对 canonical λ=0基线的 λ=1增益：Goal +5pp、Para +36pp、Plus +31.22pp；这一对比包含接触训练改写的收益，不能全部归因于Teacher项。

相对已经包含改写监督的增强组，Teacher λ=1 为 Goal +0pp、Para +7pp、Plus +6.10pp。相对对称 λ=1，为 Goal -1pp、Para +5pp、Plus +1.71pp；差距不足以支持Teacher普遍优于对称形式。

Teacher与增强组还同时改变了改写分支是否直接接受FM监督，不能将方法级差异全部归因于stop-gradient算子。尚缺少“相同canonical-only FM加一致性、仅切换stop-gradient”的直接消融；共享flow坐标是否必要也尚未通过独立noise/time对照检验。

λ=1 在本轮Teacher扫描中Para最高，且满足事先约定Goal≥87%的门槛；λ=10 在Plus最高。按此前固定的开发规则仍以λ=1作为Para开发选择，不能根据Plus反复调参后再称Plus为未用于选择的外部验证。当前也不存在跨评测集一致的最优λ。

## 二、固定Teacher λ=1：训练模块消融

Gemma列指PaliGemma中的语言/多模态Transformer，SigLIP是其视觉编码器。

| SigLIP | Gemma-2B | Action Expert | 可训练参数 | Goal | Para | Plus |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 完整训练 | LoRA | LoRA | 468.04M | 89 | 57 | 48.29 |
| 冻结 | 冻结 | 完整训练 | 314.71M | 78 | 43 | 41.22 |
| 冻结 | LoRA | 完整训练 | 342.58M | 74 | 44 | 41.22 |
| 冻结 | LoRA | LoRA | 53.24M | 71 | 39 | 33.66 |

三项直接对照：

1. 固定Gemma/Expert的LoRA设置，完整训练SigLIP相对冻结SigLIP：Goal +18pp、Para +18pp、Plus +14.63pp。这是视觉适配贡献最直接的证据，但只在当前训练配方与单种子成立。
2. 冻结SigLIP、保持Gemma LoRA，完整Expert相对Expert LoRA：Goal +3pp、Para +5pp、Plus +7.56pp。因此完整Expert更新并非必然劣于LoRA。
3. 冻结SigLIP、保持完整Expert，增加Gemma LoRA：Goal -4pp、Para +1pp、Plus +0pp。本轮不支持语言侧LoRA单独带来额外收益。

尚未完成完整SigLIP×Expert更新方式的全因子实验；缺少“完整SigLIP + Gemma LoRA + 完整Expert”对照。不能把以上结果写成某个模块在所有场景都必要或某种参数化普遍更优。

## 三、新架构与Teacher的2×2比较

新架构：`v = v_base(o,s,x,t) + U(o,s,x,t) z(L)`，rank16。冻结T5-large的1024维mean-pool特征经可训练投影和L2归一化得到z；真实语言不再进入PaliGemma，前缀统一为中性文本。U由最后的动作隐状态线性生成，新头从零残差启动。可训练参数468.58M。

| 架构 | 训练目标 | Goal | Para | Plus |
| --- | --- | ---: | ---: | ---: |
| 原架构 | Canonical FM | 84 | 21 | 17.07 |
| 原架构 | Teacher λ=1 | 89 | 57 | 48.29 |
| rank16新接口 | Canonical FM | 78 | 16 | 7.07 |
| rank16新接口 | Teacher λ=1 | 73 | 19 | 27.56 |

Teacher在原架构上的变化：Goal +5pp、Para +36pp、Plus +31.22pp。

Teacher在新接口上的变化：Goal -5pp、Para +3pp、Plus +20.49pp。Plus净增加84个成功（29→113），但仍有297/410失败。Para只是净增加3个成功，不能宣称稳定改善。

新接口相对原架构：

- FM目标下：Goal -6pp、Para -5pp、Plus -10.00pp。
- Teacher目标下：Goal -16pp、Para -38pp、Plus -20.73pp。

两组新接口的Goal均低于原先87%门槛，未达到“保持原任务性能并提升改写泛化”的目标。结论限于当前T5+mean pooling+rank16+末层线性注入+移除原语言通路的整套实现，不能单独归因于低秩或T5。

最终训练日志的最后一个已打印窗口（step29900，非独立验证集）：

| 指标 | 新接口FM | 新接口Teacher |
| --- | ---: | ---: |
| canonical FM | 0.0230 | 0.0232 |
| paraphrase FM诊断 | 0.0330 | 0.0232 |
| 同义速度MSE | 0.0104 | 约0.0001 |
| 语义code平方距离 | 0.0428 | 0.0048 |

Teacher确实降低了训练同义分支差异，但这一诊断改善没有转换为充分的闭环改写成功率。不能用一致性loss变小代替行为有效性的证据，也不能据此断言不同语义已经塌缩。

新接口训练进展：

| Step | FM Goal/Para | Teacher Goal/Para |
| ---: | ---: | ---: |
| 10000 | 31/4 | 34/8 |
| 20000 | 42/6 | 40/15 |
| 29999 | 78/16 | 73/19 |

## 四、固定样本配对核验

Plus逐条按official_id匹配，并核验instruction、seed和initial_state一致；Para按eval_id、bddl_file、initial_state_idx、episode_idx匹配。修复指原来失败现在成功，退化指原来成功现在失败。

| 对比：后者相对前者 | 集合 | 修复 | 退化 | 净成功变化 | 双侧精确McNemar p |
| --- | --- | ---: | ---: | ---: | ---: |
| 原增强 → 原Teacher λ1 | Plus410 | 47 | 22 | +25 | 0.00354 |
| 原增强 → 原对称 λ1 | Plus410 | 50 | 32 | +18 | 0.0598 |
| 原Teacher λ1 → 原Teacher λ10 | Plus410 | 72 | 36 | +36 | 0.000684 |
| 新FM → 新Teacher λ1 | Plus410 | 102 | 18 | +84 | 1.98e-15 |
| 原Teacher λ1 → 新Teacher λ1 | Plus410 | 35 | 120 | -85 | 4.24e-12 |
| 原增强 → 原Teacher λ1 | Para100 | 17 | 10 | +7 | 0.248 |
| 新FM → 新Teacher λ1 | Para100 | 11 | 8 | +3 | 0.648 |

这些为事后描述性检验，未做多重比较校正；语言样本聚集于10个基础任务和固定状态，也不是独立训练重复。p值不能被解释为跨任务/跨种子的有效性证明。

## 五、下一步建议（本次未启动新实验）

1. 保留原架构Teacher λ1为当前Para开发参照，完整报告λ10的Plus结果；优先复核和扩展评测，不把新接口作为当前最佳方法。
2. 若继续架构路线，先做保留原PaliGemma语言通路的语义残差增量，并对新增分支做开/关对照，检验能否保持预训练跨模态能力。另行控制编码器、pooling、rank和注入位置，避免一次改变四项后归因。若从已微调checkpoint继续训练，baseline也必须从同一起点接受相同额外训练预算。
3. 对当前新接口先做同观测/同flow坐标的换指令、空指令及错误指令诊断，观察是否使用语言、是否选对目标。当前尚没有反事实目标切换证据。
4. 多seed仍按此前偏好暂缓；现有结论明确标注为单seed探索性结果。

## 原始证据路径

- `reports/validation/<run>/step_29999/libero_goal_seed7.json`
- `reports/validation/<run>/step_29999/libero_para_seed7/summary.json`
- 对称λ0.1历史路径：`reports/validation/step_29999/`，映射见`examples/libero_para/runs/controlled_experiment_v1.md`。
- `reports/libero_plus_language/<run>/seed7/summary.json`及`episodes.jsonl`
- `examples/libero_para/runs/`中的各训练范围与目标manifest。
- `logs/semantic_interface_v1/train_semantic_interface_r16_{fm,teacher_lam1}_30k.log`
- `logs/semantic_interface_v1_orchestrator.log`：两组训练/评测在2026-11-14 04:59:34 +08:00全部结束。
