# BARC-SGG：预算感知、角色解耦与基线锚定的遥感场景图生成

## 1. 方法定位

建议将最终框架暂命名为：

> **BARC-SGG: Budget-Aware, Role-structured and Calibrated Scene Graph Generation**

中文可写为：

> **预算感知、角色结构化与基线锚定校准的遥感场景图生成**

### 最终冻结版本

```text
Q5 Budget-Aware ordinary Top-10k
+ Dual Role-aware RPCM step-17500
+ Bottom-20 Beta(1,1)-smoothed frozen-Dual VAL-recall selector
+ 1,000-step PPG-consistent H+U adaptation
+ Base anchor alpha=0.3
```

Canonical checkpoint SHA256：

```text
04ba8311e0c0b8e01860b60d1acd5d00de0bcc61e398feaf05a3b7c251a5b12d
```

该版本已经完成 VAL 选型与一次 TEST 报告；不得再根据 TEST 修改 K、alpha、
loss、proposal 或 checkpoint。

方法不应被描述成三个互不相关的模块拼接。统一问题是：

> 在整图关系预算受限时，如何同时控制候选关系的语义分配、关系图中的角色化信息传播，以及长尾校准相对强基线的偏移幅度？

三个组件分别回答：

1. **Budget-Aware Proposal（Q5）**：有限的 10k pair 预算分给谁？
2. **Role-Decoupled Dual RPCM**：保留下来的 pair 按什么 endpoint role 传播？
3. **Base-Anchored Predicate Recalibration**：平衡分类器应离高 R 基线走多远？

这条主线可以概括成：

```text
GT objects / object predictions
        │
        ▼
all legal directed object pairs
        │
        ▼
Released PPG score + whole-image budget-aware residual
        │
        ▼
ordinary global Top-10k pair graph                 [where to reason]
        │
        ▼
SS / OO role-decoupled Dual RPCM message passing   [how to reason]
        │
        ▼
base predicate decision + balanced adaptation
        │
        ▼
validation-controlled base anchor                  [how far to adapt]
        │
        ▼
final predicate logits and triplet ranking
```

## 2. 统一动机

### 2.1 固定预算不是简单的 pair recall 问题

已有实验表明：

- 更高 GT-pair coverage 不一定带来更高 HMR；
- 大量候选可能集中在少数重复 object-class pair group；
- proposal 的主要作用是改变关系图的语义构成和推理上下文，而非仅增加正边数量。

因此第一阶段不是预测 predicate，也不是复现 Hard Quota，而是学习整图预算竞争。

### 2.2 统一 relation adjacency 会混淆 endpoint role

对于关系边 `e_i=(s_i,o_i)` 和 `e_j=(s_j,o_j)`，共享 endpoint 包含：

```text
SS: s_i = s_j
OO: o_i = o_j
SO: s_i = o_j
OS: o_i = s_j
```

统一 adjacency 将不同语义角色混入同一聚合。受控实验显示，分离 SS 与 OO
能够提高 mR/HMR，说明遥感场景图中的 subject/object role 不是可交换的。

### 2.3 强长尾更新会破坏高频关系

完整 class-balanced H/H+U 后训练在 TEST 上将 mR 提高约 6.7 点，但 R
下降约 3.6 点。逐谓词诊断表明：15 个困难谓词全部提高，而 R 损失主要来自
多个高频非困难谓词同时下降。因此问题不是长尾方向无效，而是更新幅度过大。

## 3. Component I：Whole-Image Budget-Aware Proposal Learning

### 3.1 Scorer

对图像 `I` 中的合法有向 pair `e=(v_s,v_o)`，使用 frozen Released PPG
分数作为锚点，并学习 set-conditioned residual：

```text
q_phi(e | I) = q_PPG(e) + Delta_phi(x_e, C_I).
```

`x_e` 只包含：

- subject/object class；
- pair geometry；
- Released PPG score；
- object-class-pair group statistics；
- whole-image candidate pressure。

`C_I` 是整图集合统计。它不包含 predicate label、RPCM logit、downstream
metric 或 TEST 信息。

### 3.2 Differentiable budget allocation

对全图候选执行 differentiable Soft-TopK：

```text
a_e in [0,1],       sum_e a_e approximately K,
```

正式预算固定为 `K=10000`。对 object-class pair group
`g=(c_s,c_o)`，定义 train annotated positive pair 的软覆盖：

```text
r_g = mean_{e in P_g} a_e.
```

优化目标为：

```text
L_Q5 = L_anchor - mean_g log(epsilon + r_g),
```

其中 `L_anchor` 是 student score 与 Released PPG score 的 mean-MSE 锚点。
对数效用使低覆盖 group 具有更高边际收益，而已经充分覆盖的重复 group 收益递减。

### 3.3 关键边界

- 只把 annotated relation pairs 当 positive；
- 未标注 pair 不作为 negative；
- 不使用 Hard Quota teacher；
- 不使用 predicate supervision；
- 不使用 RPCM/downstream utility；
- 推理使用 ordinary global Top-10k，不执行硬 group quota。

因此论文名称应使用 **Budget-Aware PPG / Whole-Image Budgeted Proposal**，
不要与已归档且 Gate 失败的旧 L-RSGP 混称。

## 4. Component II：Role-Decoupled Dual RPCM

### 4.1 Typed relation graph

对保留的 `E` 条关系分别构造：

```text
A_SS[i,j] = 1[s_i = s_j],
A_OO[i,j] = 1[o_i = o_j].
```

subject 与 object entity-to-relation collector 也保持独立。第 `t` 层关系更新可
写成：

```text
m_i^sub = Collect_sub(h_i, entity states),
m_i^obj = Collect_obj(h_i, entity states),
m_i^SS  = GCN_SS(h, A_SS),
m_i^OO  = GCN_OO(h, A_OO),

h_i^(t+1) = Mean(m_i^sub, m_i^obj, m_i^SS, m_i^OO).
```

最终使用所有更新阶段的 relation state 均值，并进入原 RPCM pairwise/prototype
classifier。当前 incidence backend 只改变计算方式，不改变这一语义定义。

### 4.2 Controlled graph ablation

三个对照共享 seed、初始状态、训练协议与 classifier，仅改变 relation adjacency：

| Variant @ step 17500 | R@2k | mR@2k | HMR@2k |
|---|---:|---:|---:|
| Unified-All | 69.782 | 40.796 | 51.490 |
| Merged-SameRole | 70.625 | 41.051 | 51.922 |
| **Dual Role-aware** | 69.674 | **42.407** | **52.724** |

相对 Unified，Dual 的 mR/HMR 提高 1.611/1.234 点；相对 Merged-SameRole，
提高 1.356/0.802 点。由此可将贡献写成：

> 不是简单增加 GNN 深度，而是消除统一 relation graph 中的 endpoint-role aliasing。

## 5. Component III：Base-Anchored Predicate Recalibration

### 5.1 强平衡方向

在 frozen detector、pair extractor 和 Role-aware GNN 上，使用 Released PPG
Top-10k TRAIN relation features 做 proposal-consistent 后训练。

所有前景类参与 class-balanced CE；困难集合 `H` 额外使用 one-vs-rest
residual supervision。正式困难类选择规则为：

```text
R_tilde_c = (R_val_c * N_val_c + 1) / (N_val_c + 2)
H_K = Bottom-K predicates by R_tilde_c
```

并列时按 predicate ID。`K={10,15,20,25}` 只在 VAL 比较，最终按 R 约束后的
最大 HMR 选择 `K=20`。Frequency/confusion 保留为审计信号，不进入正式 score。
选择 manifest 明确禁止 TEST 与 recalibration outcome。

旧 development-H15 继续作为历史兼容结果，但不再代表最终可复现 selector。

实现中还保留 sample-conditioned universal residual `U`，但当前 H→H+U
消融没有显示总体增益，因此 **U 不应被单独列为论文贡献**。更稳妥的论文表述是：

> all-class balanced predicate adaptation with a hard-predicate auxiliary residual。

离线后训练目标可概括为：

```text
L_adapt = L_CE
        + 0.1 L_LA
        + 0.2 L_hard-BCE
        + L_preserve-KL
        + 1e-4 L_parameter-anchor
        + L_RPCM-prototype.
```

未标注 PPG pair 只承担 base-distribution preservation，不作为 background negative。

### 5.2 Base-anchored trust continuation

直接使用完整 balanced update 会产生剧烈 R/mR 交换。令：

- `theta_0`：高 R 的 frozen Dual base predicate classifier；
- `theta_1`：proposal-consistent balanced post-training classifier。

最终 classifier 取：

```text
theta(alpha) = theta_0 + alpha (theta_1 - theta_0).
```

H/U residual output 也使用同一 `alpha`。因此：

```text
alpha = 0  -> exact base decision function,
alpha = 1  -> full balanced adaptation,
||theta(alpha)-theta_0|| = alpha ||theta_1-theta_0||.
```

这给出一个显式、可审计的 stability-plasticity 路径。验证规则预先固定为：

1. R@2k 相对 Base 下降不得超过 1 点；
2. 在可行点中最大化 mR/HMR；
3. 只搜索 `alpha={0.1,0.2,0.3}`。

最终冻结 `alpha=0.3`。

### 5.3 与 BAL 的区别

本方法可以引用 BAL 作为“frozen strong model 上进行后训练校准”的相关工作，
但不能称为 BAL 复现：

- 不训练 adversarial bias generator；
- 不使用 BGAN/GCD；
- 不使用 final-group generator；
- 直接学习真实 PPG graph context 下的 balanced update；
- 用 base-anchored trust coefficient 限制整个 predicate decision drift。

核心问题是 **安全适应幅度**，而不是生成一个无条件 logit bias。

## 6. 完整训练与推理协议

### 6.1 分阶段训练

```text
Stage A: Role-aware Dual RPCM
    historical-6850 matched protocol
    detector frozen
    ordinary relation supervision

Stage B: Q5 proposal residual
    Released PPG frozen
    whole-image positive-only Soft-TopK objective
    no RPCM/predicate/downstream supervision

Stage C: Predicate recalibration
    frozen Role-aware representation
    Released PPG Top-10k TRAIN feature cache
    formal smoothed-VAL-recall hard Top20
    all-class CE + hard auxiliary + preservation

Stage D: Base anchoring
    validation-only alpha selection
    no gradient training and no TEST selection
```

### 6.2 最终推理

```text
all legal pairs
 -> Q5 Budget-Aware score
 -> ordinary Top-10k
 -> Dual Role-aware RPCM
 -> Base-Anchored predicate classifier (alpha=0.3)
 -> triplet ranking
```

### 6.3 当前一致性边界

当前 predicate recalibration 使用 Released PPG Top-10k TRAIN features，而最终
inference 使用 Q5 Top-10k。不能声称“Q5-matched recalibration training”。

但 2x2 结果显示 calibration 在 PPG/Q5 上均有效，且 proposal/calibration 增益近似
可加，这支持 **模块兼容与 proposal transfer**。若后续重新导出 Q5 TRAIN feature
cache 并重训，只能作为 matched-training 额外消融，不应回头用 TEST 选择超参数。

## 7. 核心结果与贡献分解

### 7.1 Development-H15 机制 2×2

STAR TEST：

| Proposal | Predicate head | R@2k | mR@2k | HMR@2k |
|---|---|---:|---:|---:|
| Released PPG | Dual Base | 69.674 | 42.407 | 52.724 |
| Q5 Budget-Aware | Dual Base | 70.117 | 42.997 | 53.306 |
| Released PPG | Development-H15 Anchored | 69.648 | 44.530 | 54.326 |
| Q5 Budget-Aware | Development-H15 Anchored | 70.118 | 45.076 | 54.875 |

贡献分解：

```text
Q5 proposal effect on Base:
    Delta R/mR/HMR = +0.443 / +0.590 / +0.582

Anchored calibration effect on PPG:
    Delta R/mR/HMR = -0.027 / +2.123 / +1.602

Final method vs Base:
    Delta R/mR/HMR = +0.443 / +2.669 / +2.151
```

proposal 与 calibration 的 interaction 很小，说明两者主要是互补、近似可加的：

- Q5 改变有限预算中的 graph/context composition；
- Role-aware RPCM 解释这些结构；
- anchored recalibration 改变 predicate decision boundary，同时保护 Base R。

### 7.2 最终 Rule-Top20 BARC-SGG

| Method | R@2k | mR@2k | HMR@2k |
|---|---:|---:|---:|
| Frozen Dual + Q5 | 70.1169 | 42.9969 | 53.3058 |
| Development-H15 + Q5 | 70.1178 | 45.0760 | 54.8750 |
| **Final Rule-Top20 + Q5** | **70.0976** | **45.0916** | **54.8803** |

最终模型相对 Frozen Dual+Q5：R −0.0193、mR +2.0947、HMR +1.5746 点。
相对 Dual+Released PPG：R +0.4232、mR +2.6844、HMR +2.1563 点。

Rule-Top20 与旧 development-H15 几乎等价，说明经验 hard set 已被一个确定性、
TEST-free protocol 替代，而没有改变方法能力。

## 8. 正式消融设计

### 8.1 Role graph ablation

```text
Unified-All
Merged-SameRole
Dual SS/OO
```

必须保持初始化、训练 sampler、step、classifier 完全一致。

### 8.2 Proposal ablation

```text
Released PPG Top-10k
Q5 Anchor（应与 Released 完全一致）
Q5 Budget-Aware Top-10k
历史 Hard Quota（reference only）
```

### 8.3 Predicate adaptation ablation

```text
Dual Base
full all-class balanced update
H-only
H+U（optional analysis）
Anchored update alpha={0.1,0.2,0.3}
Rule hard K={10,15,20,25}
```

由于 H-only 略优于 H+U，U 不能被包装成必需贡献。正式方法应聚焦：

```text
all-class adaptation + hard auxiliary + base anchor
```

K=10--25 的 VAL HMR 最大跨度仅0.0101点。K20按精确 VAL HMR选中，但不能
声称显著优于K15；正式结论是对 hard-set size 稳健。

### 8.4 最重要的 2×2

```text
              Released PPG       Q5
Dual Base          ✓              ✓
Anchored head      ✓              ✓
```

这张表直接证明 proposal effect、calibration effect 和 interaction。

## 9. 论文可安全声称的内容

### 可以声称

1. 一个不依赖 Hard Quota、predicate label 或 downstream supervision 的整图
   budget-aware proposal learner。
2. 一个显式分离 SS/OO endpoint roles 的 Dual RPCM relation graph。
3. 一个通过 base-anchored trust continuation 控制 R/mR 交换的安全后训练方法。
4. 一个由 smoothed frozen-base VAL recall 确定、完全 TEST-free 的 hard Top20。
5. 最终模型相对 Dual+PPG 同时提高 R、mR、HMR。
6. Q5 与 anchored recalibration 的增益在 2×2 中近似可加。

### 不应声称

1. 不应再称 Q5 为旧 L-RSGP；两者机制和实验证据不同。
2. 不应声称完全 end-to-end joint training；当前是模块化分阶段训练。
3. 不应声称 Q5-matched predicate post-training；当前 HPRC cache 来自 PPG。
4. 不应把 U residual 单独称为有效贡献。
5. 不应称为 BAL 复现或 BGAN/GCD 实现。
6. 不应声称更高 micro pair coverage 是增益原因；Q5 的 micro coverage可以更低。
7. 不应声称 K20 显著优于 K15；四档差异低于0.02点。

## 10. 方法章节建议结构

```text
3.1 Overview: budget allocation, role reasoning, safe calibration
3.2 Whole-Image Budget-Aware Pair Proposal
3.3 Role-Decoupled Dual Relation Reasoning
3.4 Base-Anchored Predicate Recalibration
3.5 Staged Optimization and Inference
```

贡献点可以写成：

1. **Budget allocation**：提出 positive-only、Hard-Quota-free 的整图预算学习，
   用 proportional-fair group coverage 抑制重复 semantic group 占满预算。
2. **Role reasoning**：将统一 relation graph 分解为 subject-sharing 与
   object-sharing 双视图，缓解 endpoint-role aliasing。
3. **Safe recalibration**：学习强 class-balanced predicate update，并用 frozen
   Base 的显式 trust path 保留 head recall，在 TEST 上实现 R/mR/HMR 同时提升。

## 11. 一段可直接用于摘要的方法描述

> We present BARC-SGG, a modular framework that coordinates relation-budget
> allocation, role-structured graph reasoning, and stable predicate
> recalibration for remote-sensing scene graph generation. First, a
> whole-image budget-aware proposal learner adjusts a released pair score
> using target-free set context and a positive-only proportional-coverage
> objective, producing an ordinary global Top-K graph without hard semantic
> quotas. Second, a role-decoupled RPCM propagates shared-subject and
> shared-object relation contexts through separate graph views, avoiding the
> endpoint-role aliasing of a unified relation adjacency. Third, a balanced
> predicate update is constrained along an explicit base-anchored trust path,
> retaining the recall of a strong classifier while improving under-recognized
> predicates. On STAR PredCls, the complete method improves R/mR/HMR@2000 by
> 0.42/2.68/2.16 percentage points over the Role-aware Dual + released PPG
> baseline.

## 12. 仍需补齐的正式论文证据

1. Anchored recalibration 至少 3 个 matched seeds；当前最终结果主要是 seed 1029。
2. α/K 已固定用 VAL 选择；后续只能补充 full-val bootstrap，不得再根据 TEST 修改。
3. 最终 2×2 的 paired bootstrap 可作为统计补充，但不得改变冻结配置。
4. 若篇幅允许，增加 Q5-matched recalibration 作为一致性消融，但不改当前主结论。
5. SGCls/SGDet 只能在 PredCls 结构与超参数冻结后运行。
