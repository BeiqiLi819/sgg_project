# 论文方法、贡献与实验记录

> Status: paper working draft
>
> Last updated: 2026-08-04
>
> Scope: STAR, OBB, PredCls/SGCls/SGDet
>
> Reference baseline: STAR paper and SGG-ToolKit
>
> Table unit: percentage (%)

本文档是当前论文方法、实验设置和结果来源的主记录。最终执行协议以
[submission_experiment_protocol.md](submission_experiment_protocol.md) 为准，
RSGP 的实现细节以 [rsgp_technical_route.md](rsgp_technical_route.md) 为准。
可直接复制到论文并已区分正文/附录的精简稿位于
[paper_ready_manuscript.md](paper_ready_manuscript.md)。

当前主方法固定为：

```text
role-aware dual-view RCA
+ auxiliary logit adjustment (LA)
+ statistical RSGP
```

HPRC/`tail_aux` 和基于手工类别组的 RSGP-v1 不属于当前投稿方法。它们的
checkpoint 兼容代码与历史结果可以保留，但不能改名为当前方法的结果。

---

## 1. 论文定位与贡献边界

### 1.1 一句话定位

针对大幅遥感图像中候选实体对数量庞大、关系边的 subject/object 角色容易在
统一邻接中混叠、谓词分布不均的问题，本文提出一种由角色感知双视角关系聚合、
弱先验校准和固定预算候选图构建组成的 OBB 场景图生成框架。

英文表述：

> We present a remote-sensing-oriented scene graph generation framework that
> combines role-aware dual-view relation context aggregation, weak
> prior-aware predicate supervision, and budget-constrained candidate-graph
> construction.

### 1.2 STAR/SGG-ToolKit 已有内容

以下内容属于 STAR 原工作或 SGG-ToolKit，不作为本文创新：

- STAR 数据集、48 个前景实体类别和 58 个前景关系类别；
- OBB PredCls、SGCls 和 SGDet 三项任务；
- Swin-L OBB detector 和大图多尺度 patch 检测；
- Semantic Filter、PPG 和 top-10,000 pair budget；
- RPCM 的 pair/union feature、对象与关系上下文、GloVe prototype classifier；
- RPCM 已有的 relation-to-relation 信息传播。

因此，论文不能声称“首次在 STAR 中引入 GNN”或“首次进行边到边传播”。

### 1.3 本文真正改变的部分

1. 将关系边图拆为 shared-subject 与 shared-object 两个角色视角，并在相同
   RCA 更新模块中分别传播；
2. 保留主 CE 分类目标，只增加小权重 logit-adjust auxiliary loss；
3. 将 pair proposal 从独立 pair 排序改写为固定预算下的有向候选子图选择；
4. 用 train split 自动统计的软结构角色替代 STAR 类名和手工 predicate ID；
5. 在统一 OBB detector、类别通道、角度、NMS 和 evaluator 后完成三项任务。

detector background 通道重排、OBB angle/offset、bbox coder、late NMS 和
detection cache 属于兼容性与实验基础设施，不作为算法贡献。

### 1.4 主要贡献

**Contribution 1 — Role-aware dual-view RCA.**

在关系边视角上显式区分 shared-subject 与 shared-object 邻接，避免统一
relation graph 将不同端点角色混合为同一种消息。

**Contribution 2 — Auxiliary LA.**

在 prototype CE 之外加入弱 logit adjustment 监督，以训练集 predicate prior
改善类别不均衡；推理分类器和 logits 不被手工修改。LA 是已有思想的稳定集成，
不单独宣称为全新损失。

**Contribution 3 — RSGP.**

提出 Remote-sensing Graph-aware Pair Proposal，将候选筛选表述为带 degree
capacity 与 semantic-type capacity 的固定预算最大权重有向子图构建。其统计式
版本不读取类别名称，也不使用人工类别组。

**Contribution 4 — Complete OBB evaluation.**

在相同实现中完成 PredCls、SGCls 和 SGDet，并同时给出 STAR-compatible 与
strict predicted-label 协议。该项作为完整性贡献，不替代前三项算法贡献。

### 1.5 贡献与实验依据

| 对比 | 变化 | 证明对象 |
|---|---|---|
| Dual − Base | unified → shared-subject/shared-object dual view；其余 current RCA/head 不变 | 角色分解 |
| DL − D | `lambda_LA: 0 → 0.1` | auxiliary LA |
| Full − DL | 同一 DL checkpoint，仅 PPG → statistical RSGP | RSGP |
| PPN vs PPG/RSGP | 相同 relation checkpoint 与 top-10,000 预算 | pair coverage 不是充分目标 |

这里的 Base 即原输出目录中的 `unified` 行。完整 source RPCM、STAR 论文结果和
`6850_4135.pth` replay 均作为外部/审计参考，不插入上述因果链。

---

## 2. 方法

### 2.1 总体流程

关系模型训练：

```text
OBB entities
→ Semantic Filter
→ PPG top-10000
→ pair/union representation
→ role-aware dual-view RCA
→ semantic prototype classifier
→ CE + auxiliary LA + prototype regularization
```

完整方法推理：

```text
OBB entities/detections
→ Semantic Filter
→ statistical RSGP top-10000
→ the trained dual-view RCA
→ the unchanged prototype classifier
→ graph-constrained triplet ranking
```

RSGP 是 inference-only filter，不参与 relation-head 反向传播。Full 行必须
复用字节相同的 DL checkpoint，不能单独训练“RSGP 模型”。

### 2.2 问题定义

给定实体集合

\[
\mathcal V=\{v_i=(b_i,l_i,f_i)\}_{i=1}^{N},
\]

其中 \(b_i=(x_i,y_i,w_i,h_i,\theta_i)\) 为 OBB，\(l_i\) 为实体类别，
\(f_i\) 为 RoI feature。STAR 的内部 ID 约定为：

```text
object:    0 = background, 1...48 = foreground
predicate: 0 = background, 1...58 = foreground
```

Semantic Filter 产生有向候选全集

\[
\mathcal E_0=
\{(i,j)\mid i\ne j,\ M^{sem}_{l_i,l_j}=1\}.
\]

当 \(|\mathcal E_0|>K\) 时必须筛选，本文与 STAR 协议固定
\(K=10{,}000\)。

PredCls 使用 GT boxes/labels；SGCls 使用 GT boxes 和预测 object outputs；
SGDet 使用 detector boxes/outputs。为直接对齐 STAR，legacy pair filtering
分别使用：

```text
SGCLS_FILTER_LABEL_SOURCE=gt
SGDET_FILTER_LABEL_SOURCE=matched_gt
```

strict 补充协议则使用预测标签做 filtering。最终 object/predicate 输出始终由
对应任务模型产生，filter label source 只控制候选对构建。

### 2.3 Pair and union representation

沿用 RPCM 的 pair extractor。对候选 pair \((i,j)\)：

\[
h_{ij}^{r,0}=\Phi_{pair}
\left(f_i,f_j,g(l_i),g(l_j),
\phi_{pos}(b_i,b_j),f_{ij}^{union}\right).
\]

这里包含 subject/object RoI feature、对象 GloVe、OBB 相对位置和 union
visual feature。该模块是公共基础，不作为本文创新。

### 2.4 SGG-ToolKit source audit

设 \(E=|\mathcal E|\)，subject/object incidence matrix 为

\[
(M_s)_{v,e}=\mathbb 1[v=s_e],\qquad
(M_o)_{v,e}=\mathbb 1[v=o_e].
\]

Source audit 使用统一关系邻接：

\[
A_u=\mathbb 1[(M_s+M_o)^\top(M_s+M_o)>0]-I.
\]

因此 shared-subject、shared-object 和 cross-role endpoint sharing 都进入
同一个 \(A_u\)。对象图在每张图内部为除 self-loop 外的完全图
\(A_e^{base}\)。

原 SGG-ToolKit 的六路 collector 为

\[
\operatorname{Collect}_{q}(T,S,A)=
\frac{A\,\operatorname{ReLU}(SW_q+b_q)}
{A\mathbf 1+\epsilon},
\]

其中 \(q=0,\ldots,5\) 分别表示：

```text
0 relation → subject entity
1 relation → object entity
2 subject entity → relation
3 object entity → relation
4 entity → entity
5 relation → relation
```

每轮共享同一组六个 collector，update 为参数无关的残差：

\[
\operatorname{Update}(T,C)=T+C.
\]

因此

\[
H^{e,l+1}=H^{e,l}+\frac{1}{3}
\left(m_{e\to e}^{l}+m_{r\to s}^{l}+m_{r\to o}^{l}\right),
\]

\[
H_{base}^{r,l+1}=H^{r,l}+\frac{1}{3}
\left(m_{s\to r}^{l}+m_{o\to r}^{l}+m_{u,r\to r}^{l}\right).
\]

Source audit 分类使用最后一轮 \(H_{base}^{r,L}\)。该分支由独立 predictor
`RPCM_SGG_TOOLKIT_ORIGINAL` 实现；它不仅恢复上述 GNN，也恢复 source
classifier，不能用仅设置
`RPCM_RELATION_GRAPH_MODE="sgg_toolkit"` 的 later-RPCM predictor 代替。

#### 2.4.1 Source gated prototype classifier

原版在 GNN 后另行构造 300-D object/predicate GloVe embedding（与
pairwise extractor 内部的 200-D object embedding 是两套参数）。令
\(e_i\) 为 object text embedding，\(x_i^s,x_i^o\) 为对象视觉特征的
subject/object 投影，首先得到

\[
s_i=W_se_i+
\sigma\!\left(G_s[W_se_i;h(x_i^s)]\right)\odot h(x_i^s),
\]

\[
o_j=W_oe_j+
\sigma\!\left(G_o[W_oe_j;h(x_j^o)]\right)\odot h(x_j^o).
\]

经过 residual linear layer 与 LayerNorm 后，subject-object 融合为

\[
F(s_i,o_j)=\operatorname{ReLU}(s_i+o_j)-(s_i-o_j)^2.
\]

设 \(u_{ij}=h(\operatorname{DownSamp}(H_{ij}^{r,L}))\)，原版使用减法门控
构造关系表示：

\[
\widetilde r_{ij}=F(s_i,o_j)-
\sigma\!\left(G_p[F(s_i,o_j);u_{ij}]\right)\odot u_{ij},
\]

\[
z_{ij}=\operatorname{Proj}\!\left(
\operatorname{Drop}\left[
\operatorname{ReLU}\left(
\operatorname{LN}(\widetilde r_{ij}+
\operatorname{Drop}(\operatorname{ReLU}(W_r\widetilde r_{ij})))
\right)\right]\right).
\]

predicate 文本向量 \(t_c\) 通过 \(W_p\) 和同一个 projection head 得到
\(c_c\)。最终 source logits 为

\[
\ell_{ij,c}=
\exp(\gamma)\,
\frac{z_{ij}^{\top}c_c}
{\lVert z_{ij}\rVert_2\lVert c_c\rVert_2},
\qquad \gamma=\log(1/0.07).
\]

每次 forward 还对 58 个 foreground semantic prototypes 执行固定
`random_state=0` 的 KMeans，得到包含 background 的 \(P=30\) 个 detached
coarse prototypes。coarse prototypes 不参与最终 predicate logits，只服务
原版辅助正则。Source audit 原样保留五项 add-loss：

\[
\mathcal L_{2,1}^{fine}
=\frac{\lVert C_nC_n^\top\rVert_{2,1}}{C^2},
\qquad
\mathcal L_{2,1}^{coarse}
=\frac{\lVert \widetilde C_n\widetilde C_n^\top\rVert_{2,1}}{P^2},
\]

\[
\mathcal L_{euc}^{fine}
=\frac1C\sum_c\max(0,7-d_c^-),\qquad
\mathcal L_{euc}^{coarse}
=\frac1P\sum_p\max(0,7-\widetilde d_p^-),
\]

\[
\mathcal L_{dis}
=\frac1{|\mathcal E|}
\sum_{(i,j)}
\max\left(0,d_{ij}^{+}-\overline d_{ij,10}^{-}+1\right).
\]

Source audit 总损失为主 CE 加上述五项未再缩放的 source loss。该分支同时保留
source 的历史 GloVe tokenization：object 类别只按空格取最长 token，
predicate phrase 按空格对有效 token 求均值，不做 underscore 拆分、
modifier-aware 校正或向量归一化。

### 2.5 Current RCA: Base (Unified) and Dual-view

Base、D 和 DL 使用同一 current RCA GNN；Base 与 D 的核心区别只是 relation
adjacency。

RCA 的归一化 residual GCN 为

\[
\widetilde A=A+I,\qquad
\widehat A=D^{-1/2}\widetilde A D^{-1/2},
\]

\[
\operatorname{GCN}(H,A)=
\sigma(\widehat A\,\operatorname{Drop}(H)W+b+\operatorname{Drop}(H)).
\]

实体到关系的 role-specific collection 为

\[
\operatorname{Collect}_{u}(H^e,M_u)=
\frac{M_u^\top\operatorname{ReLU}(H^eW_u+b_u)}
{M_u^\top\mathbf 1+\epsilon},
\qquad u\in\{s,o\}.
\]

Base 仍使用 \(A_u\)，其关系更新为

\[
H_B^{r,l+1}=\frac{1}{3}
\left[
\operatorname{Collect}_{s}(H^{e,l},M_s)+
\operatorname{Collect}_{o}(H^{e,l},M_o)+
\operatorname{GCN}_{r}(H^{r,l},A_u)
\right].
\]

D/DL 构造

\[
A_s=\mathbb 1[M_s^\top M_s>0]-I,\qquad
A_o=\mathbb 1[M_o^\top M_o>0]-I,
\]

并使用共享参数分别传播：

\[
H_D^{r,l+1}=\frac{1}{4}
\left[
\operatorname{Collect}_{s}(H^{e,l},M_s)+
\operatorname{Collect}_{o}(H^{e,l},M_o)+
\operatorname{GCN}_{r}(H^{r,l},A_s)+
\operatorname{GCN}_{r}(H^{r,l},A_o)
\right].
\]

关系输出聚合输入状态和全部更新状态：

\[
\bar H^r=\frac{1}{L+1}\sum_{l=0}^{L}H^{r,l}.
\]

随后

\[
H^{rel}=\operatorname{LayerNorm}
\left(
\operatorname{DownSamp}(\bar H^r)+
\operatorname{MLP}_{res}(\operatorname{DownSamp}(\bar H^r))
\right).
\]

PredCls 使用 \(L=4\)，SGCls/SGDet 使用 \(L=3\)。`exact_6850` dual-view
分支还对 object states 做 all-layer mean；PredCls 返回 GT one-hot object
logits，因此该 object-state 差异不影响 PredCls 分类结果，但会影响
SGCls/SGDet object refinement。

### 2.6 6850-compatible semantic prototype classifier（Base/D/DL）

每个 predicate 使用一个 300-D GloVe prototype。设文本向量为 \(t_c\)：

\[
p_c=\operatorname{Norm}(W_pt_c),\qquad
p_c^0=\operatorname{Norm}(W_p^0t_c).
\]

`proto_ema` 在当前 exact-6850 路线中是静态初始化锚点，不是 batch-wise
visual EMA：

\[
\bar p_c=\operatorname{Norm}
\left(\rho p_c+(1-\rho)p_c^0\right),\qquad \rho=0.9.
\]

relation embedding 和 predicate logits 为

\[
z_{ij}=\operatorname{Norm}(\Phi_{proj}(H_{ij}^{rel})),
\qquad
\ell_{ij,c}=\tau z_{ij}^{\top}\bar p_c.
\]

Base/D/DL 共同使用的 prototype regularization 为

\[
\mathcal L_{pull}
=\lambda_{pull}\frac{1}{|\mathcal E_{train}|}
\sum_{(i,j)}(1-z_{ij}^{\top}\bar p_{y_{ij}}),
\]

\[
\mathcal L_{sep}
=\lambda_{sep}\frac{1}{C(C-1)}
\sum_{c\ne d}\left(
\bar p_c^\top\bar p_d+\frac{1}{C-1}
\right)^2.
\]

exact-6850 还保留固定语义 pair set \(\mathcal A\) 的历史 margin 项：

\[
\mathcal L_{ant}
=\frac{\lambda_{ant}}{|\mathcal A|}
\sum_{(a,b)\in\mathcal A}
\max(0,m_{ant}-\bar p_a^\top\bar p_b).
\]

当前参数为
\(\lambda_{pull}=0.2\)、\(\lambda_{sep}=0.01\)、
\(\lambda_{ant}=0.1\)、\(m_{ant}=-0.2\)。这些均来自公共 RPCM-6850
基础，不作为本文新贡献。

### 2.7 Auxiliary Logit Adjustment

由 train split 的 predicate counts \(n_c\) 得到

\[
\pi_c=\frac{\max(n_c,1)}{\sum_k\max(n_k,1)}.
\]

主分类损失保持

\[
\mathcal L_{CE}=\operatorname{CE}(\ell,y).
\]

DL 额外使用

\[
\mathcal L_{LA}=
\operatorname{CE}(\ell+\tau_{LA}\log\pi,y),
\]

\[
\mathcal L_{pred}=
\mathcal L_{CE}
+\lambda_{LA}\mathcal L_{LA}
+\mathcal L_{pull}
+\mathcal L_{sep}
+\mathcal L_{ant},
\]

其中 \(\lambda_{LA}=0.1,\tau_{LA}=0.5\)。LA 仅改变训练梯度；推理仍使用
原始 \(\ell\)。SGCls/SGDet 再加入 object refinement CE：

\[
\mathcal L=
\mathcal L_{pred}+
\mathbb 1[\text{task}\ne\text{PredCls}]\,\mathcal L_{obj}.
\]

冻结 detector 的 RPN/box losses 不参与 relation-stack 优化。PredCls 使用
GT one-hot object outputs，`OBJECT_REFINE_LOSS_WEIGHT=0`。

### 2.8 Statistical RSGP

RSGP 将 candidate selection 定义为带容量正则的固定预算子图选择：

\[
\max_{\mathcal E\subseteq\mathcal E_0}
\sum_{(i,j)\in\mathcal E}S_{ij},
\]

其优先满足

\[
|\mathcal E|\le K,\quad
d_{out}(i)\le D_o,\quad
d_{in}(j)\le D_i,\quad
n_{l_i,l_j}\le Q.
\]

当前 inference-only 实现采用 greedy approximation。为避免候选不足导致与
top-$K$ 协议不一致，它先使用严格容量、再使用放宽容量；若仍不足 $K$，最后
从剩余 ranked pool 补齐。因此除 $|\mathcal E|\le K$ 外，degree 和
semantic-type capacity 是选择优先级与软约束，不保证最终补齐后的图仍严格满足。

#### Train-derived structural roles

一次性预处理只读取 train split 的 boxes、labels 和 relation annotations。
对 object class ID \(c\) 统计：

\[
\rho_c=\frac{1}{2}\mathbb E[q(A_i)]
+\frac{1}{2}\mathbb E[\operatorname{contain}(i)],
\]

\[
\alpha_c=\mathbb E\left[
1-\frac{\min(w_i,h_i)}{\max(w_i,h_i)}
\right],
\]

\[
\kappa_c=\mathbb E\left[
\frac{\log(1+d_i)}{\log(1+N_i)}
\right].
\]

三者分别表示 contextual-region、directional-alignment 和
relational-connectivity 的软 profile。统计文件仅按 class ID 索引，不读取
class-name string。

#### Generic structural kernels

几何核：

\[
S^{geom}_{ij}
=0.35IoU^{env}_{ij}
+0.30e^{-\bar d_{ij}}
+0.20c^{compact}_{ij}
+0.15|\cos\Delta\theta_{ij}|.
\]

Context kernel 从当前图选取

\[
M=\min(128,\max(16,\lceil2\sqrt N\rceil))
\]

个高承载分实体，计算实例到 carrier 的 OBB containment、距离以及
shared/different carrier affinity。Directional kernel 对所有合法 pair 计算
主轴平行度、沿轴位移和横向位移；connectivity kernel 结合 class connectivity
profile、当前 semantic degree 和距离。三者都没有 apron/vehicle/network
类别门控。

#### Frequency-adaptive support

\[
\omega_r=(f_r+\epsilon)^{-0.5},
\qquad
S^{rare}_{ab}
=\max_{r:M_{abr}=1}\widetilde\omega_r.
\]

它保护可能表达低频 predicate 的 label pair，但不使用固定 hard-predicate ID。

#### Multi-source utility and constrained selection

\[
S_{ij}=
w_p\widehat S^{PPG}_{ij}
+w_n\widehat S^{PPN}_{ij}
+w_g\widehat S^{geom}_{ij}
+w_c\widehat S^{context}_{ij}
+w_a\widehat S^{align}_{ij}
+w_k\widehat S^{connect}_{ij}
+w_rS^{rare}_{ij}
+w_d\widehat S^{balance}_{ij}.
\]

当前默认：

```text
(wp, wn, wg, wc, wa, wk, wr, wd)
= (1.0, 0.35, 0.35, 0.25, 0.10, 0.10, 0.15, 0.15)
```

候选流程：

```text
Semantic Filter
→ PPG protected top-P, P ∈ {7000, 8000, 9000}
→ PPN completion top-12000
→ statistical structural top-12000
→ hybrid utility ranking
→ in/out degree capacity 96/96
→ semantic-type capacity 800
→ relaxed second pass 128/128 and 1200
→ unconstrained ranked completion when still below top-10000
→ final top-10000
```

paper validation grid 比较了 7000/8000/9000，并由 `selection.json` 选中
9000。选型完成后，代码基础配置与公开脚本的 \(P\) 默认值已冻结为 9000；历史
RSGP-v1 最优为 8000/2000，不能据此替代 statistical validation 记录。

如果 \(|\mathcal E_0|\le10{,}000\)，不发生 proposal truncation，
\(\mathcal E^*=\mathcal E_0\)。

#### Legacy boundary

`RSGP_ROLE_MODE=legacy_manual` 使用手工 class-name groups 和 predicate IDs

```text
[7, 14, 20, 24, 25, 28, 31, 33, 36, 38, 39, 41, 53, 56, 58]
```

仅用于复现 RSGP-v1。论文默认 `RSGP_ROLE_MODE=statistical`，完全忽略
`RSGP_ANCHOR_CLASSES`、`RSGP_VEHICLE_CLASSES`、
`RSGP_NETWORK_CLASSES` 和 `RSGP_TAIL_PREDICATES`。上述 ID 也正是历史
HPRC residual head 使用的实际 predicate ID；HPRC 已排除出当前方法。

### 2.9 Graph-constrained triplet inference

每个候选 pair 只保留最高分前景 predicate：

\[
\hat r_{ij}=\arg\max_{c\in\{1,\ldots,58\}}p(r=c\mid i,j).
\]

triplet score 为

\[
S^{triplet}_{ij}=
p(r=\hat r_{ij}\mid i,j)
p(l_i\mid b_i)p(l_j\mid b_j).
\]

PredCls 的 object scores 为 1；SGCls/SGDet 使用 object confidence。
RSGP 只改变进入关系头的候选图，不改变该最终排序定义。

---

## 3. 实验设置

### 3.1 Dataset and tasks

| Item | Setting |
|---|---:|
| Dataset | STAR fixed split |
| Box representation | OBB |
| Foreground object/predicate classes | 48 / 58 |
| Train/val/test images | 771 / 245 / 264 |
| Train relation protocol | Single-label graph-constrained |
| Pair budget | top-10,000 |

| Task | Boxes | Object labels | Predicates |
|---|---|---|---|
| PredCls | GT | GT | predicted |
| SGCls | GT | predicted | predicted |
| SGDet | predicted | predicted | predicted |

### 3.2 Controlled PredCls rows

| Row | Relation adjacency | GNN/update | LA | Inference filter |
|---|---|---|:---:|---|
| Base | unified | current RCA GNN |  | PPG |
| D | shared-subject/shared-object | same current RCA GNN |  | PPG |
| DL | shared-subject/shared-object | same current RCA GNN | ✓ | PPG |
| Full | exact DL checkpoint | exact DL checkpoint | ✓ | statistical RSGP |

Base/D/DL 均只加载
`pretrained/OBB_swin_L_OBD.pth` 的 frozen detector，relation parameters 按各自
配置的 random/GloVe 规则从头训练。三个训练行使用相同数据、batch、optimizer、
预算、validation filter 和 checkpoint criterion，并使用共同的
6850-compatible initialization/classifier。因此 Base→D→DL→Full 构成严格
单变量消融链。完整 source RPCM 只作为独立 audit，不占用主表消融行。

另保留一次性决策实验 `DL-OriginalHead`：使用与 DL 相同的 current
dual-view/all-layer RCA 和 auxiliary LA，但将 GNN 后半段替换为 2.4.1
中的 source gated/KMeans prototype classifier。其输出目录为
`outputs/paper_submission/predcls/dual_la_original_head/`。在 validation
证明有效前，该行不进入主表，也不会替换现有 DL checkpoint。

### 3.3 Optimization

| Experiment | Batch | Optimizer | Base LR | Stop | LR milestones | Selection |
|---|---:|---|---:|---:|---|---|
| PredCls Base/D/DL | 16 | SGD | 0.016 | early stop; 220-epoch cap | 10,000 iters | best val HMR |
| SGCls Dual+LA | 16 | SGD | 0.001 with configured batch scaling | 15,000 iters | 8,000, 13,000 | best val HMR |
| SGDet Dual+LA, RPCM budget | 2 | SGD | 0.001 configured, 0.002 effective | 5,000 iters | 3,000, 4,000 | final endpoint |

共同设置为 momentum 0.9、weight decay \(10^{-4}\)、gradient clipping 5.0。
PredCls warmup 500 iterations。Base/D/DL 沿用已建立的收敛区间，从 epoch 120
开始 validation；完整 source classifier audit 以及一次性 `DL-OriginalHead`
决策实验尚无可靠最优区间，因此从 epoch 70 开始重新搜索。
所有行均每 2 epoch 验证并保存 improved best checkpoint。以 \(HMR@2000\)
为早停指标，但 patience 最早从 epoch 120 才开始累计，避免 audit 在早期搜索
阶段因短平台提前退出；此后连续 10 次 validation（默认约 20 epochs）没有
严格提升时终止，`min_delta=0`。220 epoch 仅为安全上限。早期 validation
同样参与 best checkpoint 选择，最终仍使用 validation 上的
`model_best_HR.pth`。

早停状态同时写入新 checkpoint。对机制加入前产生的 checkpoint，resume 会从
同一输出目录的 `validation_history.jsonl` 重建历史 best 和 patience，因此不会
为了接入早停而重复已经无收益的 epoch。

本实验不显式固定随机 seed；每个配置运行一次并报告单次结果，不写
mean±std。test 不用于选 epoch 或 RSGP 超参数。

### 3.4 Data-use and evaluation protocol

```text
train: optimization and construction of RSGP structural statistics
val: checkpoint selection and RSGP mode/protected-pool selection
test: final main results and pre-declared component ablations after freezing
```

PredCls/SGCls 使用 validation-selected `model_best_HR.pth`；固定预算 SGDet
使用 `model_last.pth`。RSGP validation 接受规则为：

\[
HMR^{RSGP}_{2000}>HMR^{PPG}_{2000},
\qquad
R^{RSGP}_{2000}\ge R^{PPG}_{2000}-0.002.
\]

未通过时不能用旧 manual RSGP 结果替代 statistical RSGP。

#### Val-only model and RSGP selection

PredCls 的正式选型分为两个互不混用的阶段：

1. **Checkpoint selection**：Base、D、DL 训练期间只在 val 上计算
   `HMR@2000`，分别保存各自的 `model_best_HR.pth`。PPG 是三行共同的
   validation filter，test 指标不得用于选 epoch、早停或恢复训练。
2. **RSGP selection**：固定 DL 的 `model_best_HR.pth` 后，仅在 val 上运行
   PPG、PPN 和 statistical RSGP grid。脚本先按上述 R/HMR 接受规则排除不合格
   配置，再按 `HMR@2000`、`R@2000` 的字典序选择唯一方案。

标准命令和产物为：

```bash
bash scripts/research/select_predcls_rsgp_on_val.sh
```

```text
outputs/paper_submission/predcls/dual_la/rsgp_val_selection/
├── comparison.json  # all validation-only candidates
└── selection.json   # criterion, eligible cases and the unique selected case
```

`legacy_manual` 可以出现在 comparison 中作为历史参照，但不会进入 statistical
候选集合。`selection.json.selected=null` 表示 statistical RSGP 未通过接受规则，
此时 Full 不得用 test 或旧 manual 结果补选。

选型完成后冻结 DL checkpoint、structural-prior 文件、`RSGP_MODE`、
`RSGP_PPG_PROTECTED_TOPK` 和 top-10,000 budget，再对 test 各运行一次 PPG 与
所选 RSGP。任何在生成 `selection.json` 前得到的 test 结果只能标记为 provisional；
不能据此继续调整 RSGP 后再作为 val-only 结果报告。

当前 val-only 选型已经完成，`selection.json` 冻结为
`rsgp_hybrid_9000_1000`：

| Validation filter | Statistical candidate | R@2000 | mR@2000 | HMR@2000 | GT-pair coverage | Decision |
|---|:---:|---:|---:|---:|---:|---|
| PPG 10000 | — | 60.85 | 39.80 | 48.12 | 72.88 | acceptance reference |
| PPN 10000 | — | 60.14 | 38.67 | 47.07 | **86.28** | proposal reference only |
| RSGP RS-only | ✓ | 62.77 | 41.53 | 49.99 | 74.43 | eligible |
| RSGP PPN-graph | ✓ | **63.31** | 42.02 | 50.52 | 76.53 | eligible |
| RSGP Hybrid 7000/3000 | ✓ | 63.03 | 42.26 | 50.60 | 70.66 | eligible |
| RSGP Hybrid 8000/2000 | ✓ | 63.01 | 42.24 | 50.58 | 70.66 | eligible |
| **RSGP Hybrid 9000/1000** | ✓ | 63.02 | **42.31** | **50.63** | 70.66 | **selected** |

这里的 `Statistical candidate` 只表示该行进入预先声明的 statistical RSGP
候选集合；PPG 是接受阈值，PPN 是独立 proposal reference。所有五个 RSGP
candidate 都满足 `HMR@2000 > PPG` 且 `R@2000 >= PPG - 0.2 points`，再按
`HMR@2000` 优先、`R@2000` 次优选择。因此 PPN-graph 虽有最高 R，最终仍由
HMR 最高的 Hybrid 9000/1000 胜出。历史 `legacy_manual 8000/2000` 只保存在
`comparison.json` 中供审计，不参与 statistical selection。

相对 PPG，选中配置在 val 上取得 +2.17 R、+2.51 mR 和 +2.51 HMR
points，因此通过接受规则。后续 PredCls、SGCls 和 SGDet 的 paper-facing RSGP
test 默认固定 `RSGP_MODE=HYBRID`、`RSGP_PPG_PROTECTED_TOPK=9000`；不再使用
历史 8000/2000 默认值。

在冻结 9000/1000 pool split 后，又在 **val** 上执行了一次预先隔离的结构角色
保留/删除检查，用于解决 test component table 中方向不一致的问题：

| Val variant | R@2000 | mR@2000 | HMR@2000 | GT-pair coverage |
|---|---:|---:|---:|---:|
| Full statistical RSGP | **63.02** | **42.31** | **50.63** | 70.66 |
| w/o contextual/alignment/connectivity roles | 62.82 | 42.11 | 50.42 | **72.35** |

Full 分别高出 +0.20 R、+0.21 mR 和 +0.21 HMR points。因而最终版本保留
train-derived contextual-region、directional-alignment 和
relational-connectivity 三类软结构角色。该结论只由 val 决定；test 上
`w/o structural roles` 的 +0.06 HMR 波动不用于反向修改方法。最终冻结记录位于
`outputs/paper_submission/predcls/dual_la/rsgp_finalization_val/final_selection.json`。

最终 paper-facing 配置为：statistical role mode、Hybrid 9000/1000、总预算
10,000、PPN/geometry/three structural roles/rarity/degree control/semantic
capacity 全部启用。项目内 PredCls、SGCls、SGDet 与公开评估脚本的默认 protected
pool 已统一为 9000；7000/8000 仅保留为历史 grid 结果。

### 3.5 Main metrics

对第 \(n\) 张图的 GT 有向 pair 集 \(G_n\) 和 top-\(K\) 命中集合
\(M_n^K\)：

\[
R@K=\frac{1}{T}\sum_{n=1}^{T}
\frac{|M_n^K|}{\max(|G_n|,1)}.
\]

对前景 predicate \(c\)：

\[
R_c@K=\frac{1}{|T_c|}
\sum_{n\in T_c}\frac{|M_{n,c}^K|}{|G_{n,c}|},
\]

\[
mR@K=\frac{1}{58}\sum_{c=1}^{58}R_c@K,
\]

\[
HMR@K=
\frac{2(R@K)(mR@K)}{R@K+mR@K}.
\]

主表报告 \(K=1500,2000\)，补充材料报告 \(K=1000\)。所有行使用同一
graph-constrained evaluator。

### 3.6 Candidate-graph metrics

同 checkpoint 的 PPG、PPN、RSGP 还报告：

- **GT-pair coverage**：逐 GT relation row 检查其 subject-object pair
  是否在最终 candidate graph 中，再以全部 GT relation rows 为分母；因此同一
  pair 若有多个 GT predicates，会按 relation row 重复计权；
- **node coverage**：至少出现在一条 candidate edge 中的实体比例；
- **degree Gini**：每图有向边 incidence degree 的 Gini，再对图像平均；
- **maximum degree**：整个 split 中单个节点的最大 incidence degree；
- **label-pair entropy**：每图候选边的 \((l_s,l_o)\) 分布熵，再对图像平均。

这些指标描述候选图结构，不等同于 relation prediction accuracy。尤其是
更高 GT-pair coverage 并不保证更高 triplet R/mR/HMR。

### 3.7 Candidate-pressure analysis

pressure 使用 **Semantic Filter 后、top-10,000 前** 的候选数分组：

| Pressure group | Semantic-filtered candidate count |
| :--- | ---: |
| No truncation | ≤ 10,000 |
| Low overload | 10,001–20,000 |
| Medium overload | 20,001–50,000 |
| High overload | > 50,000 |

`no_truncation` 仅表示 pair proposal 不需要裁剪；最终 evaluator 仍按
triplet top-1000/1500/2000 排序。分组 R 是组内 image recall 的平均值；
分组 mR 仍对固定 predicate vocabulary 做 macro aggregation，因此会同时受
组内 predicate 分布影响。不同 pressure 组之间不能直接解释为“模型性能随候选
规模单调下降”；关键受控证据是在同一组内比较 PPG 与 RSGP。

最终 DL checkpoint 上的同权重 PPG/RSGP pressure 对比已经完成：

| Pressure group | Images | R: PPG → RSGP (%) | mR: PPG → RSGP (%) | HMR: PPG → RSGP (%) |
| :--- | ---: | ---: | ---: | ---: |
| No truncation | 160 | 74.01 → 74.01 | 52.10 → 52.10 | 61.15 → 61.15 |
| Low overload | 28 | 68.98 → **69.38** | 34.17 → **36.16** | 45.70 → **47.54** |
| Medium overload | 32 | 65.72 → **69.29** | 25.16 → **27.43** | 36.39 → **39.30** |
| High overload | 44 | 41.81 → **47.80** | 20.01 → **23.17** | 27.07 → **31.21** |

无需裁剪时两种方法完全一致；从 low 到 high overload，RSGP 的 HMR 增益分别
为 +1.84、+2.91 和 +4.14 points。这是固定 top-10,000 预算下最直接的压力证据，
且不要求不同 pressure group 之间的绝对性能单调变化。来源分别为 DL/PPG 与
Full RSGP 的正式 test JSON，因此无需新增 pressure 实验。

---

## 4. 结果表

### 4.1 Controlled PredCls ablation（paper main）

| Row | R@1500 | R@2000 | mR@1500 | mR@2000 | HMR@1500 | HMR@2000 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| Base (Unified) | **68.63** | **70.39** | 39.77 | 40.98 | 50.36 | 51.80 | `outputs/paper_submission/predcls/unified/test/ppg/test_metrics.json` |
| D | 68.12 | 69.68 | 41.02 | 42.18 | 51.21 | 52.55 | `outputs/paper_submission/predcls/dual/test/ppg/test_metrics.json` |
| DL | 65.46 | 67.11 | 41.99 | 43.42 | 51.16 | 52.72 | `outputs/paper_submission/predcls/dual_la/test/ppg/test_metrics.json` |
| Full | 67.23 | 68.58 | **43.82** | **45.38** | **53.06** | **54.62** | selected 9000/1000; `outputs/paper_submission/predcls/dual_la/rsgp_component_test/rsgp_full/test_metrics.json` |

表中 Base→D→DL→Full 构成严格受控消融链。Base 是原实验目录中的 Unified
checkpoint，不是完整 SGG-ToolKit source predictor；source RPCM 仅在审计实验中
报告。串行 suite 会自动跳过已完成行、续训中断行并依次运行。

当前 test 结果在 $K=2000$ 下的逐步变化为：

| Transition | ΔR | ΔmR | ΔHMR | Evidence |
|---|---:|---:|---:|---|
| Base → D | -0.71 | +1.20 | +0.75 | dual view 改善类别均衡和综合指标，但并未提高总体 R |
| D → DL | -2.58 | +1.24 | +0.17 | LA 明显偏向 macro recall，HMR 仅小幅提高且存在 R 代价 |
| DL → Full | **+1.47** | **+1.96** | **+1.90** | 同一 checkpoint 只替换 PPG 为 statistical RSGP，三项指标同时提高 |
| Base → Full | -1.81 | +4.40 | +2.82 | 完整方法主要提升 mR/HMR，而非保持最高 micro-style R |

因此，这组结果支持以下较窄且可验证的论点：dual view 和 LA 逐步改善
class-balanced recall；statistical RSGP 在不重训关系模型的条件下同时恢复 R、
提高 mR，并取得最高 HMR。它**不支持**“dual view/LA 的每一步都不降低 R”或
“Full 在所有指标上都优于 Base”这两种更强表述。

DL checkpoint 上的同权重 candidate-graph 对比如下：

| Filter | GT-pair coverage | Node coverage | Degree Gini | Maximum degree | Label-pair entropy | R@2000 | mR@2000 | HMR@2000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPG | **79.89** | 99.28 | **0.1637** | **252** | 1.8587 | 67.11 | 43.42 | 52.72 |
| statistical RSGP | 75.33 | **99.42** | 0.1694 | 740 | **2.0263** | **68.58** | **45.38** | **54.62** |

RSGP 在较低 GT-pair coverage 下获得更高的 downstream R/mR/HMR，直接支持
“pair coverage 不是候选图质量的充分条件”；更高的 label-pair entropy 也支持
其增加语义类型多样性的论点。不过，当前结果的 degree Gini 和 maximum degree
均未优于 PPG，不能用它证明最终图的度数更均衡。实现会在受约束的两轮选择仍
不足 top-10,000 时执行无约束补齐，因此论文应将 degree/semantic capacity
表述为 greedy selection 中的软容量控制，而不是最终图必然满足的硬约束。

按 3.4 节预先写下的接受规则，validation 选择了 Hybrid 9000/1000；上述 Full
已使用相同 DL checkpoint 和冻结配置完成 test。相对 PPG，Full 在 test 上取得
+1.47 R、+1.96 mR 和 +1.90 HMR points。不得再根据 test 结果修改 RSGP
超参数。

### 4.2 Historical filter evidence

以下三行来自同一个历史 HPRC checkpoint，只用于证明“pair coverage 不是充分
目标”。原 checkpoint 已不在当前 workspace，因此不能作为投稿主结果：

| Filter | GT-pair coverage | R@2000 | mR@2000 | HMR@2000 |
|---|---:|---:|---:|---:|
| PPG | 79.89 | 69.19 | 44.17 | 53.92 |
| PPN | **91.24** | 68.88 | 43.43 | 53.27 |
| RSGP-v1/manual Hybrid 8000/2000 | 75.98 | **71.08** | **45.96** | **55.82** |

### 4.3 Cross-task comparison with RPCM

以下结果使用 STAR/SGG-ToolKit 的 filter-label 协议，直接用于与 RPCM 报告值
进行三任务完整对比：

| Task/method | R@1500 | R@2000 | mR@1500 | mR@2000 | HMR@1500 | HMR@2000 |
|---|---:|---:|---:|---:|---:|---:|
| STAR SGCls RPCM | 51.29 | 52.72 | 30.04 | 30.85 | 37.89 | 38.92 |
| Ours SGCls, PPG | 54.17 | 55.47 | 30.71 | 31.63 | 39.20 | 40.29 |
| **Ours SGCls, RSGP** | **56.75** | **57.78** | **32.90** | **33.75** | **41.65** | **42.61** |
| STAR SGDet RPCM | 27.23 | 28.50 | 11.53 | 12.07 | 16.20 | 16.96 |
| Ours SGDet, PPG | 32.51 | 33.21 | 13.79 | 14.28 | 19.36 | 19.97 |
| **Ours SGDet, RSGP** | **32.98** | **33.65** | **14.11** | **14.55** | **19.76** | **20.31** |

在 @2000 下，相对 RPCM，Ours+RSGP 在 SGCls 上提高 +5.06 R、+2.90 mR、
+3.69 HMR，在 SGDet 上提高 +5.15 R、+2.48 mR、+3.35 HMR。固定本项目模型
后将 PPG 替换为 RSGP，SGCls 再提高 +2.31 R、+2.11 mR、+2.32 HMR；SGDet
提高 +0.44 R、+0.27 mR、+0.34 HMR。

额外的 predicted-label filtering 结果如下，用于展示不使用 GT/matched-GT
filter labels 时的性能：

| Task/method | R@1500 | R@2000 | mR@1500 | mR@2000 | HMR@1500 | HMR@2000 |
|---|---:|---:|---:|---:|---:|---:|
| SGCls, PPG | 45.79 | 46.78 | 25.50 | 26.09 | 32.76 | 33.50 |
| **SGCls, RSGP** | **46.69** | **47.40** | **26.06** | **26.56** | **33.45** | **34.05** |
| SGDet, PPG | 29.21 | 30.26 | 11.62 | 12.39 | 16.63 | 17.58 |
| **SGDet, RSGP** | **29.84** | **30.90** | **12.10** | **12.85** | **17.22** | **18.15** |

正式结果来源为
`outputs/paper_submission/{sgcls,sgdet_rpcm_budget}/test/`；历史 manual
RSGP 结果不再用于该表。

### 4.4 Statistical RSGP component ablation（test, frozen protocol）

| Variant | PPN | Geometry | Statistical roles | Degree control | Semantic capacity | Rarity | R@2000 | mR@2000 | HMR@2000 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 68.58 | **45.38** | 54.62 |
| w/o PPN completion |  | ✓ | ✓ | ✓ | ✓ | ✓ | 68.35 | 45.24 | 54.45 |
| w/o geometry | ✓ |  | ✓ | ✓ | ✓ | ✓ | 68.66 | 45.24 | 54.54 |
| w/o structural roles | ✓ | ✓ |  | ✓ | ✓ | ✓ | **68.80** | 45.36 | **54.68** |
| w/o degree control | ✓ | ✓ | ✓ |  | ✓ | ✓ | 68.37 | 45.11 | 54.36 |
| w/o semantic capacity | ✓ | ✓ | ✓ | ✓ |  | ✓ | 67.85 | 44.33 | 53.62 |
| w/o rarity support | ✓ | ✓ | ✓ | ✓ | ✓ |  | 68.31 | 45.25 | 54.43 |

这些行不是候选超参数，而是预先声明的 remove-one-component 诊断。Hybrid
9000/1000 已由 val 冻结后，各行在 test 上只运行一次，统一使用同一 DL
checkpoint；结果不得反向用于修改组件、权重或容量设置。`NO_RS` 同时删除多个
模块，不属于该受控表。`w/o geometry` 使用新增的独立开关，仅关闭通用 OBB
geometry utility，仍保留 statistical structural roles，因而不与 `NO_RS` 混淆。

相对对应 ablation，Full 的 @2000 增量为：

| Restored component | ΔR | ΔmR | ΔHMR | Interpretation |
|---|---:|---:|---:|---|
| PPN completion | +0.22 | +0.14 | +0.17 | 小幅 test-side 正贡献 |
| OBB geometry utility | -0.08 | +0.14 | +0.08 | 轻微从总体 R 重分配到 macro recall |
| statistical structural roles | -0.22 | +0.02 | -0.06 | 仅轻微重分配到 macro recall，不能声称提高综合性能 |
| degree control | +0.21 | +0.27 | +0.26 | test remove-one 中为正 |
| semantic capacity | +0.73 | +1.05 | +0.99 | 最大的 test-side removal effect |
| rarity support | +0.27 | +0.14 | +0.18 | 小幅 test-side 正贡献 |

为了避免把单次 test remove-one 结果误写成稳定的 component selection，下面进一步
列出已有 validation 证据。所有数值均为 `Full − reference/ablated` 的
HMR@2000 百分点；正数表示 Full 更高：

| Comparison | Val ΔHMR | Test ΔHMR | Evidence boundary |
|---|---:|---:|---|
| Full RSGP − PPG | **+2.51** | **+1.90** | 整体候选图方法在两个 split 上方向一致 |
| Full − w/o PPN completion | -0.05 | +0.17 | 极小且方向反转；只能称为多源互补设计，不能声称独立稳定增益 |
| Full − w/o geometry | ≈0.00 | +0.08 | val 基本中性，test 上仅小幅改善 macro balance |
| Full − w/o structural roles | **+0.21** | -0.06 | 按 val 保留的辅助结构证据；test 上存在轻微反转 |
| Full − w/o degree control | +0.04 | +0.26 | 两个 split 同方向，val 影响很小 |
| Full − w/o semantic capacity | **+1.50** | **+0.99** | 两个 split 中均为最强的选择约束 |
| Full − w/o rarity support | +0.19 | +0.18 | 小幅但跨 split 高度一致 |

除前述 structural-role finalization 外，degree/capacity/rarity/geometry 的 val
行是在冻结最终配置后补做的 **post-freeze consistency audit**，只用于限定组件
结论，不构成新的模型选择，也不触发重新测试。
`w/o PPN completion` 的 val 结果来自已有一致性检查，但它不在预先声明的 pool/mode
selection grid 中，因此没有用于事后改变 Full 定义。表中真正稳定的主结论是：
**完整 RSGP 相对 PPG 的 HMR 增益在 val 和 test 上均成立**。组件层面的结论应更
具体：semantic capacity 是两个 split 中最强且稳定的选择约束；rarity support
提供高度一致的小幅增益；degree control 同方向但 val 影响较小；geometry 在 val
近似中性、test 上略改善 HMR；PPN completion 和 structural roles 则存在轻微
split reversal，不能作为独立稳定增益来表述。

所有 variant 在 160 张无需截断的图上结果完全一致，说明这些模块只改变
top-10,000 超载场景，符合设计边界。结构角色在 low overload 上有轻微正贡献，
并在 medium/high overload 中略增 mR，但同时降低 R，导致全测试集 HMR 比
`w/o structural roles` 低 0.06 points。这个 test remove-one 结果不能用于删除
组件；上面的独立 val finalization 显示 Full 高 0.21 HMR points，因而最终版本
仍保留这些角色。论文应将其谨慎描述为 **validation-selected auxiliary structural
evidence**，而不是声称它在每个 split 上都有稳定的独立增益。

### 4.5 Per-predicate supplement and qualitative cases

完整 58 类 Base/D/DL/Full 的 @2000 recall 表已经提取到
[per_predicate_supplement.md](per_predicate_supplement.md)。机器可读版本位于：

```text
outputs/paper_submission/supplement/per_predicate_ablation.{csv,json}
outputs/paper_submission/supplement/per_predicate_filter_ppg_vs_rsgp.{csv,json}
```

在固定 DL checkpoint 上只将 PPG 替换为 statistical RSGP 时，@2000 下
58 类中 32 类提高、16 类不变、10 类下降。变化最大的代表类如下；`Pair cov.`
是该 predicate 的 GT relation row 最终进入候选图的比例：

| Predicate | Count | PPG R | RSGP R | ΔR | PPG pair cov. | RSGP pair cov. |
|---|---:|---:|---:|---:|---:|---:|
| around | 152 | 33.77 | **55.06** | **+21.29** | 65.13 | 98.68 |
| parallelly docked at | 2891 | 64.39 | **76.92** | **+12.53** | 82.01 | 92.94 |
| approach | 930 | 40.01 | **51.54** | **+11.53** | 72.04 | 94.84 |
| run along | 219 | 73.61 | **83.86** | **+10.25** | 80.82 | 95.43 |
| docking at the different dock with | 525 | 30.29 | **40.36** | **+10.07** | 72.19 | 71.05 |
| parking in the different apron with | 18104 | **45.57** | 39.26 | -6.31 | 46.53 | 24.84 |
| running along the different taxiway with | 466 | **29.23** | 24.05 | -5.18 | 33.26 | 18.45 |
| in the different parking with | 509 | **47.08** | 42.17 | -4.91 | 87.03 | 69.55 |

这组结果应作两点克制解释。第一，多数大幅提高类别同时获得了更高候选覆盖，
说明 RSGP 的收益来自固定预算下的候选重分配；第二，候选覆盖下降明显的 apron/
taxiway 类会退化，因此不能声称 RSGP 对每个 predicate 都提高。与此同时，
`docking at the different dock with` 在 pair coverage 略降时仍提高 10.07 points，
再次说明 pair coverage 不是 downstream relation quality 的充分统计量。
Full 还把 DL 的零召回类别数从 5 降到 3：`randomly docked at` 从 0 提高到
1.50，`not working on` 从 0 提高到 1.92；仍为零的三类中有两类在 test 中仅有
2 和 13 个样本，因此不应只按“零类数量”评价整体方法。

当前案例选择已经固定为：

| Use | Image ID | Pressure | Semantic pairs | GT pairs | PPG Triplet R | RSGP Triplet R | Δ |
|---|---:|---|---:|---:|---:|---:|---:|
| main success | 440 | overload medium | 46,048 | 446 | 54.71 | **84.98** | **+30.27** |
| second filtered success | 748 | overload medium | 45,332 | 240 | 17.08 | **50.83** | **+33.75** |
| main failure | 235 | overload high | 63,032 | 865 | **40.58** | 30.52 | -10.06 |

主文使用 `440/748/235` 三张：分别展示机场枢纽场景中命中翻倍且输出收缩、
多组件场景中的显著改善，以及 semantic-capacity 重分配造成的明确失败。三张图的
Semantic Filter 候选数均超过 10,000，因此 PPG 和 RSGP 的 top-10,000
过滤都被实际执行。表中的 `Triplet R` 来自同一 relation checkpoint 的最终
graph-constrained top-2,000 预测，不是候选 pair coverage。具体实体/关系构成、
源图路径和缩略图位于：

```text
outputs/paper_submission/supplement/qualitative_cases.{md,json}
outputs/paper_submission/supplement/case_thumbnails/
```

复现提取过程：

```bash
python tools/extract_paper_supplement.py --render-thumbnails
```

原始卫星图过大，不应直接在底图上叠加全部关系边。下面的一键脚本仅推理
`440/748/235`，并输出“卫星裁剪图 + GT scene graph + PPG scene graph + RSGP
scene graph”。三张图使用完全一致的节点坐标、节点 ID 和类别颜色；裁剪区域内
全部 GT 实体均被保留。图中不标注具体 predicate 名称，以免文字遮挡遥感场景中
密集的小目标；灰色表示 GT 关系，绿色为匹配 GT 的实际输出，蓝色为 RSGP-only
正确输出，紫色为 PPG-only 正确输出。蓝色或紫色虚线圈叉表示“该正确关系只被
另一方法预测，本面板缺失”，它只是跨方法对照参考，不是本面板的模型输出；
红色虚线则是本方法实际输出但未匹配现有 GT 的预测。节点布局在保持原始空间
投影的前提下强制最小间距。仅对 image `440` 使用可视化层面的局部微调：下方
枢纽轻微下移，上方扇区横向展开，左侧扇区向外平移，右侧扇区保持锚定；b/c/d
严格共享调整后的坐标，预测与指标不变。该密集案例的三位节点编号使用自适应
字号，稀疏案例保持原排版。同一
有向实体对上的多个 predicate 在图中合并为一条结构边，反向实体对则使用
分离的浅曲线路由。所有 GT 实体和全部有向 GT pair 均按与预测面板相同的线宽
绘制，完整 pair 列表同时保存在 artifact 和 manifest 中。弧线路由只根据各面板
实际显示的边计算，未显示的反向预测不会使孤立可见边发生弯曲。精确 triplet
数和结果仍保存在标题及 manifest 中：

```bash
bash scripts/research/export_paper_qualitative_cases.sh
```

结果位于：

```text
outputs/paper_submission/qualitative_figures/scene_graphs/
  0440_spatial_scene_graph.png
  0748_spatial_scene_graph.png
  0235_spatial_scene_graph.png
  scene_graph_manifest.json
```

预测图绘制局部全部 GT-matched top-2,000 输出，并仅保留最高分的 8 条
GT-unmatched 输出以控制可读性；manifest 保存未截断的局部预测数和匹配数。
脚本会拒绝未真正发生候选截断的案例，不训练模型，也不改变最终数值结果。

### 三张定性图的具体解读

**Image 440：机场枢纽场景中的主要成功案例。** 裁剪区域包含 68 个实体和
105 个 GT triplet（对应 105 个有向实体对），全部在 GT 面板中绘制。该区域
主要由 terminal/apron 与分布在
周围的 airplane、boarding bridge 和 taxiway 组成。PPG 在 562 个局部输出中
命中 40 个，RSGP 在仅 305 个局部输出中命中 80 个：输出数量减少约 46%，正确
命中数却翻倍；image-level Triplet R 从 54.71% 提高到 84.98%。绿色与蓝色边
显示 RSGP 更完整地恢复了围绕 terminal/apron 枢纽的局部关系，同时避免 PPG
产生的大量未匹配输出。因此该案例同时体现候选图的有效性与选择性，而不只是
候选数量变化。

**Image 748：候选数量不等于下游质量的第二个成功案例。** 该区域包含 26 个实体
和 22 个 GT 有向 pair，并呈现多个空间分离的局部连通分量。PPG 产生 174 个
局部输出但只命中 4 个，RSGP 将局部输出减少到 108 个的同时命中 18 个；命中数
提高 4.5 倍，image-level Triplet R 从 17.08% 提高到 50.83%。图中 RSGP 保留了
更多与 GT 局部组件一致的边，并减少若干跨组件的长距离未匹配输出。这是论文中
支撑“更高 pair/output 数量并不保证更优关系图”的最直观证据。需要注明，红色
虚线只展示分数最高的 8 个 GT-unmatched 输出，而且 STAR 标注并非穷尽，因此
不能把红边数量直接解释为精确的 false-positive 数。

**Image 235：同质密集子图上的明确失败。** 裁剪区域只有 6 个 tank 实体，但
形成 16 个 GT 有向 pair。PPG 在 24 个局部输出中命中全部 16 个，RSGP 在 15 个
输出中仅命中 1 个；image-level Triplet R 从 40.58% 降至 30.52%。紫色实线表示
PPG-only 正确输出，RSGP 面板中的带圈叉虚线表示相应 pair 缺失。该现象与图容量
机制在全图高负载条件下将预算从高度重复的同质局部模式中移走相一致，说明强调
语义多样性和度数平衡可能误伤真实的密集同类关系。因此该失败案例应与两个成功
案例共同报告，并在 limitation 中作为未来自适应 capacity 控制的动机。

---

## 5. 实验命令

### 5.1 PredCls controlled suite

```bash
# Base(Unified) → D → DL 串行；完成一行后自动用 best val-HMR checkpoint 跑 PPG test
bash scripts/research/run_paper_predcls_suite.sh

# 在 DL checkpoint 上用 val 选择 statistical RSGP
bash scripts/research/select_predcls_rsgp_on_val.sh

# 用 val 选中的 9000/1000 配置完成正式 Full test
RUN_BACKGROUND=0 bash scripts/research/eval_predcls_rsgp_ablation.sh FULL

# 汇总冻结后的结果
python tools/summarize_paper_experiments.py
```

单行训练：

```bash
bash scripts/research/run_predcls_minimal_ablation.sh BASE
bash scripts/research/run_predcls_minimal_ablation.sh D
bash scripts/research/run_predcls_minimal_ablation.sh DL

# 可选 source-RPCM 审计，不属于主消融链
bash scripts/research/run_predcls_minimal_ablation.sh SOURCE
```

suite 以 `<row>/test/ppg/test_metrics.json` 判定完成；重启时跳过完成行并从
`model_last.pth` 自动续训中断行。`CASES=BASE,D` 可限制队列。

### 5.2 Statistical prior and RSGP ablation

```bash
bash scripts/build_rsgp_structural_prior.sh

bash scripts/research/eval_predcls_rsgp_ablation.sh FULL
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_PPN
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_GEOMETRY
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_STRUCTURE
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_DEGREE
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_QUOTA
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_RARITY
```

### 5.3 SGCls and SGDet

```bash
# SGDet train/val/test cache 必须使用同一 v5 detector hash
SPLITS=val OVERWRITE=0 bash scripts/build_sgdet_detection_cache.sh

bash scripts/research/run_star_sgcls_experiment.sh
bash scripts/research/run_star_sgdet_rpcm_budget.sh
bash scripts/research/eval_paper_cross_task_suite.sh
```

cross-task suite 生成：

```text
legacy: SGCls gt filter labels; SGDet matched_gt filter labels
pred:   strict predicted filter labels
```

PPG/RSGP 对比必须使用完全相同的 task checkpoint、detection cache、
label-source protocol 和 evaluator。

---

## 6. 结果来源与回填规则

### 6.1 Authoritative outputs

```text
outputs/paper_submission/
├── predcls/
│   ├── unified/       # paper Base
│   ├── dual/
│   ├── dual_la/
│   └── base_original/ # optional source audit
├── rpcm_audit/
├── sgcls/
└── sgdet_rpcm_budget/
```

旧开发结果均在 `outputs_old/`，只能填入明确标记为 historical/archive 的表。

标准 JSON 路径中的值为 \([0,1]\)，论文表乘以 100：

```text
R@K   <- metrics.R[str(K)]  * 100
mR@K  <- metrics.mR[str(K)] * 100
HMR@K <- metrics.HR[str(K)] * 100
```

逐 predicate recall 从同次运行的 `test.log` 中
`Per-Relation Recall` 读取。不要混用另一个 checkpoint 或 filter 的日志。
新 evaluator 还会把同样的数据写入 JSON 的 `per-predicate-recall`、
`predicate-counts` 和 `per-image`；旧正式结果仍由上述提取工具兼容解析日志。

### 6.2 PredCls main provenance

| Row | Result JSON |
|---|---|
| Base (Unified) | `outputs/paper_submission/predcls/unified/test/ppg/test_metrics.json` |
| D | `outputs/paper_submission/predcls/dual/test/ppg/test_metrics.json` |
| DL | `outputs/paper_submission/predcls/dual_la/test/ppg/test_metrics.json` |
| Full | `outputs/paper_submission/predcls/dual_la/rsgp_component_test/rsgp_full/test_metrics.json` |
| DL + PPN diagnostic | `outputs/paper_submission/predcls/dual_la/test/ppn/test_metrics.json` |
| Source RPCM audit | `outputs/paper_submission/predcls/base_original/test/ppg/test_metrics.json` |

### 6.3 Cross-task provenance

| Row | Result JSON |
|---|---|
| SGCls PPG | `outputs/paper_submission/sgcls/test/legacy/ppg/test_metrics.json` |
| SGCls statistical RSGP | `outputs/paper_submission/sgcls/test/legacy/rsgp_statistical/test_metrics.json` |
| SGDet PPG | `outputs/paper_submission/sgdet_rpcm_budget/test/legacy/ppg/test_metrics.json` |
| SGDet statistical RSGP | `outputs/paper_submission/sgdet_rpcm_budget/test/legacy/rsgp_statistical/test_metrics.json` |

strict `pred` 结果位于相同 task 目录的 `test/pred/` 分支。

### 6.4 Historical provenance

```text
STAR paper:
  Star A first-ever dataset and a large-scale benchmark for scene graph
  generation in large-size satellite imagery.pdf

RPCM-6850:
  checkpoint: /home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth
  replay: outputs_old/paper_ablation_predcls/B_dual_rca_6850_ppg/

Historical Dual+LA/manual RSGP:
  outputs_old/paper_ablation_predcls/
  outputs_old/rsgp_grid/

Historical SGCls/SGDet:
  outputs_old/paper_cross_task/
  outputs/star_sgdet_obb_dual_la_rpcm_budget_eval_last_{ppg,rsgp}/

Recovered compatible checkpoints:
  outputs/star_sgcls_obb_dual_la_train/
  outputs/star_sgdet_obb_dual_la_rpcm_budget/
```

`6850_4135.pth` 的名称记录历史 iteration 17,600 的
R@1500/mR@1500=`0.6850/0.4135`；当前项目 replay 为
`0.6812/0.4102`。两者不能静默互换。

### 6.5 Final completion audit

| Group | State | Remaining action |
|---|---|---|
| PredCls Base/D/DL | main test metrics complete | none |
| PredCls Full | val-selected Hybrid 9000/1000 test complete | none |
| RSGP pool/role selection | val grid and final role decision complete | none |
| Statistical RSGP component ablation | Full plus six frozen-protocol remove-one rows complete | none |
| Candidate-graph and pressure analysis | same-checkpoint PPG/RSGP results complete | only format figures/tables |
| Source RPCM audit | PPG test complete; optional and outside main table | none |
| SGDet v5 detection cache | train/val/test complete: 771/245/264 images | none |
| Cross-task SGCls | PPG/RSGP under both filter-label protocols complete | none |
| Same-budget SGDet | PPG/RSGP under both filter-label protocols complete | none |
| Per-predicate supplement | 58-class table and machine-readable CSV/JSON complete | none |
| Qualitative prediction cases | exact actively filtered cases `440/748/235` rendered as aligned GT/PPG/RSGP scene graphs from final top-2,000 outputs | none |

严格按运行记录审计时还需注意：DL 主训练在 epoch 204 被手动终止，训练
`exit_code=143`；其 best checkpoint 来自 epoch 144，之后 29 次 validation 均未
超过该值。现有 test/RSGP 结果因此数值上可用，但若投稿材料或开源审计要求每条
训练都有干净终止状态，应从 `model_last.pth` 续跑至 early stop 或 epoch 220。
只有续跑产生新的 `model_best_HR.pth` 时，才需要重跑 DL 的 PPG、PPN、RSGP、
component 与 pressure 评估；否则现有结果保持不变。

### 6.6 Experiment completion

当前所有计划内数值实验均已完成。机器汇总中
`completion.missing_cross_task=[]`。重新生成汇总使用：

```bash
python tools/summarize_paper_experiments.py
```

无需再做 PredCls 新训练、SGCls/SGDet 新训练、RSGP 新 grid、多 seed、效率/
显存或新的 component sweep。案例渲染已由
`scripts/research/export_paper_qualitative_cases.sh` 固化；剩余工作仅为从生成结果中完成论文
图片排版，不再需要为案例选择重训模型。

---

## 7. 论文图表建议

1. **Method overview**：Semantic Filter → multi-source RSGP → constrained
   candidate graph → dual-view RCA → prototype classifier + LA；
2. **Dual-view relation graph**：同一实体作为 subject 与 object 时分别进入
   \(A_s\) 和 \(A_o\)，并对比 unified graph 的混合；
3. **Candidate graph comparison**：优先用 test image `440` 展示同一图像上的
   PPG/PPN/RSGP，附
   GT-pair coverage、degree Gini、maximum degree、label-pair entropy 和
   downstream triplet recall；
4. **Success/failure cases**：image `440/748` 展示 PPG/RSGP 均执行
   top-10,000 过滤后产生的最终正确 triplet 改善，image `235` 用相同空间节点
   布局及状态编码边展示被 semantic-capacity 重分配移除的具体正确 triplet；
   不得用 candidate coverage 线条替代模型预测。SGDet 另行区分 missing box、
   missing pair 和 wrong predicate。

类别 recall 基数差异较大，正文更适合使用按关系组整理的表格；图中只展示少量
具有代表性的 head/mid/tail 类，完整 per-predicate 表放补充材料。

---

## 8. 可直接使用的贡献段落

> 本文面向大幅遥感图像中候选实体对数量庞大、关系端点角色混叠和谓词分布不均衡
> 等问题，提出一种图感知的 OBB 场景图生成框架。首先，角色感知 dual-view RCA
> 分别在 shared-subject 与 shared-object 关系图上传播信息，避免统一邻接造成的
> 跨角色语义混合。其次，在主 prototype CE 之外引入小权重 logit-adjust
> auxiliary supervision，在不改变推理分类头的情况下加入类别先验。最后，RSGP
> 将 pair proposal 重新表述为固定预算下的最大权重有向子图构建，融合 PPG、PPN、
> OBB 几何以及从训练集自动统计的区域承载、方向排列和关系连通软角色，并通过
> degree 与 semantic-type capacity 约束候选图。STAR OBB 上的受控 PredCls
> 消融用于分别验证 RCA 更新、角色分解、LA 和 RSGP，SGCls/SGDet 的同 checkpoint
> 对比进一步验证其在不同任务设置下的适用性。
