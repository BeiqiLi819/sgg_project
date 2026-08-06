# RSGP Technical Route

> Paper-facing default: category-name-agnostic statistical RSGP
> Legacy reproduction mode: `RSGP_ROLE_MODE=legacy_manual`

## 1. Method boundary

Remote-sensing Graph-aware Pair Proposal (RSGP) is an inference-time candidate
graph constructor. It does not update the detector, PPG, PPN, RPCM, or
Dual-view RCA checkpoint.

The only data preparation is a deterministic pass over the STAR `train`
annotations to build class structural profiles, predicate frequencies, and
rarity support. This pass has no optimizer, epoch, gradient, or learned
network parameter. Validation selects the RSGP configuration; test never
updates the prior.

Given the semantic-valid directed pair set \(\mathcal E_0\), RSGP searches for
a fixed-budget graph:

\[
\max_{\mathcal E\subseteq\mathcal E_0}
\sum_{(i,j)\in\mathcal E}S_{ij},
\]

\[
|\mathcal E|\le K,\quad
d_{out}(i)\le D_o,\quad
d_{in}(j)\le D_i,\quad
n_{l_i,l_j}\le Q.
\]

The central claim is not that RSGP maximizes isolated pair recall. It builds a
candidate graph whose evidence, degree distribution, and semantic diversity
are compatible with downstream relation message passing.

## 2. Category-name-agnostic structural prior

The paper-facing implementation never matches class-name strings and never
stores a hand-written predicate ID list. A one-time preprocessing pass reads
only the training split:

```bash
bash scripts/build_rsgp_structural_prior.sh
```

Default output:

```text
pretrained/rsgp_structural_prior.json
```

For every object class \(c\), it records:

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

These are soft contextual-region, directional-alignment, and
relational-connectivity profiles. Class IDs index the profile table, but class
names never participate in scoring.

The JSON stores its train split, build-configuration hash, metadata order
hash, annotation-content signature, payload hash, profile vectors, predicate
frequencies, and rarity-pair support. Missing files, payload corruption, or
class/predicate order mismatch terminate evaluation instead of silently
reverting to manual rules.

## 3. Generic structural kernels

### 3.1 OBB geometry

The class-independent geometry proxy combines axis-aligned envelope IoU,
normalized center distance, pair compactness, and OBB-axis consistency:

\[
S^{geom}_{ij}
=0.35IoU^{env}_{ij}
+0.30e^{-\bar d_{ij}}
+0.20c^{compact}_{ij}
+0.15|\cos\Delta\theta_{ij}|.
\]

This is an axis-aligned envelope proxy, not exact rotated IoU.

### 3.2 Context association

At inference, training context profile and current image area rank form a
preliminary carrier score. RSGP retains:

\[
M=\min(128,\max(16,\lceil2\sqrt N\rceil))
\]

carrier candidates. Exact OBB containment and distance are computed only in an
\(N\times M\) workspace. Each entity is softly assigned to its best carrier,
and the pair score preserves both shared-carrier and different-carrier
possibilities.

### 3.3 Directional alignment

The class alignment profile is averaged with current OBB elongation. Its pair
gate weights:

- OBB-axis parallelism;
- displacement along the subject axis;
- displacement lateral to the subject axis.

It is evaluated for all semantic-valid pairs and has no vehicle-class gate.

### 3.4 Local connectivity

The class connectivity profile is averaged with current semantic-candidate
degree. The resulting soft gate weights normalized proximity:

\[
S^{connect}_{ij}
=\sqrt{g_i^kg_j^k}\exp(-0.5\bar d_{ij}).
\]

It has no network-class gate.

## 4. Frequency-adaptive compatibility

Fixed hard-predicate IDs are replaced with training frequency:

\[
\omega_r=(f_r+\epsilon)^{-0.5}.
\]

After normalization by the maximum active rarity weight:

\[
S^{rare}_{ij}
=\max_{r:F(l_i,l_j,r)=1}\widetilde{\omega}_r.
\]

This protects semantically compatible pairs that may express infrequent
predicates without naming those predicates.

## 5. Multi-source scoring and constrained selection

RSGP pools:

```text
PPG top-10000 precision candidates
PPN top-12000 recall-completion candidates
structural-score top-12000 candidates
```

Its final score is:

\[
S_{ij}=
w_p\hat S^{PPG}_{ij}
+w_n\hat S^{PPN}_{ij}
+w_g\hat S^{geom}_{ij}
+w_c\hat S^{context}_{ij}
+w_a\hat S^{align}_{ij}
+w_k\hat S^{connect}_{ij}
+w_rS^{rare}_{ij}
+w_d\hat S^{balance}_{ij}.
\]

Defaults:

```text
(wp, wn, wg, wc, wa, wk, wr, wd)
= (1.0, 0.35, 0.35, 0.25, 0.10, 0.10, 0.15, 0.15)
```

Selection order:

```text
semantic filter
→ PPG protected top-P, P in {7000, 8000, 9000}
→ PPN recall-completion top-12000
→ generic structural top-12000
→ hybrid ranking
→ degree capacity 96/96
→ semantic-type capacity 800
→ relaxed capacity 128/1200
→ final top-10000
→ RPCM
```

The paper validation grid compared 7000/8000/9000 and selected \(P=9000\).
After selection, the base configs and public wrappers were frozen to 9000;
`selection.json` remains the provenance record. The old manual route happened
to favor 8000/2000, but that does not preselect the statistical route.

## 6. Runtime modes

Paper-facing mode:

```bash
RSGP_ROLE_MODE=statistical \
RSGP_STRUCTURAL_PRIOR_PATH=pretrained/rsgp_structural_prior.json \
FILTER_METHOD=RSGP \
bash scripts/_internal/eval_star_predcls.sh
```

Historical replay:

```bash
RSGP_ROLE_MODE=legacy_manual \
FILTER_METHOD=RSGP \
bash scripts/_internal/eval_star_predcls.sh
```

In statistical mode the following legacy fields are ignored:

```text
RSGP_ANCHOR_CLASSES
RSGP_VEHICLE_CLASSES
RSGP_NETWORK_CLASSES
RSGP_TAIL_PREDICATES
```

## 7. Validation protocol

The relation checkpoint is the validation-selected checkpoint from the
controlled PredCls DL row. All RSGP/PPG/PPN cases reuse that exact checkpoint,
the same Semantic Filter, and the fixed top-10,000 output budget.

Run validation-only mode selection:

```bash
bash scripts/research/select_predcls_rsgp_on_val.sh
```

The grid contains PPG and PPN references, statistical RS-only, PPN-graph, and
Hybrid 9000/1000, 8000/2000, and 7000/3000. A legacy-manual replay may be
generated for audit, but it is not eligible to become the paper-facing
statistical result.

Run statistical component ablations:

```bash
bash scripts/research/eval_predcls_rsgp_ablation.sh FULL
bash scripts/research/eval_predcls_rsgp_ablation.sh LEGACY_MANUAL
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_PPN
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_GEOMETRY
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_STRUCTURE
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_DEGREE
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_QUOTA
bash scripts/research/eval_predcls_rsgp_ablation.sh NO_RARITY
```

Freeze a statistical configuration only if validation HMR exceeds PPG and
R@2000 falls by no more than 0.2 percentage points. Test is run once after
freezing. `eval_rsgp_grid.sh` records the qualifying cases and the
highest-HMR choice in `selection.json`; an empty `selected` field means that
statistical RSGP has not passed the paper acceptance rule. The exact task
checkpoint must be reused for every filter.

The completed validation grid selected Hybrid 9000/1000. A second val-only
check at that fixed pool split retained the three statistical structural-role
kernels: Full reached R/mR/HMR@2000 = 0.6302/0.4231/0.5063, versus
0.6282/0.4211/0.5042 without the contextual, alignment, and connectivity
roles. This freezes the final method with all statistical components enabled;
test remove-one results are diagnostic and cannot override this decision.

## 8. Candidate-graph and pressure evidence

Every final PPG/PPN/RSGP JSON should be compared using:

- final relation-row-weighted GT-pair coverage;
- candidate-node coverage;
- average degree Gini and split-wide maximum degree;
- average label-pair entropy;
- R/mR/HMR under the same candidate-pressure group.

Pressure is defined by the Semantic-Filter candidate count before pair
top-10,000 selection:

```text
no truncation:   <= 10,000
low overload:    10,001--20,000
medium overload: 20,001--50,000
high overload:   > 50,000
```

`No truncation` only means that pair filtering does not discard a
semantic-valid pair. Triplets are still ranked at top-1000/1500/2000.
Pressure-group mR remains a macro statistic over the fixed predicate
vocabulary and is affected by group composition. The claim therefore comes
from same-group PPG-versus-RSGP comparison, not from requiring monotonic
performance across groups.

## 9. Legacy result boundary

All previously reported RSGP numbers were generated by the manual semantic
groups and fixed hard-predicate list. They are now named
`RSGP-v1/legacy_manual` and remain useful only as route-development evidence.
They cannot be relabeled as category-name-agnostic statistical RSGP.

The new method may claim:

> category-name-agnostic structural roles derived from training statistics.

It must not claim cross-dataset generalization until another dataset is
evaluated.
