# Minimal Submission Experiment Protocol

> Last updated: 2026-08-04

This file is the authoritative protocol for the final STAR OBB experiments.
Historical one-shot results under `outputs_old/` are useful for motivation and
debugging, but they are not substitutes for the controlled runs below.

## 1. Scope and method rows

Dataset and tasks are fixed to STAR OBB PredCls, SGCls, and SGDet. The proposed
method consists of role-aware dual-view RCA, auxiliary logit adjustment (LA),
and inference-only RSGP.

The paper-facing RSGP is the statistical, category-name-agnostic route:

```text
RSGP_ROLE_MODE=statistical
RSGP_STRUCTURAL_PRIOR_PATH=pretrained/rsgp_structural_prior.json
```

Build the prior once from `train` only with
`bash scripts/build_rsgp_structural_prior.sh`. Historical results obtained
with `legacy_manual` are development evidence, not final-table values.

The controlled PredCls rows are:

| Row | Relation adjacency | GNN/update | LA | Test filter |
|---|---|---|---:|---|
| Base | unified | current RCA GNN |  | PPG |
| D | shared-subject/shared-object dual view | same current RCA GNN |  | PPG |
| DL | dual view | same current RCA GNN | ✓ | PPG |
| Full | exact DL checkpoint | exact DL checkpoint | ✓ | RSGP |

Base is the existing `unified` branch: current RCA GNN, unified relation
adjacency, 6850-compatible classifier, and PPG. Base/D/DL share all settings
except the stated adjacency/LA change, so the controlled causal sequence is
Base→D→DL→Full. The complete source SGG-ToolKit predictor is retained only as
an audit (`SOURCE`) and is not duplicated in the main ablation table.

The Full row is not trained separately. RSGP has no learned relation-head
parameters, so it must reuse the byte-identical DL checkpoint.

The published STAR/RPCM result is an external baseline. A local source-RPCM
audit is run once and is not used to manufacture the headline improvement.

## 2. Data-use protocol

- `train`: optimization only.
- `val`: checkpoint selection and RSGP mode/protected-pool choices. SGDet is the
  explicit exception: validation is monitored, but the fixed-budget
  `model_last.pth` endpoint is reported.
- `test`: one final report after every choice is frozen.
- Pre-declared RSGP remove-one-component ablations also run once on `test`
  after the Full configuration is frozen; they must not be used for tuning.
- Pair budget: fixed top-10,000 for PPG, PPN, and RSGP.
- Each controlled Base/D/DL configuration is run once without explicitly
  fixing a random seed. The isolated D6850 audit is the sole exception: it
  fixes seed 1029 because reproducing that historical trajectory is the
  purpose of the experiment.
- Every run stores `run_manifest.json`/`eval_manifest.json`, the resolved
  config copy, log, checkpoint, metric JSON, PID, and exit status.

Do not choose an epoch, RSGP mode, protected-pool ratio, or component setting
from test results.

## 3. Commands

### Controlled PredCls training

Run the complete three-training-row queue:

```bash
bash scripts/research/run_paper_predcls_suite.sh
```

Each Base/D/DL training is followed immediately by one PPG test using
`model_best_HR.pth`. A failed training or test stops the serial queue.

Or launch one row:

```bash
bash scripts/research/run_predcls_minimal_ablation.sh BASE
bash scripts/research/run_predcls_minimal_ablation.sh D
bash scripts/research/run_predcls_minimal_ablation.sh DL
```

The source-RPCM audit is optional and outside the causal chain:

```bash
bash scripts/research/run_predcls_minimal_ablation.sh SOURCE
```

An isolated architecture-decision run is also available:

```bash
bash scripts/research/run_predcls_minimal_ablation.sh DL_ORIGINAL_HEAD
```

It keeps the current Dual+LA pairwise/RCA front end but replaces the
post-GNN classifier with source gated fusion, 300-D GloVe projection,
per-forward KMeans coarse prototypes, cosine logits, and five source losses.
It writes to `outputs/paper_submission/predcls/dual_la_original_head/` and is
not part of the main table unless validation justifies adopting this head
consistently and rerunning Base/D/DL.

The former MV/cross-role trial is replaced by a historical Dual audit:

```bash
bash scripts/research/run_predcls_minimal_ablation.sh D6850
```

It uses the Dual SS/OO graph and exact single-prototype head, seed 1029,
20,000 optimizer steps, LR milestones 13,000/18,000, and PPG validation every
200 steps from step 14,000. It is not an additional row in the controlled
Base→D→DL causal chain.

All three training rows load only `pretrained/OBB_swin_L_OBD.pth`, use the same
optimization protocol, and select `model_best_HR.pth` on `val` with PPG.
Base/D/DL share the 6850-compatible initialization and classifier and begin
validation at epoch 120. The optional source audit and `DL_ORIGINAL_HEAD`
decision run start validation at epoch 70 because their best-epoch ranges are
not established. Every row validates every two epochs. Early-stop patience is
not counted before epoch 120, then training stops after 10 validation events
without a strict improvement in HMR@2000. The 220-epoch limit remains the
safety cap. Early source-audit validations can save `model_best_HR.pth`, but cannot
prematurely stop the run.

The launcher automatically resumes from a row's existing `model_last.pth`.
The current Python config remains authoritative for `MAX_EPOCHS`, validation
cadence, early-stop patience, and output metadata; `run_manifest.json` and
`source_config.py` are refreshed when training continues. New checkpoints
store the early-stop state. Older checkpoints are supported by reconstructing
the state from the row's `validation_history.jsonl`. Set `AUTO_RESUME=0` only
when intentionally starting over in an empty/new output directory.

A row is considered complete only when
`<row>/test/ppg/test_metrics.json` exists. Re-running the suite appends a new
session to `suite.log`, skips completed rows, resumes interrupted rows, and
starts only missing rows. Set `SKIP_COMPLETED=0` to force traversal.

### RSGP validation selection

For the completed DL checkpoint:

```bash
bash scripts/research/select_predcls_rsgp_on_val.sh
```

Read:

```text
outputs/paper_submission/predcls/dual_la/
└── rsgp_val_selection/
    ├── comparison.json
    └── selection.json
```

`selection.json` only admits statistical cases with validation
`HMR@2000 > PPG` and `R@2000 >= PPG - 0.002`, then selects the highest HMR.
If `selected` is null, do not report the old manual result as the new method.

This is the only permitted RSGP hyperparameter-selection pass. The script
evaluates PPG/PPN, RS-only, PPN-graph, and the statistical Hybrid
9000/1000, 8000/2000, and 7000/3000 variants on `val`. The manual 8000/2000
case is logged only as historical context and is excluded from eligibility.

Before final test, record and freeze:

```text
DL checkpoint: outputs/paper_submission/predcls/dual_la/model_best_HR.pth
structural prior: pretrained/rsgp_structural_prior.json
selected case: rsgp_val_selection/selection.json -> selected
pair budget: 10000
RSGP mode and protected-pool size implied by the selected case
```

Selected-case mapping:

| `selection.json.selected` | Test setting |
|---|---|
| `rsgp_rs_only` | `RSGP_MODE=RS_ONLY`, protected top-k 0 |
| `rsgp_ppn_graph` | `RSGP_MODE=PPN_GRAPH`, protected top-k 0 |
| `rsgp_hybrid_9000_1000` | `RSGP_MODE=HYBRID`, protected top-k 9000 |
| `rsgp_hybrid_8000_2000` | `RSGP_MODE=HYBRID`, protected top-k 8000 |
| `rsgp_hybrid_7000_3000` | `RSGP_MODE=HYBRID`, protected top-k 7000 |

Run test once only after this record exists. Any test produced before
`selection.json` is a provisional engineering result and must not be used to
retune RSGP while still being described as val-only selection.

The completed validation run selected `rsgp_hybrid_9000_1000`. Its val
R/mR/HMR@2000 are 0.6302/0.4231/0.5063, compared with
0.6085/0.3980/0.4812 for PPG. This selection is now frozen for the final
PredCls and cross-task RSGP evaluations.

| Validation filter | Eligible statistical candidate | R@2000 | mR@2000 | HMR@2000 | GT-pair coverage | Decision |
|---|:---:|---:|---:|---:|---:|---|
| PPG 10000 | — | 0.6085 | 0.3980 | 0.4812 | 0.7288 | acceptance reference |
| PPN 10000 | — | 0.6014 | 0.3867 | 0.4707 | **0.8628** | proposal reference only |
| RSGP RS-only | yes | 0.6277 | 0.4153 | 0.4999 | 0.7443 | eligible |
| RSGP PPN-graph | yes | **0.6331** | 0.4202 | 0.5052 | 0.7653 | eligible |
| RSGP Hybrid 7000/3000 | yes | 0.6303 | 0.4226 | 0.5060 | 0.7066 | eligible |
| RSGP Hybrid 8000/2000 | yes | 0.6301 | 0.4224 | 0.5058 | 0.7066 | eligible |
| **RSGP Hybrid 9000/1000** | yes | 0.6302 | **0.4231** | **0.5063** | 0.7066 | **selected** |

All five statistical RSGP candidates satisfy the pre-declared acceptance
rule. Selection then maximizes HMR@2000 and uses R@2000 only as a tie-breaker;
therefore the highest-R PPN-graph variant is not selected. The historical
manual 8000/2000 row remains an audit result and is excluded from this table's
eligible set.

A final validation-only component decision was then made at the frozen
9000/1000 split. Full statistical RSGP achieved R/mR/HMR@2000 of
0.6302/0.4231/0.5063, while disabling all three statistical structural roles
gave 0.6282/0.4211/0.5042. Full is therefore the frozen final method. The
slightly higher no-role HMR observed later on test is reported as an ablation
result but is not used to revise this validation decision. The machine-readable
record is:

```text
outputs/paper_submission/predcls/dual_la/rsgp_finalization_val/final_selection.json
```

Final defaults are `RSGP_MODE=HYBRID`, `RSGP_PPG_PROTECTED_TOPK=9000`,
`RSGP_TOPK=10000`, `RSGP_ROLE_MODE=statistical`, with PPN completion,
geometry, contextual/alignment/connectivity roles, rarity support, degree
control, and semantic capacity enabled.

Component evidence must be reported with the following split boundary. Values
are Full-minus-reference HMR@2000 in percentage points:

| Comparison | Validation ΔHMR | Test ΔHMR | Permitted interpretation |
|---|---:|---:|---|
| Full RSGP − PPG | **+2.51** | **+1.90** | overall RSGP gain is directionally consistent |
| Full − no PPN completion | -0.05 | +0.17 | small split-dependent effect, not a stable isolated gain |
| Full − no geometry | ≈0.00 | +0.08 | validation-neutral, small test-side macro balancing effect |
| Full − no structural roles | **+0.21** | -0.06 | validation-selected auxiliary evidence |
| Full − no degree control | +0.04 | +0.26 | same direction, weak validation effect |
| Full − no semantic capacity | **+1.50** | **+0.99** | strongest stable selection constraint |
| Full − no rarity support | +0.19 | +0.18 | small but highly consistent gain |

The validation no-PPN run was a consistency diagnostic outside the declared
pool/mode grid and must not be presented as a second model-selection pass.
Except for the pre-recorded structural-role finalization, the added
degree/capacity/rarity/geometry validation rows are post-freeze consistency
audits, not a new model-selection pass and do not trigger re-testing. They show
semantic capacity as the strongest stable constraint and rarity as a small
consistent term. Degree control has a weaker validation effect; geometry is
validation-neutral; PPN completion and structural roles remain split-dependent.
Therefore the table does not support a claim that every auxiliary component
improves every split independently.

### Final PredCls test

```bash
RUN_BACKGROUND=0 bash scripts/research/eval_predcls_rsgp_ablation.sh FULL
```

This writes the frozen 9000/1000 Full result to
`dual_la/rsgp_component_test/rsgp_full/` while leaving the older engineering
8000/2000 JSON untouched. The paper wrapper now defaults to the
validation-selected protected-pool value 9000. Supplying a different value is
a new ablation, not the final protocol. Summarize the completed runs with:

```bash
python tools/summarize_paper_experiments.py
```

### SGCls and SGDet

Build all three cache splits before SGDet validation. The current v5 cache is
complete for train/val/test (771/245/264 images), so it should be reused rather
than rebuilt:

```bash
# Audit only; no rebuild is required.
python -m json.tool star_sgdet_detection_cache/val/manifest.json >/dev/null
```

Train:

```bash
bash scripts/research/run_star_sgcls_experiment.sh
bash scripts/research/run_star_sgdet_rpcm_budget.sh
```

SGCls automatically tests `model_best_HR.pth`. SGDet deliberately tests
`model_last.pth`, matching the requested fixed-budget endpoint convention.

Final same-checkpoint PPG/RSGP evaluation under both protocols:

```bash
bash scripts/research/eval_paper_cross_task_suite.sh
```

`legacy` uses GT filter labels for SGCls and matched-GT filter labels for
SGDet, matching the STAR/SGG-ToolKit filtering convention. `pred` is the
strict fully predicted-label filtering protocol.

## 4. Metric JSON contract

Every final JSON contains:

- `R`, `mR`, and `HR` at 1000/1500/2000;
- `per-predicate-recall` and `predicate-counts` for the complete class-wise
  supplement (new evaluations; old frozen logs remain supported);
- `per-image`: all image-wise graph/recall rows used for paired qualitative
  selection (new evaluations);
- `candidate-stage-coverage.final`: relation-row-weighted GT pair coverage
  (each GT relation row checks whether its subject-object pair survives);
- `candidate-graph.avg_candidate_node_coverage`;
- `candidate-graph.avg_gt_relation_node_coverage` where entity indices are
  aligned with GT boxes;
- `candidate-graph.avg_degree_gini`;
- `candidate-graph.max_degree`;
- `candidate-graph.avg_label_pair_entropy`;
- `candidate-pressure`: no-truncation and low/medium/high overload R/mR/HMR;
- `failure-cases`: ten lowest-recall images with graph diagnostics.

Pressure groups are defined from the post-semantic-filter candidate count:

| Group | Candidate count |
|---|---:|
| no truncation | ≤10,000 |
| low overload | 10,001–20,000 |
| medium overload | 20,001–50,000 |
| high overload | >50,000 |

The count is measured after the Semantic Filter and before pair top-10,000
selection. `No truncation` therefore means that the pair filter keeps all
semantic-valid pairs; the evaluator still performs graph-constrained
top-1000/1500/2000 triplet ranking.

Pressure-group `R` is the average of per-image recall inside the group.
Pressure-group `mR` is still macro-aggregated over the fixed predicate
vocabulary, so it is affected by the predicate composition of each group.
No monotonic gain with load is required, and values from different groups
should not be interpreted as a pure scaling curve. The required controlled
evidence is the PPG-versus-RSGP difference inside the same pressure group:
they should be similar when pruning is unnecessary, while RSGP should be more
effective when the fixed top-10,000 budget must discard candidates.

## 5. Output layout

```text
outputs/paper_submission/
├── predcls/
│   ├── base_original/
│   ├── unified/
│   ├── dual/
│   └── dual_la/
├── rpcm_audit/
├── sgcls/
├── sgdet_rpcm_budget/
└── summary.json

star_sgdet_detection_cache/
├── train/
├── val/
└── test/
```

Old development outputs are archived under `outputs_old/`; do not let wrapper
fallbacks select checkpoints from that directory.

## 6. Paper figures

The frozen PredCls outputs select image `440` as the airport-hub success,
`748` as a readable multi-component success, and `235` as the paired failure. All
three have more than 10,000 post-semantic-filter pairs,
so both PPG and RSGP actually execute their top-10,000 selection. Cases are
ranked by the final graph-constrained triplet prediction difference, not by GT
pair coverage alone. Their records and thumbnails are generated by
`tools/extract_paper_supplement.py`. Use them as follows:

1. method overview: semantic filter → multi-source RSGP → constrained graph →
   dual-view RCA → prototype classifier;
2. dual-view illustration: one entity shared as subject versus object, showing
   the two adjacency channels and the unified-graph mixing that is avoided;
3. PPG/PPN/RSGP candidate graphs on image `440`, annotated with pair
   coverage, node coverage, degree Gini, maximum degree, label-pair entropy,
   and downstream triplet recall;
4. image `748` as a second readable filtered success and image `235` as the RSGP
   failure. Use spatially aligned GT/PPG/RSGP scene graphs with status-coded
   edges; omit predicate text, collapse predicates by directed entity pair,
   separate reverse pairs, and enforce a minimum node distance. Near-complete
   star graphs use bounded angular relaxation around the projected hub. Separately
   distinguish missing box, missing pair, and wrong predicate errors for SGDet.

## 7. Completion audit

Completed without further numerical runs:

- PredCls Base/D/DL tests;
- validation-only RSGP grid and Hybrid 9000/1000 selection;
- validation-only structural-role finalization;
- final PredCls Full test;
- all six RSGP remove-one component rows;
- same-checkpoint candidate-graph and pressure comparisons;
- train/val/test SGDet v5 detection caches.
- complete 58-class per-predicate supplement and qualitative case selection.

Regenerate the paper supplement artifacts without inference:

```bash
python tools/extract_paper_supplement.py --render-thumbnails
```

Run targeted PPG/RSGP inference and render readable local crops for the frozen
qualitative cases:

```bash
bash scripts/research/export_paper_qualitative_cases.sh
```

This processes only image IDs `440,748,235`. The output combines one satellite
crop with spatially aligned GT, PPG, and RSGP scene graphs. Every cropped GT
entity keeps the same location and node ID across the three graphs. Green,
blue, and purple solid edges are actual matched outputs. A blue or purple dashed
edge with a circled x means that the correct relation is predicted only by the
other method and is absent here; it is a comparison reference, not a model output.
Predicate text is omitted, crowded projected nodes are deterministically spread,
and predicates on one directed pair are collapsed into one structural arrow.
Image 440 additionally uses the documented visualization-only hub/sector spacing
adjustment shared by all three graph panels; reverse pairs retain separate shallow curves.
The renderer rejects no-truncation cases by default and does not present
candidate coverage as prediction correctness.

The cross-task evaluation is complete. Re-running it, if needed, uses:

```bash
bash scripts/research/eval_paper_cross_task_suite.sh

python tools/summarize_paper_experiments.py
```

The comparison uses the recovered SGCls Dual+LA best checkpoint and the
5,000-step SGDet endpoint. Their older manual-RSGP JSON files are historical;
the final tables use the statistical Hybrid 9000/1000 outputs under
`outputs/paper_submission/{sgcls,sgdet_rpcm_budget}/test/`.

The cross-task suite has produced eight JSON files: SGCls/SGDet × legacy/pred
label-source protocol × PPG/RSGP. Legacy is the direct STAR comparison;
the predicted-label results may be placed in the main paper or supplement.

Protocol hygiene: the current DL metrics use a valid validation-selected
checkpoint, but its training process was manually terminated at epoch 204
(`exit_code=143`). The best epoch was 144 and remained unbeaten for 29 later
validations. Resume only if a clean terminal state is required for release;
rerun downstream evaluations only if the best checkpoint changes.
