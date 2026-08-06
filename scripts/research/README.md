# Research-only launchers

This directory contains paper ablations, validation-only RSGP selection, and
artifact export workflows. They are retained for auditability, but they are
not part of the stable public API.

For ordinary use, run only the six task launchers in the parent directory:

```bash
bash scripts/train_star_predcls_full.sh
bash scripts/train_star_sgcls_full.sh
bash scripts/train_star_sgdet_full.sh

bash scripts/test_star_predcls_full.sh
bash scripts/test_star_sgcls_full.sh
bash scripts/test_star_sgdet_full.sh
```

The provenance commands in this directory are documented in
`docs/submission_experiment_protocol.md` and
`docs/paper_contributions_and_experiments.md`.
