# IPM revision completion runbook

This runbook is the ordered command list for the corrected revision experiments after the GRU4Rec architecture audit. It assumes the server working directory is `/root/FARE` and that GPU jobs are run on the experiment server, not on the Windows editing machine.

## Scope and assumptions

- Protocol source: `configs/revision/revision_protocol.yaml`.
- Breadth seeds: `2024,2025,2026`.
- Confirmatory seeds: `2024,2025,2026,2027,2028,2029`.
- Datasets: `Video_Games`, `Musical_Instruments`, `Baby_Products`, `MicroLens_100K`.
- Backbones: `sasrec`, `gru4rec`, `bert4rec`.
- GRU4Rec correction: FARE-side GRU4Rec must use `num_layers=1`, matching the ID checkpoint architecture. Corrected GRU FARE run IDs are tagged with `archfix_v1`; do not merge old untagged GRU FARE artifacts into the selector.
- Baby Products + BERT4Rec is included in the confirmatory six-seed scope because reviewer comments singled out that setting as a weak/negative case.
- `--repetitions 10000` means bootstrap resamples for CI estimation, not 10000 random seeds.

## 0. Preflight on the server

```bash
cd /root/FARE

python scripts/validate_revision_protocol.py \
  --config configs/revision/revision_protocol.yaml \
  --output revision_outputs/preflight_corrected.json

python -m py_compile \
  src/revision_protocol.py \
  src/revision_diagnostics.py \
  src/revision_reporting.py \
  scripts/run_fare.py \
  scripts/run_modality_debias.py \
  scripts/build_revision_sweeps.py \
  scripts/build_revision_jobs.py \
  scripts/run_revision_jobs.py \
  scripts/evaluate_selected_fare.py \
  scripts/evaluate_selected_findrec.py \
  scripts/collect_sweep_run_registry.py \
  scripts/merge_revision_run_registries.py \
  scripts/audit_findrec.py \
  scripts/benchmark_revision_efficiency.py \
  scripts/summarize_revision_experiments.py \
  scripts/build_revision_evidence_report.py
```

Expected protocol counts after the corrected scope:

```text
confirmatory pairs per method: 42
extra-seed pairs per method: 21
total training jobs: 690
total scheduled jobs: 822
```

## 1. Fill missing confirmatory ID checkpoints

Run the missing extra ID checkpoints first. At minimum this includes Baby Products + BERT4Rec seeds `2027,2028,2029`; MicroLens GRU extra IDs are also needed if absent.

```bash
python scripts/build_revision_jobs.py \
  --protocol configs/revision/revision_protocol.yaml \
  --id-registry revision_outputs/id_checkpoints.csv \
  --output-dir revision_outputs/generated_phase1_corrected

python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase1_corrected/jobs.csv \
  --phase confirmatory_dependency \
  --method ID \
  --dataset Baby_Products \
  --backbone bert4rec \
  --seed 2027 --seed 2028 --seed 2029 \
  --status-out revision_outputs/generated_phase1_corrected/status_confirmatory_id_baby_bert.csv \
  --resume --continue-on-error

python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase1_corrected/jobs.csv \
  --phase confirmatory_dependency \
  --method ID \
  --dataset MicroLens_100K \
  --backbone gru4rec \
  --seed 2027 --seed 2028 --seed 2029 \
  --status-out revision_outputs/generated_phase1_corrected/status_confirmatory_id_microlens_gru.csv \
  --resume --continue-on-error

python scripts/collect_id_checkpoint_registry.py \
  --results-root /root/autodl-tmp/results_sdr \
  --datasets-config configs/datasets.yaml \
  --output revision_outputs/id_checkpoints_all_seeds_corrected.csv
```

If your current server stores the corrected ID runs under `/root/FARE/results` instead, replace `--results-root /root/autodl-tmp/results_sdr` with `--results-root results`. The registry must point to the same root that contains the `*_id` run folders with `config_resolved.json`, `best_model.pt`, `topk_val.npz`, and `metrics_summary.json`.

Audit the registry:

```bash
python - <<'PY'
import pandas as pd
p = 'revision_outputs/id_checkpoints_all_seeds_corrected.csv'
df = pd.read_csv(p)
print('rows=', len(df))
print(df.groupby(['dataset','backbone'])['seed'].nunique())
assert len(df) >= 57
PY
```

## 2. Rebuild corrected phase-1 sweep jobs

```bash
python scripts/build_revision_jobs.py \
  --protocol configs/revision/revision_protocol.yaml \
  --id-registry revision_outputs/id_checkpoints_all_seeds_corrected.csv \
  --output-dir revision_outputs/generated_phase1_corrected
```

Only rerun GRU4Rec FARE breadth sweeps, because SASRec and BERT4Rec breadth sweeps are already valid under the architecture audit.

```bash
python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase1_corrected/jobs.csv \
  --phase breadth_selection \
  --method FARE-Sweep \
  --backbone gru4rec \
  --status-out revision_outputs/generated_phase1_corrected/status_fare_gru_archfix.csv \
  --resume --continue-on-error
```


## 3. Replace the invalid GRU validation registry

Collect only the corrected GRU sweep outputs, then replace the old GRU rows in the full validation registry. Do not append corrected GRU rows to the old registry; that would double-count GRU candidates.

```bash
python scripts/collect_sweep_run_registry.py \
  --sweep-results sweeps/fare_gamma_*_gru4rec_*archfix_v1*/sweep_results.csv \
  --output revision_outputs/fare_validation_runs_gru_archfix.csv

python scripts/merge_revision_run_registries.py \
  --base revision_outputs/fare_validation_runs.csv \
  --replacement revision_outputs/fare_validation_runs_gru_archfix.csv \
  --replace-backbone gru4rec \
  --output revision_outputs/fare_validation_runs_corrected.csv

python scripts/collect_revision_validation.py \
  --registry revision_outputs/fare_validation_runs_corrected.csv \
  --output revision_outputs/fare_validation_metrics_corrected.csv \
  --audit revision_outputs/fare_validation_metrics_corrected_audit.json \
  --k 10

python scripts/select_revision_runs.py \
  --input revision_outputs/fare_validation_metrics_corrected.csv \
  --selected-out revision_outputs/selected_configs_corrected.csv \
  --sensitivity-out revision_outputs/policy_sensitivity_corrected.csv \
  --audit-out revision_outputs/selection_audit_corrected.json

python scripts/materialize_selected_runs.py \
  --selected revision_outputs/selected_configs_corrected.csv \
  --run-registry revision_outputs/fare_validation_runs_corrected.csv \
  --output revision_outputs/selected_seed_runs_corrected.csv
```

Audit the selected breadth runs:

```bash
python - <<'PY'
import pandas as pd
sel = pd.read_csv('revision_outputs/selected_seed_runs_corrected.csv')
print('selected rows=', len(sel))
print(sel.groupby(['dataset','backbone'])['seed'].nunique())
assert len(sel) == 36
assert (sel.groupby(['dataset','backbone'])['seed'].nunique() == 3).all()
PY
```

## 4. Rebuild phase-2 jobs from corrected selections

```bash
python scripts/build_revision_jobs.py \
  --protocol configs/revision/revision_protocol.yaml \
  --id-registry revision_outputs/id_checkpoints_all_seeds_corrected.csv \
  --selected-fare-registry revision_outputs/selected_seed_runs_corrected.csv \
  --selected-findrec-registry revision_outputs/findrec_selected.csv \
  --output-dir revision_outputs/generated_phase2_corrected
```

## 5. Run selected FindRec final tests

Run the frozen validation-selected FindRec checkpoints once on the test split before generating Top-10 diagnostics. This keeps tuning evidence validation-only and separates it from final test reporting.

```bash
python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase2_corrected/jobs.csv \
  --method FindRec-Selected-Test \
  --status-out revision_outputs/generated_phase2_corrected/status_findrec_selected_test.csv \
  --resume --continue-on-error
```

## 6. Run corrected GRU final FARE tests

```bash
python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase2_corrected/jobs.csv \
  --method FARE-Selected-Test \
  --backbone gru4rec \
  --status-out revision_outputs/generated_phase2_corrected/status_fare_selected_gru_archfix.csv \
  --resume --continue-on-error
```

## 7. Run GRU controls and rerun FARE+MD for efficiency

```bash
python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase2_corrected/jobs.csv \
  --method ExposureReweight-ID \
  --backbone gru4rec \
  --status-out revision_outputs/generated_phase2_corrected/status_exposure_reweight_gru.csv \
  --resume --continue-on-error

python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase2_corrected/jobs.csv \
  --method FARE-IndependentPrior \
  --backbone gru4rec \
  --status-out revision_outputs/generated_phase2_corrected/status_prior_gru.csv \
  --resume --continue-on-error

python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase2_corrected/jobs.csv \
  --method FARE+ModalityDebias \
  --status-out revision_outputs/generated_phase2_corrected/status_fare_md_all_efficiency.csv \
  --resume --continue-on-error
```

## 8. Run confirmatory additions

Use the frozen selected FARE policy. These are the minimal additions needed for six paired seeds under the corrected scope.

```bash
python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase2_corrected/jobs.csv \
  --method FARE-Confirmatory \
  --dataset MicroLens_100K \
  --backbone gru4rec \
  --seed 2027 --seed 2028 --seed 2029 \
  --status-out revision_outputs/generated_phase2_corrected/status_confirmatory_fare_microlens_gru.csv \
  --resume --continue-on-error

python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase2_corrected/jobs.csv \
  --method FARE-Confirmatory \
  --dataset Baby_Products \
  --backbone bert4rec \
  --seed 2027 --seed 2028 --seed 2029 \
  --status-out revision_outputs/generated_phase2_corrected/status_confirmatory_fare_baby_bert.csv \
  --resume --continue-on-error
```

## 9. Rerun all 36 formal FairRR breadth jobs

The corrected baseline uses the Top-100 ID candidate pool and optimizes only the
three-column `popularity_group` taxonomy. Lambda is selected on validation by
minimizing discounted exposure deviation under all four fixed constraints:
validation NDCG@10 retention at least 95%, rank-change rate at least 1%, set-change
rate at least 1%, and positive exposure-penalty improvement. A run fails when no
candidate lambda is feasible or when the frozen lambda does not meet the test
intervention thresholds.

Regenerate the manifest so all FairRR outputs use the independent
`fairrrfix_v1` run tag:

```bash
python scripts/build_revision_jobs.py \
  --protocol configs/revision/revision_protocol_topkfix.yaml \
  --id-registry revision_outputs/id_checkpoints_all_seeds_corrected.csv \
  --selected-fare-registry revision_outputs/selected_seed_runs_topkfix.csv \
  --selected-findrec-registry revision_outputs/findrec_selected.csv \
  --output-dir revision_outputs/generated_phase2_fairrrfix_v1
```

Verify the 4 datasets x 3 backbones x 3 seeds scope before execution:

```bash
python - <<'PY'
import pandas as pd

p = 'revision_outputs/generated_phase2_fairrrfix_v1/jobs.csv'
df = pd.read_csv(p)
jobs = df[(df['phase'] == 'breadth_control') & (df['method'] == 'FairRR')]
print(jobs.groupby(['dataset', 'backbone'])['seed'].apply(list))
assert len(jobs) == 36
assert jobs['status'].eq('ready').all()
assert jobs['expected_run_dir'].str.endswith('fairrrfix_v1').all()
PY
```

Run the 36 jobs. Do not reuse an old status file because it may mark the legacy
runs as complete.

```bash
python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase2_fairrrfix_v1/jobs.csv \
  --phase breadth_control \
  --method FairRR \
  --status-out revision_outputs/generated_phase2_fairrrfix_v1/status_fairrrfix_v1.csv \
  --continue-on-error
```

If the process is interrupted, rerun the same command with `--resume`. Accept the
batch only when all 36 rows are successful:

```bash
python - <<'PY'
import pandas as pd

p = 'revision_outputs/generated_phase2_fairrrfix_v1/status_fairrrfix_v1.csv'
df = pd.read_csv(p)
print(df['execution_status'].value_counts(dropna=False))
assert len(df) == 36
assert df['execution_status'].eq('ok').all()
PY
```
## 10. Rebuild phase-2 manifests after all reruns

Rebuild after artifacts exist so readiness, significance, and efficiency manifests point to current outputs.

```bash
python scripts/build_revision_jobs.py \
  --protocol configs/revision/revision_protocol.yaml \
  --id-registry revision_outputs/id_checkpoints_all_seeds_corrected.csv \
  --selected-fare-registry revision_outputs/selected_seed_runs_corrected.csv \
  --selected-findrec-registry revision_outputs/findrec_selected.csv \
  --output-dir revision_outputs/generated_phase2_corrected
```

## 11. Significance, efficiency, and FindRec Top-10 diagnostics

```bash
python scripts/analyze_revision_significance.py \
  --manifest revision_outputs/generated_phase2_corrected/significance_manifest.csv \
  --output-csv revision_outputs/significance_corrected.csv \
  --output-json revision_outputs/significance_corrected_audit.json \
  --repetitions 10000 \
  --bootstrap-seed 2026

python scripts/benchmark_revision_efficiency.py \
  --manifest revision_outputs/generated_phase2_corrected/efficiency_manifest.csv \
  --output revision_outputs/efficiency_corrected.csv \
  --audit revision_outputs/efficiency_corrected_audit.json

python scripts/audit_findrec.py \
  --registry revision_outputs/findrec_selected.csv \
  --k 10 \
  --output-csv revision_outputs/findrec_diagnostics_top10.csv \
  --output-json revision_outputs/findrec_diagnostics_top10.json
```

The corrected `efficiency_manifest.csv` should contain 222 run rows when all corrected artifacts are present: ID/FARE for 57 breadth+confirmatory keys, plus 108 breadth control/composition runs.

## 12. Unified result aggregation

```bash
python scripts/summarize_revision_experiments.py \
  --protocol configs/revision/revision_protocol.yaml \
  --id-registry revision_outputs/id_checkpoints_all_seeds_corrected.csv \
  --fare-registry revision_outputs/selected_seed_runs_corrected.csv \
  --jobs-manifest revision_outputs/generated_phase2_corrected/jobs.csv \
  --findrec-registry revision_outputs/findrec_selected.csv \
  --output-dir revision_outputs/final_evidence \
  --k 10
```

The expected breadth-level run total is 228 rows: six common methods times 36 dataset-backbone-seed settings, plus 12 FindRec dataset-seed settings.

## 13. Tables, point-to-point response, and claim audit

```bash
python scripts/build_revision_evidence_report.py \
  --results-aggregate revision_outputs/final_evidence/revision_results_aggregate.csv \
  --results-per-run revision_outputs/final_evidence/revision_results_per_run.csv \
  --significance revision_outputs/significance_corrected.csv \
  --efficiency revision_outputs/efficiency_corrected.csv \
  --findrec-diagnostics revision_outputs/findrec_diagnostics_top10.csv \
  --output-dir revision_outputs/final_report \
  --k 10
```

Generated outputs:

- `revision_outputs/final_report/revision_main_table.md`
- `revision_outputs/final_report/revision_main_table.tex`
- `revision_outputs/final_report/revision_efficiency_table.md`
- `revision_outputs/final_report/claim_audit.json`
- `revision_outputs/final_report/experimental_comments_point_by_point_response.md`

If this command exits non-zero, do not use the generated response as final submission text. Read `claim_audit.json` and fix the listed missing artifacts first.

## 14. Claim contraction rule

Use the claim audit to set the manuscript conclusion.

- If all utility confirmatory outcomes have non-negative support and all fairness outcomes have strictly positive support, the original broad claim can remain.
- If utility or fairness support is mixed, use the narrower conclusion: FARE provides a validation-selected utility-exposure tradeoff whose direction and statistical certainty vary by dataset, backbone, and fairness endpoint.
- If GRU4Rec corrected results remain collapsed, report the corrected architecture audit and treat GRU4Rec as a limitation rather than evidence for universal backbone robustness.
