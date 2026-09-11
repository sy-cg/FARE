# IPM Major Revision Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a protocol-correct, reproducible experiment and analysis suite that addresses every code-related IPM reviewer concern.

**Architecture:** Repair the shared evaluation protocol, add exposure-prior and composition capabilities to existing runners, and keep analysis/diagnostics in Torch-independent modules. Reuse the existing generic sweep runner with revision-specific YAML configurations.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, PyYAML, PyTorch in the RTX 3090 environment, and standard-library unittest.

---

### Task 1: Correct full-sort history masking

**Files:**
- Create: `src/eval_protocol.py`
- Modify: `scripts/run_id_backbone.py`
- Test: `tests/test_eval_protocol.py`

1. Write a failing test showing that a test prefix item absent from `seen_train` is still masked while the target remains unmasked.
2. Run the test and verify the missing module/behavior failure.
3. Add a small prefix/seen union helper and call it from `evaluate_full_sort`.
4. Run the focused test, existing tests, and syntax checks.

### Task 2: Add independent exposure priors and the ID-branch control

**Files:**
- Create: `src/exposure_prior.py`
- Modify: `scripts/run_fare.py`
- Modify: `configs/fare_3090.yaml`
- Create: `configs/revision/fare_id_exposure_reweight_3090.yaml`
- Test: `tests/test_exposure_prior.py`
- Test: `tests/test_fare_revision_cli.py`

1. Write failing tests for Top-K, train-popularity, and platform-view group exposure weights.
2. Verify that missing priors, invalid shapes, and test-split Top-K references are rejected.
3. Implement pure NumPy prior construction and connect it to FARE tensors.
4. Add `--fair_rec_exposure_source`, `--fair_rec_exposure_item_weights_path`, and `--train_id_backbone`.
5. Define the residual-off, ID-unfrozen control config and verify its resolved gradient path statically.

### Task 3: Add validation-only selection and policy sensitivity

**Files:**
- Create: `src/revision_statistics.py`
- Create: `scripts/select_revision_runs.py`
- Test: `tests/test_revision_selection.py`

1. Write failing tests proving that test metrics cannot influence selection.
2. Implement per-dataset/backbone validation normalization, utility/exposure scoring, deterministic tie-breaking, and alpha sensitivity.
3. Emit selected-run CSV, policy-sensitivity CSV, and a JSON audit containing the selection fields used.
4. Reject manifests without split, seed, run path, and required validation metrics.

### Task 4: Add paired confidence intervals and significance analysis

**Files:**
- Modify: `src/revision_statistics.py`
- Create: `scripts/analyze_revision_significance.py`
- Test: `tests/test_revision_statistics.py`

1. Write failing tests using identical arrays, a known positive effect, misaligned users, and incomplete seed pairs.
2. Implement per-user HR/NDCG contributions, exact seed sign-flip tests, paired standardized effects, and seed-level paired bootstrap intervals, while retaining hierarchical bootstrap as an optional sensitivity mode.
3. Add exposure-gap bootstrap using item group labels.
4. Emit long-form CSV and machine-readable JSON including seeds, user counts, bootstrap seed, repetitions, estimates, CIs, p-values, and effect sizes.

### Task 5: Make FairRR and FindRec auditable

**Files:**
- Create: `src/revision_diagnostics.py`
- Create: `scripts/audit_fairrr.py`
- Create: `scripts/audit_findrec.py`
- Modify: `scripts/run_reranker_fair.py`
- Test: `tests/test_revision_diagnostics.py`

1. Write failing tests for unchanged lists, reordered-only lists, changed item sets, collapsed recommendation coverage, and non-improving learning curves.
2. Implement rank-change, set-change, candidate-width, target-rank, coverage, Gini, score-spread, and learning-curve summaries.
3. Make formal FairRR execution fail when its effective candidate pool is not wider than final Top-K or when no list changes.
4. Emit diagnostics that distinguish adaptation failure, optimization failure, and genuine weak performance.

### Task 6: Add FARE+ModalityDebias composition

**Files:**
- Modify: `scripts/run_modality_debias.py`
- Create: `configs/revision/fare_md_3090.yaml`
- Test: `tests/test_fare_md_spec.py`

1. Write failing tests for the new `fare_sasrec`, `fare_gru4rec`, and `fare_bert4rec` base-model choices.
2. Build FARE with the same resolved dimensions as its checkpoint and require `--init_base_checkpoint`.
3. Freeze the pretrained FARE base by default for the formal joint control and train only ModalityDebias branches.
4. Record both FARE and MD settings in `config_resolved.json` and label the method `FARE+ModalityDebias`.

### Task 7: Add efficiency and reproducibility reports

**Files:**
- Create: `src/efficiency_audit.py`
- Create: `scripts/benchmark_revision_efficiency.py`
- Test: `tests/test_efficiency_audit.py`

1. Write failing tests for timing summaries, warm-up exclusion, percentile latency, memory fields, and parameter accounting.
2. Implement exposure-weight construction timing plus trainable/total parameter, epoch time, batched inference latency, throughput, and peak CUDA memory reporting.
3. Save per-run JSON and aggregate CSV with hardware and software metadata.

### Task 8: Define and validate the formal experiment matrix

**Files:**
- Modify: `configs/datasets.yaml`
- Create: `configs/revision/gamma_wide_3090.yaml`
- Create: `configs/revision/exposure_prior_control_3090.yaml`
- Create: `configs/revision/microlens_fare_3090.yaml`
- Create: `configs/revision/revision_protocol.yaml`
- Create: `scripts/validate_revision_protocol.py`
- Test: `tests/test_revision_protocol.py`

1. Write failing tests for missing MicroLens registration, narrow gamma grids, test-split selection keys, missing seeds, and incomplete reviewer coverage.
2. Register MicroLens-100K and define gamma `{0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80}`, policy alpha `{0.25, 0.40, 0.50, 0.60, 0.75}`, and seeds `{2024, 2025, 2026}`.
3. Define Amazon train-prior and MicroLens platform-view-prior controls, ID-reweight, FARE+MD, FairRR, FindRec diagnostics, and efficiency phases.
4. Add a preflight command that validates data, paths, protocol invariants, and expected job counts without starting GPU training.

### Task 9: Full verification and handoff

**Files:**
- Update: `README.md`

1. Run all standard-library tests under the Torch-independent Python environment.
2. Run syntax compilation for every added or modified Python file.
3. Run the revision protocol preflight and generic sweep dry-runs.
4. Record any GPU-only checks that remain blocked by the local PyTorch DLL issue.
5. Provide the ordered server execution commands and map their outputs back to reviewer comments.

The workspace is not currently recognized as a Git repository, so this plan omits commit steps and will preserve unrelated files by limiting edits to the paths listed above.
