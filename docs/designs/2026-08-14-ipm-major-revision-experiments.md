# IPM Major Revision Experiment Design

## Understanding summary

- Build a protocol-correct revision experiment suite for the IPM major revision of FARE.
- Address statistical robustness, baseline validity, hyperparameter selection, ablation validity, exposure-prior circularity, cross-platform generalization, method composition, and efficiency.
- Reuse the existing training code wherever the scientific control is already expressible; add new behavior only where an existing runner cannot express the required experiment.
- Target the existing RTX 3090 experiment environment and the processed Amazon and MicroLens-100K datasets.
- Produce code, configurations, diagnostics, tests, and a reproducible execution plan. Manuscript and rebuttal text are outside this implementation task.

## Assumptions and non-functional requirements

- Formal runs use seeds 2024, 2025, and 2026 and keep test data unavailable to model or hyperparameter selection.
- Validation metrics select checkpoints and configurations; test metrics are read only after selection is frozen.
- Evaluation is full-sort and batched. Test-history masking uses the actual per-example prefix, including the validation interaction.
- MicroLens-100K has 100,000 users, 19,738 items, and 719,405 interactions; text and image features are already present under `data/Processed_MicroLens_100K`.
- The execution target is an NVIDIA RTX 3090 with 24 GB VRAM. Planning and non-Torch validation must remain runnable on the current Windows workstation.
- Only public datasets and local results are processed. No credentials, tokens, or private user data are required.
- Runs must be resumable, deterministic at the declared seed, and fail fast on missing checkpoints, test leakage, inconsistent users, or ineffective reranking.
- The implementation should remain maintainable by reusing `run_hparam_sweep.py` and existing model runners rather than introducing another training framework.

## Approaches considered

### Selected: protocol repair plus reusable audit tools

Repair the shared evaluator, extend FARE exposure sources, express new controls with existing runners, and add independent analysis tools for selection, inference, diagnostics, and efficiency. This changes the smallest common components while supporting a full rerun.

### Rejected: append-only revision experiments

Keep all existing Amazon results and add only MicroLens and statistical tables. This is cheaper, but it preserves inconsistent test-history masking and does not make the old and new results comparable.

### Rejected: rewrite all runners into one framework

Replace the current scripts with a single trainer and evaluator. This could improve long-term consistency but introduces excessive regression risk during a journal revision.

## Architecture

1. `src/eval_protocol.py` owns prefix-aware masking rules and is called by the shared full-sort evaluator.
2. `src/exposure_prior.py` builds exposure-derived group weights from a validation Top-K reference, train interactions, or a platform-level item prior.
3. Existing FARE and ModalityDebias runners receive only the CLI and construction changes needed for the ID-reweight control and FARE+MD composition.
4. `src/revision_statistics.py` contains NumPy/SciPy statistical and policy-selection functions. Scripts under `scripts/` turn run artifacts into CSV/JSON reports.
5. Revision YAML files under `configs/revision/` define the wider gamma sweep, independent-prior controls, MicroLens runs, and FARE+MD runs. The existing generic sweep runner executes them.
6. Diagnostic tools validate FairRR output changes, summarize FindRec learning/tuning behavior, and benchmark parameter count, time, latency, memory, and exposure-weight construction cost.

## Reviewer-comment map

| Reviewer issue | Classification | Implementation response | Acceptance evidence |
|---|---|---|---|
| R5-1 wider exposure sensitivity | evidence gap | wider gamma sweep and selection-weight sensitivity | dry-run grid and report tests |
| R5-2 reproducibility details | clarity/data-code | explicit revision configs and resolved manifests | configuration validation |
| R5-3/R6-7 cross-platform validation | evidence gap | MicroLens-100K baseline and FARE matrix | dataset preflight and run plan |
| R6-1/R3-1 significance and CIs | statistical, major | seed-level paired bootstrap, exact paired seed test, effect sizes | synthetic known-effect tests |
| R6-2/R3-3 FindRec anomaly | baseline validity, major | learning-curve, tuning, ranking-collapse diagnostics | fail-fast diagnostic report |
| R6-3 selection rule | methodological, major | validation-only selection and alpha sensitivity | test metrics excluded by schema |
| R6-4 tuning leakage | high-risk | validation-only selector; frozen selected config | leakage guard tests |
| R6-5 residual ablation | methodological, major | exposure-reweighted ID branch control | config and gradient-path checks |
| R6-6 circular exposure prior | methodological, major | train-popularity and platform-view priors | prior-source comparison matrix |
| R6-8 FARE+MD | evidence gap | ModalityDebias wrapper accepts pretrained FARE | construction/checkpoint tests |
| R3-2 FairRR unchanged | baseline validity, major | list-change, candidate-pool, set-change diagnostics | nonzero-change requirement |
| R3-7 efficiency | evidence gap | standardized efficiency benchmark | JSON/CSV resource report |

## Statistical protocol

- Pair runs by dataset, backbone, and seed.
- Compute per-user paired HR@10 and NDCG@10 differences from aligned Top-K archives.
- Use a seed-level paired bootstrap over the six matched random-seed differences. Report mean difference, percentile 95% CI, probability of improvement, and standardized paired effect.
- Also report the exact two-sided sign-flip test across seed-level differences. With six paired seeds, the minimum non-zero two-sided p-value is 0.03125.
- Keep the original seed/user hierarchical bootstrap implementation available as a sensitivity option for small subsets, but do not make it the default for full MicroLens/Baby significance reporting because it repeatedly recomputes aggregate Top-K fairness metrics over large test matrices.
- Do not interpret CI overlap as a significance test and do not claim utility preservation when the CI includes a practically meaningful loss.

## Decision log

- Correct the evaluation protocol before generating any revision result because current runners do not all mask the same history.
- Treat validation Top-K exposure as the primary FARE mechanism and train/platform priors as explicit circularity controls.
- Implement ID-reweighting through FARE with the residual disabled and the ID backbone unfrozen, avoiding a duplicate trainer.
- Implement FARE+MD as a ModalityDebias wrapper over a pretrained FARE model, with the FARE base frozen for an interpretable additive control.
- Keep proxy-group terminology in code and reports; platform views are an exposure proxy, not impression logs.
- Reuse the generic sweep runner and add configuration files instead of another GPU orchestration framework.

## Risks

- Corrected masking invalidates direct reuse of most earlier full-sort results; affected methods must be rerun.
- Three model seeds limit seed-level test power even with user-level bootstrap. Conclusions must reflect both sources of variation.
- The current workstation cannot import its installed PyTorch build, so GPU/model integration must be smoke-tested on the Linux experiment environment; local pure-Python, syntax, and artifact tests remain mandatory.
- FindRec may remain weak after tuning. If diagnostics show no adaptation defect, it should be reported as an unsuccessful reproduction rather than presented as a credible competitive SOTA value.
