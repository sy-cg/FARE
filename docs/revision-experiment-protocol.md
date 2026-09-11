# Revision 实验执行手册

本手册以 `configs/revision/revision_protocol.yaml` 为唯一协议源。任何正式结果都必须满足：验证集选择、冻结政策、一次性测试、配对种子、完整产物审计。

## 1. 固定协议

- Breadth：种子 2024、2025、2026；4 个数据集 × 3 个骨干，用于完整泛化矩阵、超参数选择和控制实验。
- Confirmatory：种子 2024--2029；三个 Amazon 数据集仅确认 SASRec，MicroLens-100K 确认 SASRec、GRU4Rec、BERT4Rec。
- 6 个配对种子使双侧精确 sign-flip 检验的最小非零 p 值达到 0.03125；3-seed breadth 结果仍用于描述性均值、标准差和覆盖面。
- 模型选择指标为验证集 NDCG@10；FARE 政策选择采用 `0.5 × utility + 0.5 × exposure`，其他 alpha 只做敏感性分析。
- 正式公平指标为 Coverage、RecGini，以及 PopGap、BrandGap、ClusterGap。三个 gap 均取 `utility_aware_gap`，即位置折扣曝光份额相对训练集 utility 份额之比的 max-minus-min。

先运行预检：

```bash
python scripts/validate_revision_protocol.py \
  --config configs/revision/revision_protocol.yaml \
  --output revision_outputs/preflight.json
```

## 2. Breadth ID 与安全注册表

```bash
DATASETS="Video_Games Musical_Instruments Baby_Products MicroLens_100K" \
BACKBONES="sasrec gru4rec bert4rec" \
SEEDS="2024 2025 2026" \
bash scripts/run_all_id_backbones.sh

python scripts/collect_id_checkpoint_registry.py \
  --results-root results \
  --datasets-config configs/datasets.yaml \
  --output revision_outputs/id_checkpoints.csv
```

注册器只接受 `run_name=<backbone>_id`、匹配的 `model_arg` 和验证集 best metric；FARE、Adv、MD 等目录即使分数更高也不会进入 ID 注册表。

## 3. 第一阶段统一作业生成

```bash
python scripts/build_revision_jobs.py \
  --protocol configs/revision/revision_protocol.yaml \
  --id-registry revision_outputs/id_checkpoints.csv \
  --output-dir revision_outputs/generated_phase1
```

第一阶段会生成：

- 每个 breadth dataset/backbone/seed 的 FARE 宽 gamma 验证 sweep；
- FairRR 控制；
- 每个 dataset/seed 的 FindRec validation-only 调参 sweep；
- confirmatory 层尚缺少的 ID 作业；
- `jobs.csv`、`significance_manifest.csv`、`efficiency_manifest.csv` 和 `generation_audit.json`。

MicroLens sweep 自动使用 `configs/revision/microlens_fare_3090.yaml`，不会误用包含 Amazon-only group 的默认配置。

可先干跑，再按阶段执行：

```bash
python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase1/jobs.csv \
  --phase breadth_selection \
  --status-out revision_outputs/generated_phase1/status.csv \
  --dry-run

python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated_phase1/jobs.csv \
  --phase breadth_selection \
  --status-out revision_outputs/generated_phase1/status.csv \
  --resume --continue-on-error
```

## 4. 验证集选择

```bash
python scripts/collect_sweep_run_registry.py \
  --sweeps-root sweeps/revision \
  --output revision_outputs/fare_validation_runs.csv

python scripts/collect_revision_validation.py \
  --registry revision_outputs/fare_validation_runs.csv \
  --output revision_outputs/fare_validation_metrics.csv

python scripts/select_revision_runs.py \
  --input revision_outputs/fare_validation_metrics.csv \
  --selected-out revision_outputs/selected_configs.csv \
  --sensitivity-out revision_outputs/policy_sensitivity.csv \
  --audit-out revision_outputs/selection_audit.json

python scripts/materialize_selected_runs.py \
  --selected revision_outputs/selected_configs.csv \
  --run-registry revision_outputs/fare_validation_runs.csv \
  --output revision_outputs/selected_seed_runs.csv
```

上述脚本默认从协议读取 breadth seeds。收集器读取 `utility_aware_gap`，不再使用 `exposure_share_gap`。

FindRec 调参证据单独汇总：

```bash
python scripts/collect_findrec_tuning.py \
  --sweep-results sweeps/findrec_*/sweep_results.csv \
  --trials-out revision_outputs/findrec_tuning_trials.csv \
  --selected-out revision_outputs/findrec_selected.csv \
  --audit-out revision_outputs/findrec_tuning_audit.json
```

该脚本发现 `test_final`、`test_ranking.npz` 或 `topk_test.npz` 会直接判定调参泄漏。
对 `findrec_selected.csv` 中的每个 `run_dir`，只执行一次冻结 checkpoint 测试：

```bash
python scripts/evaluate_selected_findrec.py \
  --run-dir <SELECTED_FINDREC_RUN_DIR> --device cuda
```

该入口会记录 checkpoint/config SHA-256，并生成 `test_evaluation_audit.json`；已有测试产物时默认拒绝重复执行。

## 5. 第二阶段控制、组合与 confirmatory

补齐 confirmatory ID 后重新生成 ID 注册表，然后带上冻结的 FARE 选择结果再次生成：

```bash
python scripts/build_revision_jobs.py \
  --protocol configs/revision/revision_protocol.yaml \
  --id-registry revision_outputs/id_checkpoints_all_seeds.csv \
  --selected-fare-registry revision_outputs/selected_seed_runs.csv \
  --selected-findrec-registry revision_outputs/findrec_selected.csv \
  --output-dir revision_outputs/generated_phase2
```

第二阶段增加：

- `ExposureReweight-ID`：移除多模态残差，使用同一验证选定 gamma/group；
- `FARE-IndependentPrior`：Amazon 使用 train popularity，MicroLens 使用 platform views；
- `FARE+ModalityDebias`：加载冻结的 validation-selected FARE；
- `FARE-Confirmatory`：对新增种子复用已冻结政策并直接进行一次性最终评估。

FARE+MD 会检查来源 checkpoint 的方法、数据集、骨干、item 数、关键结构字段和 SHA-256，并要求目标 state keys 完整加载；随后生成与 FARE 相同口径的验证/测试公平表。

## 6. 显著性与效率

当 `generation_audit.json` 中 confirmatory pairs 全部 ready 后：

```bash
python scripts/analyze_revision_significance.py \
  --manifest revision_outputs/generated_phase2/significance_manifest.csv \
  --output-csv revision_outputs/significance.csv \
  --output-json revision_outputs/significance_audit.json \
  --repetitions 10000 --bootstrap-seed 2026 --ci-mode seed

python scripts/benchmark_revision_efficiency.py \
  --manifest revision_outputs/generated_phase2/efficiency_manifest.csv \
  --output revision_outputs/efficiency.csv \
  --audit revision_outputs/efficiency_audit.json
```

显著性脚本输出 HR/NDCG 的 method-minus-baseline、Coverage 的 method-minus-baseline、RecGini 与 utility-aware gaps 的 baseline-minus-method，并默认报告 seed-level paired bootstrap 95% CI、改善概率、配对效应量及 exact sign-flip p 值。Seed-level CI 将 6 个配对随机种子作为独立实验单位，计算更快，也更直接回应审稿人对 random-seed robustness 的质疑。如需运行原 seed/user 两层 bootstrap 敏感性检查，可改用 `--ci-mode hierarchical`，但在 MicroLens/Baby 等大 test split 上会明显变慢。

效率表包含参数量、训练时间、测试吞吐、每用户时延、CUDA 峰值显存、软硬件版本；FARE 还强制要求 `exposure_weight_build_sec`。

## 7. 服务器三骨干 smoke

```bash
DRY_RUN=1 bash scripts/server/smoke_microlens_three_backbones.sh
bash scripts/server/smoke_microlens_three_backbones.sh
```

脚本依次对 MicroLens-100K 的 SASRec、GRU4Rec、BERT4Rec 运行 1 epoch ID 与 FARE，并检查 checkpoint、验证/测试 Top-K、公平 flat metrics，最终写入 `revision_outputs/server_smoke/smoke_audit.csv`。

