# FARE 实验项目

FARE 是当前项目保留的唯一多模态公平推荐框架版本。代码已收敛为：

- `scripts/run_fare.py`: FARE 单次训练入口
- `scripts/build_revision_jobs.py`: IPM 修订实验的统一作业与 manifest 生成器
- `scripts/run_revision_jobs.py`: 按阶段或方法安全执行生成的作业
- `scripts/run_all_fare.sh`: 旧式便捷批量入口，不用于正式修订结果选择
- `src/model_fare.py`: FARE 模型定义
- `configs/fare_3090.yaml`: FARE 默认配置

FARE 结构为 ID sequential backbone 加多模态公平残差分支：

```text
score(u, i) = score_id(u, i) + residual_score_weight * fair_weight * score_mm(u, i)
```

训练公平机制使用 exposure-aware recommendation-loss reweighting。运行 FARE 前需要先准备一个参考 Top-K 文件，用于估计曝光分布。通常使用对应 ID backbone 在验证集上的 `topk_val.npz`。

注意：正式训练默认拒绝使用 `topk_test.npz` 作为 exposure reweighting 输入，避免测试集曝光信息泄漏到训练过程。只有诊断实验才应显式开启 `--allow_test_exposure_topk` 或 `ALLOW_TEST_EXPOSURE_TOPK=1`。

## 典型实验流程

1. 训练 ID backbones：

```bash
bash scripts/run_all_id_backbones.sh
```

2. 探索性运行可使用最近的 ID checkpoint 和 `topk_val.npz`：

```bash
bash scripts/run_all_fare.sh
```

3. 单次运行示例：

```bash
python scripts/run_fare.py \
  --dataset Video_Games \
  --config configs/fare_3090.yaml \
  --backbone sasrec \
  --init_backbone_checkpoint results/Video_Games/sasrec_id/<RUN_ID>/best_model.pt \
  --fair_rec_exposure_topk_path results/Video_Games/sasrec_id/<RUN_ID>/topk_val.npz \
  --run_id fare_video_sasrec
```

FARE 输出目录：

```text
results/<Dataset>/fare/<run_id>/
```

## 保留内容

项目仍保留 ID-only、多模态、训练期公平和后处理公平 baseline 代码，用于论文对比实验。旧的 FARE 前身版本入口、without-proxy 包装器、消融 launcher、旧参数敏感性脚本和旧配置已移除。

##Revision 完整协议

正式修订采用两级种子协议：breadth 层使用 2024--2026 三个种子覆盖 4 个数据集与 3 个骨干；confirmatory 层使用 2024--2029 六个配对种子，对三个 Amazon 数据集的 SASRec 以及 MicroLens-100K 的三个骨干做推断性验证。模型和政策只在验证集上选择，冻结后测试一次。

Coverage、RecGini、PopGap、BrandGap 和 ClusterGap 的唯一口径位于 `configs/revision/revision_protocol.yaml`。其中三个 group gap 均为 position-discounted exposure 相对 train-only utility 的 `utility_aware_gap`，选择器、论文表格和显著性脚本共同使用该定义。

正式入口：

```bash
python scripts/build_revision_jobs.py \
  --protocol configs/revision/revision_protocol.yaml \
  --id-registry revision_outputs/id_checkpoints.csv \
  --output-dir revision_outputs/generated

python scripts/run_revision_jobs.py \
  --manifest revision_outputs/generated/jobs.csv \
  --phase breadth_selection \
  --status-out revision_outputs/generated/job_status.csv \
  --dry-run
```

完整执行顺序、二阶段生成方式、检查点审计和统计产物见 [修订实验执行手册](docs/revision-experiment-protocol.md)。服务器上的 MicroLens 三骨干端到端检查使用 `scripts/server/smoke_microlens_three_backbones.sh`。


## Corrected revision completion runbook

For the GRU4Rec architecture-corrected IPM revision rerun and final evidence assembly, follow [docs/revision-completion-runbook.md](docs/revision-completion-runbook.md).
