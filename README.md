# FARE 实验项目

FARE 是当前项目保留的唯一多模态公平推荐框架版本。代码已收敛为：

- `scripts/run_fare.py`: FARE 单次训练入口
- `scripts/run_all_fare.sh`: 三个数据集、三个 ID backbone 的批量 FARE 实验入口
- `src/model_fare.py`: FARE 模型定义
- `configs/fare_3090.yaml`: FARE 默认配置

FARE 结构为 ID sequential backbone 加多模态公平残差分支：

```text
score(u, i) = score_id(u, i) + residual_score_weight * fair_weight * score_res(u, i)
```

训练公平机制使用 exposure-aware recommendation-loss reweighting。运行 FARE 前需要先准备一个参考 Top-K 文件，用于估计曝光分布。通常使用对应 ID backbone 在验证集上的 `topk_val.npz`。

注意：正式训练默认拒绝使用 `topk_test.npz` 作为 exposure reweighting 输入，避免测试集曝光信息泄漏到训练过程。只有诊断实验才应显式开启 `--allow_test_exposure_topk` 或 `ALLOW_TEST_EXPOSURE_TOPK=1`。

## 典型实验流程

1. 训练 ID backbones：

```bash
bash scripts/run_all_id_backbones.sh
```

2. 使用最近的 ID checkpoint 和 `topk_val.npz` 运行 FARE：

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
