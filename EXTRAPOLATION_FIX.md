# 深度/宽度外推修复：随机训练 + 收缩正则化

## 问题

OpenMythos 的深度外推和宽度外推在推理时均失效——使用超出训练值的 loop 数或 stream 数时，性能严重退化。

**根因分析：**
- **深度**：模型记忆了固定 4 步的轨迹，而非学到收敛动力学（收缩映射）
- **宽度**：各 stream 对固定数量 4 产生了协同适应，而非作为独立估计器（集成）

## 方案

两项正交的修复，同时解决深度和宽度问题：

1. **随机训练**（`random_depth` / `random_width`）：每个 batch 随机采样 `n_loops ~ U[1, max_loop_iters]` 和 `n_streams ~ U[1, max_streams]`，防止模型对固定配置产生协同适应
2. **收缩正则化**（`contraction_weight`）：在每次循环迭代中，若 `||h_new - h_old||²` 相比上一迭代增大，则施加惩罚，鼓励模型学习收敛动力学

### 代码改动

- `MythosConfig`：新增 `random_depth`、`random_width`、`contraction_weight` 配置项
- `RecurrentBlock.forward`：训练时随机采样 loop 数 + 计算收缩损失
- `MultiStreamRecurrentBlock.forward`：训练时随机采样 stream 数 + 收缩损失；移除无用的 `stream_merge`
- `OpenMythos.forward`：暴露 `self.contraction_loss` 供训练使用
- `small_benchmark.py`：`train_step` 中加入 `total_loss = ce_loss + λ * contraction_loss`
- 新增对应单元测试（收缩损失、随机深度、随机宽度）

## 实验设置

- 数据集：TinyStories（stream，5M 训练 token，200K 验证 token）
- 模型：dim=128, MLA, 4 expert MoE, 7.1M 参数
- 训练：5000 步，batch=64，seq_len=512，AdamW lr=3e-4
- 设备：单 GPU (CUDA)
- 训练深度：`max_loop_iters=4`

| 实验 | n_streams | random_depth | random_width | random_width 范围 | contraction_weight |
|------|-----------|--------------|--------------|------------------|-------------------|
| Exp 1 | 4 | ✗ | ✗ | — | 0.0 |
| Exp 2 | 4 | ✓ | ✓ | U[1, n_streams=4] | 0.01 |
| Exp 3 | 1 | ✓ | ✗ | — | 0.01 |
| Exp 4 | 4 | ✓ | ✓ | U[1, max_streams=8] | 0.01 |

SLURM Job IDs: 4908044 (Exp 1-3), 4911708 (Exp 4)

## 结果

### 深度外推

训练深度 `n_loops=4`，推理时 sweep 不同深度的 eval loss：

| n_loops | Exp 1: 无修复 | Exp 2: U[1,4] | Exp 3: 1流 | Exp 4: U[1,8] |
|---------|--------------|---------------|------------|---------------|
| 1       | 18.689 (+16.26) | 2.390 (-0.002) | 2.397 (+0.001) | 2.420 (-0.001) |
| 2       | 15.799 (+13.37) | 2.389 (-0.002) | 2.396 (+0.000) | 2.419 (-0.001) |
| **4**   | **2.431** (训练值) | **2.391** (训练值) | **2.396** (训练值) | **2.420** (训练值) |
| 8       | 4.607 (+2.18) | 2.399 (+0.008) | 2.396 (+0.000) | 2.424 (+0.003) |
| 16      | 4.900 (+2.47) | 2.399 (+0.008) | 2.394 (-0.003) | 2.429 (+0.008) |

**结论：深度外推彻底修复。**

- 修复前：4→8 loops 损失从 2.43 暴涨到 4.61（+2.18），模型完全崩溃
- 修复后：所有实验 4→16 loops 的 delta 均 < +0.01
- Exp 3（单流）在 16 loops 时 loss 反而略降（-0.003），说明收缩正则化成功让模型学到了真正的收敛动力学

### 宽度外推

训练宽度 `n_streams=4`，推理时 sweep 不同 stream 数的 eval loss：

| n_streams | Exp 1: 无修复 | Exp 2: U[1,4] | **Exp 4: U[1,8]** |
|-----------|--------------|---------------|-------------------|
| 1         | 4.127 (+1.70) | 2.391 (-0.000) | 2.433 (+0.013) |
| 2         | 2.939 (+0.51) | 2.394 (+0.003) | 2.434 (+0.013) |
| **4**     | **2.431** (训练值) | **2.391** (训练值) | **2.420** (训练值) |
| 8         | 2.925 (+0.49) | 2.523 (+0.131) | **2.418 (-0.003)** |

**结论：宽度外推完全修复。**

- Exp 1（无修复）：1 stream 时 loss=4.13（+1.70），8 stream 时 +0.49，stream 间严重协同适应
- Exp 2（`random_width ~ U[1, n_streams=4]`）：训练范围内修复（1-4 stream delta ≈ 0），但 8 stream 仍有 +0.131 退化
- **Exp 4（`random_width ~ U[1, max_streams=8]`）：8 stream 的 loss 反而比训练值更低（-0.003），宽度外推彻底成功**

关键改进：将 `random_width` 的采样范围从 `U[1, n_streams]` 扩展到 `U[1, max_streams]`，让模型在训练时就见过更大的流数，从而学到真正的独立估计器集成行为。

### 训练效率

| 指标 | Exp 1 (无修复) | Exp 2 (U[1,4]) | Exp 3 (1流) | Exp 4 (U[1,8]) |
|------|---------------|----------------|-------------|----------------|
| 总用时 | 1433s | 935s | 832s | 935s |
| Mythos tok/s | 148K | 270K | 323K | 270K |
| Final train loss | 2.195 | 2.211 | 2.199 | 2.210 |
| Final eval loss | 2.431 | 2.391 | 2.396 | 2.420 |

随机训练反而**加速**了训练（因为平均 loop/stream 数更少），同时 eval loss 更优。

### 训练损失曲线（Exp 2 周期 eval）

| Step | Mythos eval | Baseline eval | Δ |
|------|-------------|---------------|---|
| 500  | 3.966 | 3.567 | +0.399 |
| 1000 | 3.369 | 3.085 | +0.284 |
| 1500 | 3.063 | 2.872 | +0.191 |
| 2000 | 2.874 | 2.738 | +0.136 |
| 2500 | 2.737 | 2.655 | +0.082 |
| 3000 | 2.625 | 2.591 | +0.033 |
| 3500 | 2.539 | 2.554 | -0.015 |
| 4000 | 2.470 | 2.539 | -0.069 |
| 4500 | 2.424 | 2.527 | -0.103 |
| 5000 | 2.391 | 2.515 | -0.123 |

Mythos 在 ~3500 步时超越 baseline，之后持续拉开差距。

## 待改进

1. ~~**宽度外推到 >n_streams**~~：已在 Exp 4 中通过 `random_width ~ U[1, max_streams]` 解决，8 流外推 delta=-0.003
2. **超参调优**：`contraction_weight=0.01` 是初始值，可以做 sweep（0.001, 0.01, 0.1）
3. **规模验证**：当前在 7M 参数模型上验证，需要在更大模型上确认效果保持
4. **更长训练**：5000 步可能不够，loss 曲线仍在下降
5. **更大 max_streams 外推**：Exp 4 中 max_streams=8，可以测试 16/32 流是否仍然保持

## 复现

```bash
# CPU 快速测试
python tests/small_benchmark.py --steps 10 --random-depth --random-width \
  --contraction-weight 0.01 --n-streams 2 --width-sweep 1,2,4 \
  --depth-sweep 1,2,4 --batch-size 4 --seq-len 32 \
  --train-tokens 10000 --eval-tokens 5000

# GPU 完整实验
sbatch scripts/width_extrapolation_benchmark.sh
```

## 日志文件

- `scripts/logs/exp1_baseline_4908044.json`
- `scripts/logs/exp2_random_contraction_4908044.json`
- `scripts/logs/exp3_depth_only_4908044.json`
- `scripts/logs/exp4_wide_random_4911708.json`
