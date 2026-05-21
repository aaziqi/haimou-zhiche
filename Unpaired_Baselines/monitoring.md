# 训练监测与里程碑（CUT 系列）

## 当前任务
- FastCUT：`euvp_fastcut_full`（训练于 `EUVP_Unpaired`）

## 你如何监测

### 1) 看训练日志（最直接）
路径：
`Unpaired_Baselines/checkpoints/euvp_fastcut_full/loss_log.txt`

看两点：
- `epoch:` 是否在增长（例如从 2→3→...）
- 同一个 epoch 内 `iters:` 是否在持续增长

### 2) 看 checkpoint 是否产出/更新（里程碑判据）
路径：
`Unpaired_Baselines/checkpoints/euvp_fastcut_full/`

关键文件：
- `latest_net_G.pth`（持续更新）
- `10_net_G.pth`（当训练完成 epoch=10 时出现：这是**第一个里程碑**）

当 `10_net_G.pth` 出现后，就代表可以开始跑：
1) EUVP test_samples 的推理输出
2) UIEB raw-890 的跨域推理输出
3) 统一评测脚本生成 `summary.json`
4) 写回论文表格并重新编译 PDF

## 我这边的下一步
- 等待训练到 epoch 10（以 `10_net_G.pth` 作为触发点）
- 自动进行推理 + 评测 + 更新论文

