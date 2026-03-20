# Tech Debt Tracker

本文件记录的是 debt bucket，不是任务系统。具体执行请在 `bd` 中建 issue。

## Runtime boundary debt

- 当前仍存在 “真实 IsaacLab runtime” 与 “本地 Newton-only shim” 双路径
- 这本身不是 bug，但需要持续防止 shim 膨胀成伪兼容层
- 相关设计依据见 `../harness.md`

## Geometry parity debt

- 当前 `tasks/obstacle_manager.py` 仅支持 sphere
- 尚未进入 harness contract 的能力包括 cube、wall、ceiling、ground plane
- 这些能力会直接影响 sensor realism 和 risk-loss 设计

## Sensor parity debt

- 当前支持 `relpos` / `camera` / `lidar`
- 尚无 IMU、mount error、sensor randomization、mixed-geometry sensing
- 这类能力应先补任务与几何 contract，再补传感器实现

## Dynamics parity debt

- Quadrotor body-rate 主路径已落地
- 仍缺更接近 reference 的 aerodynamic/randomization 深度
- 简化模型与全模型的职责分工还需要进一步固定

## Training surface debt

- 当前 feed-forward actor/critic 为主
- RNN actor/critic 仍未纳入正式 surface
- world 路径已可运行，但覆盖面仍窄于主线 APG/SHAC surface

## Tooling debt

- exporter / deploy / logger 等工具面仍弱于 reference
- 这部分不应阻塞核心训练 contract，但需要有明确入口和文档

## Rule

如果某项 debt 改变了公开 surface、验证门禁或 deferred/support 的边界，需要同步更新：

- `../progress.md`
- `../harness.md`
- 对应的 `bd` issue
