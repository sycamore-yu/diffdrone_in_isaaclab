# Architecture Map

本仓当前的架构入口分成四层：

- `diffaero_newton/source/diffaero_newton/scripts/registry.py`
  - 统一暴露 algo / env / dynamics surface
- `diffaero_newton/source/diffaero_newton/envs/`
  - 任务环境、观测 contract、reset/step 语义
- `diffaero_newton/source/diffaero_newton/dynamics/`
  - point-mass / quadrotor 动力学与 Newton/Warp 可微推进
- `diffaero_newton/source/diffaero_newton/training/`
  - APG / PPO / SHAC / SHA2C / MASHAC / world 训练循环

文档入口如下：

- `docs/development.md`
  - 目标架构、接口约束、工程方向
- `docs/progress.md`
  - 当前主干真实能力、验证状态、已知缺口
- `docs/harness.md`
  - 当前 harness contract、参考实现决策、实现边界
- `docs/references/`
  - 参考代码映射与 IsaacLab 官方实践摘录

如果要判断“当前代码是否符合预期”，读取顺序应为：

1. `docs/progress.md`
2. `docs/harness.md`
3. `docs/development.md`

如果要新增能力，优先检查：

1. `scripts/registry.py`
2. 对应 `configs/*`
3. 对应 `envs/*` / `dynamics/*` / `training/*`
4. `docs/exec-plans/tech-debt-tracker.md`
