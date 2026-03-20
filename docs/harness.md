# DiffAero Newton Harness

本文件定义 `diffaero_newton` 的工程级 harness 约束。

目标不是再写一份泛泛的“迁移综述”，而是把当前代码库、参考实现和 IsaacLab 官方实践收敛成一个可执行的设计文档，供以下场景直接使用：

- 新增或重构 `envs/`、`dynamics/`、`training/` 时确认边界
- 审查某个实现是否把“当前 reality”和“目标 contract”写混
- 给后续补齐 mixed geometry、IMU、RNN actor、导出工具时提供稳定接口

## 1. 设计输入

本文件以三类证据为准：

- 本仓当前代码
  - `diffaero_newton/source/diffaero_newton/scripts/registry.py`
  - `diffaero_newton/source/diffaero_newton/envs/drone_env.py`
  - `diffaero_newton/source/diffaero_newton/envs/obstacle_env.py`
  - `diffaero_newton/source/diffaero_newton/envs/racing_env.py`
  - `diffaero_newton/source/diffaero_newton/dynamics/drone_dynamics.py`
  - `diffaero_newton/source/diffaero_newton/tasks/obstacle_manager.py`
  - `diffaero_newton/source/diffaero_newton/common/direct_rl_shim.py`
- 参考实现
  - `reference/diffaero/env/obstacle_avoidance.py`
  - `reference/diffaero/env/racing.py`
  - `reference/rpg_flightning/flightning/objects/quadrotor_obj.py`
  - `reference/rpg_flightning/flightning/objects/quadrotor_simple_obj.py`
- 官方 IsaacLab 实践
  - Context7 `/isaac-sim/isaaclab` 关于 `DirectRLEnv`、`@configclass`、`InteractiveSceneCfg(replicate_physics=True)`、显式声明 `action_space` / `observation_space` / `state_space` 的建议

## 2. 当前代码现实

先固定现状，避免文档把“目标态”误写成“已实现”。

| 模块 | 当前 reality | 结论 |
| --- | --- | --- |
| Runtime boundary | `common/isaaclab_launch.py` 负责真实 IsaacLab 启动，`common/direct_rl_shim.py` 负责 Newton-only headless 合约 | 这是双运行时边界，不是完整 IsaacLab 原生环境 |
| Env base | `DroneEnv.step()` 返回 `(obs, state, loss, reward, extras)`，不是 Gym 五元组 | 训练 harness 以项目私有 contract 为准 |
| Direct workflow | 当前大量环境继承本地 `DirectRLEnv` shim | 合理，但必须明确这是过渡层 |
| Config | `configs/obstacle_env_cfg.py` 用了 `configclass`，但底层仍是本地 shim 实现；`training_cfg.py` 还是普通 `dataclass` | 配置风格部分对齐 IsaacLab，尚未完全统一 |
| Dynamics | `dynamics/drone_dynamics.py` 已有 Newton + Warp autograd bridge 和 body-rate 控制链路 | Quadrotor 微分主路径已落地，但空气动力学和随机化仍远窄于参考 |
| Task geometry | `tasks/obstacle_manager.py` 仅支持 sphere `[x, y, z, radius]` | mixed geometry、ground plane、wall/ceiling 都还没进当前 harness |
| Perception | `envs/obstacle_env.py` 已支持 `relpos` / `camera` / `lidar` 三类观测，并向 world path 暴露 `state`/`perception` 拆分 | 当前 perception contract 已成形，应稳定下来 |
| Racing | `envs/racing_env.py` 已对齐 gate-frame 观测、gate pass/collision、OOB、reward/loss 拆分 | racing 已经不是空壳，但仍主要覆盖 point-mass 验证路径 |
| Registry surface | `scripts/registry.py` 统一暴露 algo / env / dynamics 组合 | 这是当前 harness 的正式入口面 |

## 3. Harness 的非协商 contract

后续实现必须围绕以下 contract 收敛。

### 3.1 运行时 contract

- 真实 IsaacLab 启动和项目内 Newton-only shim 必须继续分离。
- 不允许再引入“默默 fallback”的兼容层，把 IsaacLab 导入失败隐藏成普通 headless 路径。
- 任何需要真实场景复制、USD 资源、原生 IsaacLab 传感器的能力，都应走真实 IsaacLab runtime，而不是扩展 shim。

### 3.2 环境 contract

项目训练 harness 的环境步进以 `DroneEnv.step()` 的五元语义为准：

- `obs`: actor 输入，通常为 `{"policy": ...}`，在感知路径上可附带 `state` / `perception`
- `state`: critic 或 world model 使用的全局状态
- `loss`: 可微标量项，供短视域优化直接回传
- `reward`: detached 的 RL 记账信号
- `extras`: reset mask、terminated/truncated、next_obs_before_reset、任务诊断

关键约束：

- 不允许把 `loss` 与 `reward` 混成同一个匿名标量。
- `terminated` 和 `truncated` 必须在内部保持分离，即使某些 trainer 最后把它们合并处理。
- `next_obs_before_reset` / `next_state_before_reset` 这类 reset 前快照属于 harness contract，不是调试细节。

### 3.3 可微边界 contract

来自 DiffAero 的核心约束必须保留：

- rollout 主路径不隐藏 `detach()`
- 风险项、几何代价、短视域 loss 保持可微
- RL 统计、episode 指标、日志值默认 detached

来自当前代码的具体化要求：

- `dynamics/` 负责数值推进和控制映射
- `tasks/` 负责障碍几何、距离查询、reward/loss term 组成
- `envs/` 负责 reset、episode cadence、观测拼装、extras 输出
- `training/` 负责 horizon accumulation、bootstrap、optimizer step

### 3.4 配置 contract

目标配置风格采用 IsaacLab direct workflow 的显式配置方式，但要承认当前实现是“部分到位”：

- 环境配置继续优先使用 `@configclass`
- 每个环境配置必须显式声明
  - `sim.dt`
  - `decimation`
  - `episode_length_s`
  - `num_envs`
  - `action_space`
  - `observation_space`
  - `state_space` 或 `num_states`
- 训练配置可以暂时保留 `dataclass`，但字段命名和默认值应与环境配置并行清晰，不要再回到 Hydra 风格的隐式层叠

### 3.5 Registry contract

`scripts/registry.py` 是当前 harness 的稳定外表面。

新增能力时必须同步以下三层：

- registry 暴露名
- 对应 config builder
- 对应 smoke 验证入口

不允许出现“代码里能 import，但不在 registry 中可发现”的半接入状态。

## 4. 来自参考实现的明确决策

这一节回答“什么应该保留，什么应该重写”。

### 4.1 来自 DiffAero 的保留项

应该保留的不是 Hydra 或原文件组织，而是以下语义：

- `obstacle_avoidance.py` 的 reward/loss 双通道设计
- `racing.py` 的 gate-frame 观测、gate pass/collision、target gate advancement
- 传感器输出与任务 loss 的解耦
- 短视域训练对 differentiable loss 的依赖

因此：

- 保留任务语义与 loss 结构
- 重写环境壳层和运行时对接
- 不以 1:1 复制 reference 文件结构为目标

### 4.2 来自 Flightning 的保留项

`quadrotor_obj.py` 和 `quadrotor_simple_obj.py` 提供了两个重要工程启发：

- 全模型与简化模型应当同时存在，并各自承担不同职责
- 机体参数、分配矩阵、drag 语义要在一个中心位置维护

因此本仓 quadrotor harness 的方向应是：

- 保持当前 Newton 全模型主路径
- 允许简化模型继续作为便宜 rollout 或 surrogate 梯度路径
- 不要把 rotor geometry、allocation matrix、控制参数散落到环境层

### 4.3 来自 IsaacLab 的保留项

官方 direct workflow 实践强调：

- 自定义任务在语义尚未稳定前，优先使用 `DirectRLEnv`
- `@configclass` 是推荐配置载体
- `InteractiveSceneCfg(num_envs=..., env_spacing=..., replicate_physics=True)` 应显式声明
- observation/action/state space 要显式定义，而不是靠运行期推断

因此本仓的清晰做法是：

- 继续坚持 direct workflow
- 但把“本地 shim”和“真实 IsaacLab DirectRLEnv”明确区分
- 不要提前上 manager-based abstraction

## 5. 当前 harness 的模块职责图

```text
scripts/registry.py
  -> configs/*
  -> envs/*
      -> dynamics/*
      -> tasks/*
      -> envs/sensors.py
  -> training/*
```

推荐的职责分割如下：

- `dynamics/`
  - 控制输入到 body wrench 的映射
  - 单步积分与可微 rollout
  - 与 Newton/Warp 的 autograd bridge
- `tasks/`
  - obstacle geometry ownership
  - nearest distance / collision / gate progress 等任务几何
  - reward term / loss term 原子项
- `envs/`
  - reset policy
  - observation 组装
  - step cadence
  - extras 汇总
- `training/`
  - SHAC/APG/PPO/MASHAC/world 的 rollout buffer、bootstrap、更新步骤
- `configs/`
  - 与 registry surface 一一对应的显式配置

## 6. 当前明确的缺口和约束

这里不是任务列表，而是 harness 级限制说明。

### 6.1 仍未进入 harness contract 的能力

- mixed geometry: cube / wall / ceiling / ground plane
- IMU
- richer sensor mount / randomization
- RNN actor/critic
- 完整 exporter / deploy surface
- 更接近 DiffAero 的 quadrotor aerodynamic/randomization parity

这些能力在落地前都不应写进“当前支持”。

### 6.2 明确禁止的实现方式

- 在 sensor 模块里偷偷定义最终风险函数
- 在 rollout helper 内部隐式 `detach`
- 在 environment 里同时混入 reset、几何查询、optimizer state、日志聚合
- 把 shim 扩成一个伪 IsaacLab 全兼容层
- 用文档把目标态描述成主干已具备能力

## 7. 验证门禁

文档变更不应脱离现有门禁结构。

当前 harness 继续沿用三层验证：

- `runtime_preflight`
- `cpu_smoke`
- `gpu_smoke`

加上一层手工统一入口验证：

- `scripts/train.py --algo <algo> --env <env> --dynamics <dyn>`

新增能力进入 harness 时，至少要回答四个问题：

1. registry 名称是什么
2. config builder 在哪里
3. 最低 smoke 用例是什么
4. 它属于当前支持能力，还是明确 deferred 能力

## 8. 文档维护规则

当以下内容变化时，必须同步更新本文件：

- `scripts/registry.py` 的公开 surface
- `envs/` 的步进 contract
- `dynamics/` 的控制语义或可微边界
- `tasks/obstacle_manager.py` 的几何能力范围
- 运行时边界是否仍然是 “IsaacLab launch + Newton shim” 双路径

相关文档分工：

- `docs/development.md`: 目标态架构与工程方向
- `docs/progress.md`: 当前已实现与已验证事实
- `docs/harness.md`: 把目标态和现实态压成稳定 contract
- `docs/references/`: 参考代码与官方实践摘录

## 9. 一句话结论

`diffaero_newton` 的 harness 不应再被描述为“把 DiffAero 搬到 IsaacLab 上”。

更准确的定义是：

一个以 `scripts/registry.py` 为入口、以 `DroneEnv.step()` 五元语义为训练 contract、以 Newton/Warp quadrotor rollout 为可微核心、以本地 direct workflow shim 为当前运行壳层、并逐步向 IsaacLab 官方 direct workflow 收敛的显式工程栈。
