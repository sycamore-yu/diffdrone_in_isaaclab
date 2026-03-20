# Code Reference Map

本文件把常用 reference 代码映射到当前 `diffaero_newton` 模块，避免后续实现时只记得论文、不记得落点。

| Reference source | Key idea | Local module | Status |
| --- | --- | --- | --- |
| `reference/diffaero/env/obstacle_avoidance.py` | reward/loss 双通道、障碍任务语义、感知观测拼装 | `envs/obstacle_env.py`, `envs/drone_env.py`, `tasks/reward_terms.py`, `tasks/obstacle_manager.py` | 部分对齐，几何与 sensor parity 仍窄 |
| `reference/diffaero/env/racing.py` | gate-frame observation、gate progression、gate pass/collision | `envs/racing_env.py` | 已对齐核心任务语义 |
| `reference/diffaero/utils/sensor.py` | camera/lidar/relpos/IMU/mixed geometry sensing | `envs/sensors.py` | 仅迁移 relpos/camera/lidar，IMU 与 mixed geometry 缺失 |
| `reference/diffaero/utils/assets.py` | obstacle manager、障碍几何 ownership | `tasks/obstacle_manager.py` | 当前仅 sphere |
| `reference/diffaero/algo/SHAC.py` | 短视域可微训练 contract | `training/shac.py` | 已有本地实现 |
| `reference/rpg_flightning/flightning/objects/quadrotor_obj.py` | 全模型、allocation matrix、drag、集中式机体参数 | `dynamics/drone_dynamics.py` | 核心思想已吸收，细节未全量对齐 |
| `reference/rpg_flightning/flightning/objects/quadrotor_simple_obj.py` | 简化模型作为便宜 rollout / surrogate 路径 | `dynamics/pointmass_dynamics.py` 与后续简化 quadrotor 路径 | 方向明确，尚未形成正式双模型 contract |

## Reading note

优先提取的是“接口与职责”，不是字面实现。

错误做法：

- 直接照搬 reference 文件布局
- 把 reference 中的 deferred 能力写成当前主干已支持

正确做法：

- 先找 reference 的 contract
- 再决定应该落到 `configs/`、`dynamics/`、`tasks/`、`envs/` 还是 `training/`
