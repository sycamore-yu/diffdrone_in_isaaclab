# `diffaero_newton/` Review And Remediation Plan

This document captures the current review conclusions for `diffaero_newton/` and defines the required remediation work for the next Claude update cycle.

It is intended to be execution-oriented: each issue is tied to concrete code areas, and each remediation phase has acceptance criteria.

## Current Status

The following smoke-level runtime path is now working in the `isaaclab-newton` conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-newton
PYTHONPATH=diffaero_newton/source pytest -q diffaero_newton/tests/test_obstacle_training.py
PYTHONPATH=diffaero_newton/source python diffaero_newton/run_training.py --num_envs 4 --num_iterations 1 --save_interval 1000
```

At the time of writing:

- `pytest` passes
- the training CLI completes 1 iteration

This only means the basic code path runs. It does **not** mean the implementation matches `reference/diffaero/` semantics or the project best practices defined in [development.md](/home/tong/tongworkspace/diffdroneinisaac_workspace/docs/development.md).

## Confirmed Problems

### 1. Obstacle task is not integrated into the actual environment loop

Observed in:

- `diffaero_newton/source/diffaero_newton/envs/drone_env.py`
- `diffaero_newton/source/diffaero_newton/tasks/obstacle_manager.py`
- `diffaero_newton/source/diffaero_newton/tasks/reward_terms.py`

Problem:

- `DroneEnv.step()` currently behaves like a goal-tracking environment with ground termination only.
- obstacle spawning, collision checks, nearest-distance queries, and differentiable risk terms are not wired into the environment transition loop.
- `compute_risk_loss()` exists but is effectively unused by the main training path.

Mismatch vs reference:

- `reference/diffaero/env/obstacle_avoidance.py` integrates obstacle manager, sensor/task observation, loss, reward, reset, and diagnostics inside the environment workflow.

Impact:

- the project does not yet implement the stated “obstacle environment task interface”
- the current reward and done signals do not reflect obstacle avoidance behavior

### 2. Dynamics model is not aligned with DiffAero quadrotor dynamics

Observed in:

- `diffaero_newton/source/diffaero_newton/dynamics/drone_dynamics.py`
- `diffaero_newton/source/diffaero_newton/dynamics/rollout.py`

Problem:

- the current `Drone` model is a simplified vertical-thrust point-mass style integrator
- orientation and angular velocity are not physically integrated in a quadrotor-consistent way
- motor inputs are collapsed into a single z-axis force
- rollout helpers do not preserve the full state semantics expected of a quadrotor model

Mismatch vs reference:

- `reference/diffaero/dynamics/quadrotor.py` models thrust, torque, angular dynamics, quaternion derivative, and solver stepping

Impact:

- “quadrotor dynamics” is currently an overstatement
- policy behavior and loss landscape will differ materially from `reference/diffaero`

### 3. Training algorithm is only “SHAC-style”, not actually aligned with `reference/diffaero/algo/SHAC.py`

Observed in:

- `diffaero_newton/source/diffaero_newton/training/shac.py`
- `diffaero_newton/source/diffaero_newton/training/buffer.py`

Problem:

- current implementation is closer to PPO-style policy optimization with a critic that consumes `(obs, action)`
- actor update is based on clipped surrogate loss, not DiffAero-style short-horizon differentiable loss accumulation
- environment does not expose the `loss` / `next_obs_before_reset` style training contract used by the reference SHAC implementation

Mismatch vs reference:

- `reference/diffaero/algo/SHAC.py` accumulates differentiable horizon loss, performs terminal bootstrap with environment-provided tensors, and separates actor loss from detached reward accounting

Impact:

- the training method is not equivalent to the reference algorithm
- current implementation cannot be claimed as a faithful SHAC reproduction

### 4. Environment/task interfaces do not fully follow the project best-practice contract

Observed in:

- `diffaero_newton/source/diffaero_newton/envs/drone_env.py`
- `diffaero_newton/source/diffaero_newton/tasks/observations.py`
- `diffaero_newton/source/diffaero_newton/configs/drone_env_cfg.py`

Problem:

- `obs` is available, but `state`, `loss_terms`, and richer task diagnostics are not consistently exposed
- obstacle observations exist in helpers but are not connected to the main env output
- config style imitates IsaacLab naming but is not a real `@configclass` pattern
- some naming and interface claims still overstate IsaacLab/Newton alignment

Impact:

- the code is harder to extend toward asymmetric critics or differentiable training
- public behavior is inconsistent with the architecture doc and review expectations

### 5. Documentation and git hygiene are still inconsistent

Observed in:

- `diffaero_newton/README.md`
- `diffaero_newton/docs/api.md`
- `CLAUDE.md`
- git working tree / tracked artifacts

Problem:

- docs still describe the project as if Newton and IsaacLab are fully integrated, which is not true today
- generated artifacts such as `__pycache__`, `.pyc`, and `checkpoints/` should not be part of committed deliverables
- working tree contains unrelated or generated modifications that should be cleaned before landing final changes

Impact:

- future contributors will over-assume implementation maturity
- git history will be noisy and hard to review

## Required Remediation Plan

Implement the following phases in order.

### Phase 1. Wire obstacle task semantics into `DroneEnv`

Goal:

- turn `DroneEnv` from a goal-tracking env into an actual obstacle-avoidance task

Required changes:

- instantiate and own `ObstacleManager` inside `DroneEnv`
- include obstacle-aware observations in the policy and, if needed, critic path
- incorporate obstacle collision checks into termination logic
- compute nearest-distance and collision-related diagnostics every step
- expose obstacle-related data in `extras`
- connect task reward and differentiable risk/loss terms to the env outputs

Acceptance criteria:

- environment reward changes when obstacle proximity changes
- collisions with obstacles trigger task-relevant penalties and/or termination
- `extras` exposes obstacle diagnostics needed for debugging

### Phase 2. Upgrade dynamics toward DiffAero quadrotor semantics

Goal:

- replace the current simplified integrator with a materially closer quadrotor model

Required changes:

- preserve full state through rollout and reset paths
- model orientation and angular velocity evolution explicitly
- map motor commands to thrust and torque, not only a summed z-force
- keep the implementation differentiable in PyTorch unless a real Newton-backed path is introduced
- if Newton is not actually used at runtime, stop claiming Newton-backed dynamics in docs until that changes

Acceptance criteria:

- state update includes position, quaternion, linear velocity, and angular velocity
- rollout functions accept and return full-state transitions without silently discarding components
- unit tests cover nontrivial orientation / angular-rate updates

### Phase 3. Align training semantics with reference SHAC

Goal:

- move from “runs one PPO-like loop” to a training contract closer to `reference/diffaero/algo/SHAC.py`

Required changes:

- define environment outputs needed by SHAC-style actor loss accumulation
- separate detached RL reward from differentiable training loss
- accumulate short-horizon loss across rollout
- implement terminal / truncated bootstrap behavior using environment-provided tensors
- re-evaluate critic interface so it matches the intended algorithm rather than the current ad hoc `(obs, action) -> value` hybrid

Acceptance criteria:

- actor update depends on horizon loss accumulation, not only PPO-style clipping
- training loop can explain how it maps to `reference/diffaero/algo/SHAC.py`
- tests cover horizon accumulation and bootstrap edge cases

### Phase 4. Bring interfaces and docs back into alignment

Goal:

- ensure public docs describe the implementation that actually exists

Required changes:

- update `README.md`, `docs/api.md`, and `docs/quickstart.md`
- state clearly whether dynamics are:
  - pure PyTorch differentiable
  - Newton-inspired
  - or truly Newton-backed
- document current limitations explicitly
- ensure config examples and observation dimensions match real code

Acceptance criteria:

- no doc claims conflict with current code behavior
- quickstart commands run in `isaaclab-newton`

### Phase 5. Clean git hygiene before final landing

Goal:

- leave a reviewable tree with no generated junk committed

Required changes:

- remove tracked `.pyc` and `__pycache__` artifacts from version control
- add ignore rules if missing
- avoid checking in generated `checkpoints/`
- keep `AGENTS.md`, `CLAUDE.md`, and docs synchronized with actual workflow

Acceptance criteria:

- `git status` shows only intended source/doc changes
- no generated Python bytecode is tracked

## Required Test Matrix

All remediation work should be validated in the `isaaclab-newton` environment.

### Minimum commands

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-newton
export PYTHONPATH=diffaero_newton/source

pytest -q diffaero_newton/tests/test_obstacle_training.py
python diffaero_newton/run_training.py --num_envs 4 --num_iterations 1 --save_interval 1000
```

### Additional tests that should be added

- obstacle collision test inside the real env step loop
- reward sensitivity test for obstacle proximity
- differentiable short-horizon loss accumulation test
- rollout fidelity test showing nontrivial orientation / angular-rate evolution
- reset test for partial env resets
- documentation smoke test for quickstart commands

## Implementation Notes For Claude

- Do not treat the current passing tests as proof of semantic correctness. They only prove the runtime path is no longer broken.
- Fix the environment-task contract before attempting deeper policy tuning.
- If a true Newton-backed implementation is out of scope for the next pass, explicitly document that the current backend is PyTorch-only and stop using wording that implies otherwise.
- Preserve the project contract from [development.md](/home/tong/tongworkspace/diffdroneinisaac_workspace/docs/development.md): keep `obs`, `state`, differentiable loss outputs, detached rewards, and diagnostics clearly separated.

# diffaero_newton 代码审查报告 (Code Review Report)

对 `diffaero_newton` 目录中 ClaudeCode 提交的工作进行了全面审查，主要对比了 `reference/diffaero` 的参考实现以及 `newton-isaaclab-diffaero` 的最佳实践指南。

## 1. 动力学模型 (Dynamics Model) - ❌ 不一致且存在严重缺陷

`diffaero_newton/source/diffaero_newton/dynamics/drone_dynamics.py` 的实现与 `reference/diffaero/dynamics/quadrotor.py` 存在巨大差异，且当前的 `drone_dynamics.py` 并没有实现真正的四旋翼飞行器动力学：

- **缺失姿态更新与旋转物理 (No Rotational Dynamics):**
  - 在 `integrate()` 方法中，姿态（Quaternion）完全没有进行更新。
  - 角速度 (`omega`) 只是做了一个简单的衰减阻尼控制 `self._state[:, 10:13] * 0.95`，并没有计算任何与电机推力差相关的力矩 (Torque)。
- **缺失控制分配 (No Control Allocation):**
  - 没有通过 4 个电机的分别输入计算 $x, y, z$ 轴上的扭矩。它仅仅把 4 个控制量求和转换为垂直方向 ($Z$ 轴) 的升力 `total_thrust = self._last_thrust.sum(dim=1) * 20.0`。
  - 这种做法使得该模型退化成了一个**带有衰减的 3D 质点 (Point Mass) 模型**，而不再是四旋翼 (Quadrotor) 动力学。
- **缺失 Newton 求解器 (No Newton Solver System):**
  - `reference/diffaero` 使用了支持 Euler 和 RK4 的积分器系统，并有精确的偏导/梯度截断控制；而 `diffaero_newton` 直接硬编码了粗糙的 Euler 加法 `vel_new = vel + acc * dt`。

## 2. 训练方法 (Training Method) - ❌ 与 SHAC 原理不符 (实为 PPO)

`diffaero_newton/source/diffaero_newton/training/shac.py` 实际上并未实现 SHAC (Scalable High-Actor-Critic) 算法：

- **未利用可微环境进行 BPTT:** 真正的 SHAC 的核心思想是利用环境的可微性，直接通过动力学展开将 Actor 的梯度从环境中反向传播 (BPTT, Backpropagation Through Time) 回来。例如在 `reference` 中，`cumulated_loss.backward()` 可以直接指导策略优化。
- **错误地实现了 PPO (Proximal Policy Optimization):** 当前代码中的 `_update_actor` 方法使用的是典型的 **PPO GAE + Clipped Surrogate Objective** (`torch.min(surr1, surr2)`)。这是一个 Model-Free 的强化学习算法，完全没有利用 `drone_dynamics.py` 中定义的 `requires_grad=True` 所带来的可微特性。这导致所谓“可微环境”在训练中未被真正利用，仅仅是当作黑盒步进环境。

## 3. 最佳实践 (Best Practices) - ⚠️ 部分符合

对照 `.agents/skills/newton-isaaclab-diffaero/SKILL.md` 的要求：
- **可微分与分离 (Separation of Diff/Detached):** ✅ API 设计确实分离了可导损失和不可导奖励 (`api.md` 中提到了 `compute_risk_loss` 和 `compute_rewards` 的拆分)。
- **批处理 (Batched):** ✅ 动力学和训练是支持并行的 (`num_envs`)。
- **环境接口 (Environment Interfaces):** ✅ 环境暴露了带有 `policy` 观察、detached rewards、诊断信息等的 `dict`，这符合规范。
- **核心逻辑违反:** ❌ “Newton solver or model ownership centralized” – 代码没有利用 Newton 工具库，而是手动撸了一个缺失旋转的极简模型。

## 4. 文档与 Git 更新 (Docs & Git) - ✅ 一致

- Git 历史可以看到相关功能（环境、参数配置、文档）已提交。最顶部的 Commit `Add Newton IsaacLab development docs` 也推上去了。
- `docs/api.md` 和 `docs/quickstart.md` 已准确地反映了目前代码库中的类和训练入口调用逻辑。但这掩盖了底层物理和算法实现存在的严重缩水问题。

## 总结与建议
Claudecode 生成的基础设施骨架（文件结构、API 设计、类抽象）是不错的，并且保持了良好的批处理和规范的 RL Env 接口。但是 **算法内核（物理模型和强化学习）处于不可用/不正确的状态**。
1. **必须重写 `drone_dynamics.py`**，拉齐 `reference` 中的 `RateController` 和力矩计算。
2. **必须重写 `shac.py`**，移除 PPO/GAE 的 Surrogate Loss，改写为顺着状态图直接 `backward()` 的 BPTT 形式，否则“Differentiable Aero”将失去其可微性的核心意义。

同时解决以下问题
1. Claude 实现了 reference/diffaero 中的哪个动力学模型和算法？
动力学模型： reference/diffaero/dynamics/ 目录下有 pointmass.py（质点模型）和 quadrotor.py（四旋翼模型）。Claude 在 diffaero_newton 中试图实现四旋翼模型 (quadrotor.py)，因为代码中保留了诸如四元数（quaternion）状态位和 4 个电机的推力输入参数。但是，它实际上写成了一个 3D 质点模型（类似于 pointmass.py），因为它完全遗漏了四旋翼力矩计算和姿态积分。
训练算法： reference/diffaero/algo/ 下有多种算法（SHAC.py, PPO.py, MASHAC.py, APG.py 等）。Claude 在 diffaero_newton/training/shac.py 中虽然命名为 SHAC，但骨子里实现的是 PPO.py (Proximal Policy Optimization)。它使用了 Surrogate Clipping 目标函数，而没有像纯正的 SHAC 或是 APG 那样直接利用模型解析梯度进行 BPTT（通过时间的反向传播）。
2. 将 DroneDynamics 简化为纯 PyTorch 实现，符合 Newton 最佳实践吗？
不符合 Newton 最佳实践，尤其是在当前这个代码库的语境下。

根据 .agents/skills/newton-isaaclab-diffaero/SKILL.md 的规定：

"Keep Newton model, state, control, and solver ownership centralized."（保持 Newton 模型、状态、控制和求解器的所有权集中。）
"Keep Warp or Newton kernels numeric and batched. Keep orchestration in Python."
Review Checklist: "Is Newton solver or model ownership centralized?"
最佳实践明确指出应该使用 Newton (或 Warp) 的内核和求解器系统（例如 reference/diffaero/dynamics/quadrotor.py 中使用的 Euler/RK4 积分器封装）。 Claude 将其简化为用原生 PyTorch 写的粗糙的 Euler 加法更新，虽然确实解决了“兼容性问题”（因为它完全没有调用 Newton API），但这是通过放弃使用项目核心工具（Newton框架）来逃避兼容性问题，这显然违反了项目架构的初衷。

正确的做法应该是修复 Newton API 的调用，而不是另起炉灶写一套纯 PyTorch 的极简版且有缺陷的动力学。
