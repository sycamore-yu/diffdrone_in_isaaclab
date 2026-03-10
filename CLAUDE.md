# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This workspace reproduces DiffAero's key training capabilities on top of **Newton** (differentiable physics) and **IsaacLab** (scalable vectorized task orchestration):

- Differentiable quadrotor dynamics rollout
- Short-horizon risk loss
- Policy training loop
- Obstacle environment task interface

## Architecture

The project follows a **four-layer architecture**:

1. **Dynamics Layer** (`source/*/dynamics/`) - Newton model, control interfaces, differentiable rollout
2. **Environment Layer** (`source/*/envs/`) - IsaacLab environments, DirectRLEnv-style
3. **Training Layer** (`source/*/training/`) - SHAC-style short-horizon training loops
4. **Task Layer** (`source/*/tasks/`) - Obstacle generation, observations, risk terms

## Reference Sources

This workspace contains local reference copies in `reference/`:

- `reference/newton/` - Newton differentiable physics engine
- `reference/IsaacLab/` - IsaacLab robotics simulation framework
- `reference/diffaero/` - Original DiffAero implementation
- `reference/DiffPhysDrone/` - Differentiable physics drone reference

Key reference patterns:
- Newton drone example: `reference/newton/newton/examples/diffsim/example_diffsim_drone.py`
- IsaacLab DirectRLEnv: `reference/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py`
- DiffAero obstacle task: `reference/diffaero/env/obstacle_avoidance.py`
- DiffAero SHAC: `reference/diffaero/algo/SHAC.py`

## Core Conventions

### Configuration
- Use `@configclass` for environment and task configuration
- Use `cfg` for config objects (not `config`)
- Put dimensions, timing, noise, and rollout horizon in config, not hardcoded

### Tensor Semantics
Document all public tensors with:
- shape (e.g., `[num_envs, 3]`)
- dtype (e.g., `float32`)
- units for physical values (e.g., `meters`)
- device expectations

### Differentiable Design
- Keep gradient-carrying tensors separate from detached logging/rewards
- Do not hide `.detach()` inside utility functions
- Keep rollout math and control application independent from task bookkeeping
- Expose one-step and multi-step rollout entrypoints explicitly

### Naming
- Use `obs`, `state`, `reward`, `terminated`, `truncated`, `extras` consistently
- Use `xform` for Newton-style transforms
- Prefer `None` defaults for optional arguments

### Environment Outputs
Distinguish these channels explicitly:
- `obs` - actor-facing observation
- `state` - critic-facing global state
- `loss_terms` - differentiable outputs for training
- `reward` - detached RL signal for accounting
- `extras` - diagnostics, success flags

## Issue Tracking

This project uses **bd (beads)** for issue tracking. See `AGENTS.md` for details.

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Skills

Use the `newton-isaaclab-diffaero` skill when:
- Adding or refactoring quadrotor dynamics code
- Implementing differentiable rollout with Newton or Warp
- Building IsaacLab environments for drone tasks
- Designing short-horizon risk or collision losses

## Non-Interactive Shell Commands

Use non-interactive flags to avoid hanging on confirmation prompts:

```bash
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file
rm -rf directory            # NOT: rm -r directory
```

---

## diffaero_newton 项目

在 `diffaero_newton/` 目录下实现了一个使用 Newton 可微分物理引擎的四旋翼 RL 训练框架。

### 项目结构
```
diffaero_newton/
├── source/diffaero_newton/
│   ├── common/constants.py    # 物理常量
│   ├── configs/              # 配置类
│   ├── dynamics/             # 无人机动力学
│   ├── envs/                # IsaacLab 风格环境
│   ├── tasks/               # 障碍物任务
│   └── training/            # SHAC 训练算法
```

### 测试命令
```bash
conda activate isaaclab-newton
export CUDA_VISIBLE_DEVICES=''  # 使用 CPU
python -c "
import sys; sys.path.insert(0, 'diffaero_newton/source')
from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
import torch
env = DroneEnv(DroneEnvCfg(num_envs=4))
obs, _ = env.reset()
obs, r, t, tr, e = env.step(torch.zeros(4, 4))
print('OK')
"
```

### 已修复问题
- DroneDynamics 简化为纯 PyTorch 实现（解决 Newton API 兼容性问题）
- DroneEnvCfg 添加 `__init__` 方法
- Gymnasium spaces 使用 numpy dtype
- 设备按 `torch.cuda.is_available()` 自动选择，并在环境与动力学对象之间保持一致
