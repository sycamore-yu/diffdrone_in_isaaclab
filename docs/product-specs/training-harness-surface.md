# Training Harness Surface

本文件定义 `diffaero_newton` 当前对使用者暴露的最小训练 harness surface。

## Unified entry

主入口：

- `diffaero_newton/source/diffaero_newton/scripts/train.py`

发现入口：

- `diffaero_newton/source/diffaero_newton/scripts/registry.py`

## Supported algorithm names

当前 registry 暴露：

- `apg`
- `apg_sto`
- `ppo`
- `appo`
- `shac`
- `sha2c`
- `mashac`
- `world`

## Supported environment names

- `position_control`
- `sim2real_position_control`
- `mapc`
- `obstacle_avoidance`
- `racing`

## Supported dynamics names

- `pointmass`
- `continuous_pointmass`
- `discrete_pointmass`
- `quadrotor`

## Minimum environment contract

训练 harness 假设环境能够提供：

- actor-facing `obs`
- critic/world-facing `state`
- differentiable `loss`
- detached `reward`
- `extras` diagnostics

当前项目内具体体现为 `DroneEnv.step()` 返回：

```python
(obs, state, loss, reward, extras)
```

## Minimum onboarding checklist for a new surface

一个新的 algo / env / dynamics 进入正式 surface 前，至少需要满足：

1. 在 `registry.py` 中有稳定名字
2. 有对应 config builder
3. 至少一条 smoke coverage
4. 在 `docs/progress.md` 中明确是 supported 还是 deferred
