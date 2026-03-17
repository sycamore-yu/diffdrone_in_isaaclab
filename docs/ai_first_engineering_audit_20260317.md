# AI-First Engineering Audit 2026-03-17

This document records a behavior-first audit of `diffaero_newton` against the current target state in `docs/progress.md`.

## Scope

The audit focused on:

- capability claims in `docs/progress.md`
- registry and entrypoint reachability
- explicit environment contracts
- regression risk in recently changed runtime and training paths
- executable validation of the key smoke paths currently listed in `docs/progress.md`

## Findings

### 1. Racing is overstated in the current progress document

Severity: high

`RacingEnv` is still missing the core task semantics required by the current progress claim. The environment allocates `target_gates` and `n_passed_gates`, and it defines `is_passed()`, but those signals are never wired into `step()`. `_get_rewards()` currently returns an all-zero tensor, so there is no usable gate-progress reward path at all.

Relevant code:

- [registry.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/scripts/registry.py#L183)
- [racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/racing_env.py#L62)
- [racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/racing_env.py#L101)
- [racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/racing_env.py#L121)

Follow-up issue: `diffdroneinisaac_workspace-r66`

### 2. Obstacle avoidance and racing expose stale explicit observation contracts

Severity: medium

The current registry path builds both `obstacle_avoidance` and `racing` from `DroneEnvCfg`, which still declares the base DroneEnv observation shape. `ObstacleAvoidanceEnv` appends sensor features to the policy observation, and `RacingEnv` emits a 10-dimensional gate-frame observation, but neither path updates the explicit `observation_space` or `num_observations` contract accordingly.

This contradicts the contract discipline described in `docs/development.md` and creates wrapper-facing drift: training code that derives dimensions from the live tensor still works, but any consumer relying on the declared space can be misled.

Relevant code:

- [registry.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/scripts/registry.py#L173)
- [registry.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/scripts/registry.py#L183)
- [obstacle_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/obstacle_env.py#L35)
- [racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/racing_env.py#L75)
- [drone_env_cfg.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/configs/drone_env_cfg.py#L37)

Follow-up issue: `diffdroneinisaac_workspace-d7e`

### 3. Package-level `__main__` still has a device contract mismatch risk

Severity: medium

`diffaero_newton.__main__` prints and stores `args.device` into `TrainingCfg`, but it constructs `DroneEnv` without passing that same device through. On CUDA-capable hosts, the environment can therefore resolve to its own default device while the trainer config claims a different one. That is a contract bug at the entrypoint layer and it is not currently covered by a dedicated smoke.

Relevant code:

- [__main__.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/__main__.py#L61)
- [__main__.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/__main__.py#L82)
- [__main__.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/__main__.py#L86)

Follow-up issue: `diffdroneinisaac_workspace-69r`

## Validation Run

The following checks were executed during this audit:

- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_dynamics.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_env.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_ppo_training.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_mashac_training.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_position_control.py -q`

Observed result:

- No regression failure was reproduced in the currently covered runtime, world, pointmass, obstacle, PPO, MASHAC, or position-control smoke paths.
- `test_pointmass_dynamics.py` and `test_obstacle_training.py` emitted Warp/Torch warnings about non-leaf `.grad` access, but those runs still passed.

## Conclusion

The currently validated mainline paths do not show an immediate executable regression in the smoke coverage that exists today. The main risk is not a failing tested path, but a mismatch between documented capability claims and the behavior actually wired for untested or weakly tested surfaces, especially `racing` and wrapper-facing observation contracts.

## Coverage Gaps Worth Tracking

These were not promoted to primary findings because no deterministic failure was reproduced, but they are credible regression blind spots:

- The recent obstacle autograd reset fix in [drone_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/drone_env.py#L146) is not directly covered by a reset-triggering differentiable regression test. Current differentiable obstacle checks intentionally avoid resets.
- Runtime preflight currently covers [train.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/scripts/train.py) and launcher importability, but not the package entry [__main__.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/__main__.py).
- The `world` smoke in [test_world_training.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/tests/test_world_training.py) validates a single `agent.step()` path, not replay readiness or actual world-model updates.
- [test_mashac_training.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/tests/test_mashac_training.py) is not assigned to the new `cpu_smoke` or `gpu_smoke` marker tiers, so marker-based smoke runs can skip it silently.
