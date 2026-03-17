# AI-First Engineering Audit 2026-03-17

This document records a behavior-first audit of `diffaero_newton` against the current target state in `docs/progress.md`.

## Scope

The audit focused on:

- capability claims in `docs/progress.md`
- registry and entrypoint reachability
- explicit environment contracts
- regression risk in recently changed runtime and training paths
- executable validation of the key smoke paths currently listed in `docs/progress.md`

## Findings Status Update

The three primary findings recorded in the initial audit have now been addressed on `main`. The sections below remain as the original behavior-oriented findings, followed by their remediation status.

### 1. Racing is overstated in the current progress document

Severity at audit time: high

`RacingEnv` is still missing the core task semantics required by the current progress claim. The environment allocates `target_gates` and `n_passed_gates`, and it defines `is_passed()`, but those signals are never wired into `step()`. `_get_rewards()` currently returns an all-zero tensor, so there is no usable gate-progress reward path at all.

Relevant code:

- [registry.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/scripts/registry.py#L183)
- [racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/racing_env.py#L62)
- [racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/racing_env.py#L101)
- [racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/racing_env.py#L121)

Remediation status:

- `RacingEnv` now wires gate pass/collision detection into `step()`.
- Gate pass advances `target_gates` and increments `n_passed_gates`.
- Racing reward/loss are no longer zeroed placeholders.
- Dedicated regression coverage now exists in [test_racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/tests/test_racing_env.py).

Follow-up issue: `diffdroneinisaac_workspace-r66` (ready to close after this remediation pass)

### 2. Obstacle avoidance and racing expose stale explicit observation contracts

Severity at audit time: medium

The current registry path builds both `obstacle_avoidance` and `racing` from `DroneEnvCfg`, which still declares the base DroneEnv observation shape. `ObstacleAvoidanceEnv` appends sensor features to the policy observation, and `RacingEnv` emits a 10-dimensional gate-frame observation, but neither path updates the explicit `observation_space` or `num_observations` contract accordingly.

This contradicts the contract discipline described in `docs/development.md` and creates wrapper-facing drift: training code that derives dimensions from the live tensor still works, but any consumer relying on the declared space can be misled.

Relevant code:

- [registry.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/scripts/registry.py#L173)
- [registry.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/scripts/registry.py#L183)
- [obstacle_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/obstacle_env.py#L35)
- [racing_env.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/envs/racing_env.py#L75)
- [drone_env_cfg.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/configs/drone_env_cfg.py#L37)

Remediation status:

- Registry now builds obstacle avoidance from `ObstacleAvoidanceEnvCfg` and racing from `RacingEnvCfg`.
- Obstacle sensor selection now explicitly updates `sensor_cfg`, `num_observations`, and `observation_space`.
- Dedicated contract coverage now exists for camera-backed obstacle observations and racing gate-frame observations.

Follow-up issue: `diffdroneinisaac_workspace-d7e` (ready to close after this remediation pass)

### 3. Package-level `__main__` still has a device contract mismatch risk

Severity at audit time: medium

`diffaero_newton.__main__` prints and stores `args.device` into `TrainingCfg`, but it constructs `DroneEnv` without passing that same device through. On CUDA-capable hosts, the environment can therefore resolve to its own default device while the trainer config claims a different one. That is a contract bug at the entrypoint layer and it is not currently covered by a dedicated smoke.

Relevant code:

- [__main__.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/__main__.py#L61)
- [__main__.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/__main__.py#L82)
- [__main__.py](/home/tong/tongworkspace/diffdroneinisaac_workspace/diffaero_newton/source/diffaero_newton/__main__.py#L86)

Remediation status:

- `diffaero_newton.__main__` now resolves one requested device path and passes it to both `DroneEnvCfg` and `TrainingCfg`.
- Runtime preflight now includes a package-entry smoke that asserts the environment and trainer receive the same resolved device.

Follow-up issue: `diffdroneinisaac_workspace-69r` (ready to close after this remediation pass)

## Validation Run

The following checks were executed during the original audit and remediation pass:

- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_dynamics.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_env.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_ppo_training.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_mashac_training.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_position_control.py -q`
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_racing_env.py -q`

Observed result:

- No regression failure was reproduced in the currently covered runtime, world, pointmass, obstacle, PPO, MASHAC, or position-control smoke paths.
- `test_pointmass_dynamics.py` and `test_obstacle_training.py` emitted Warp/Torch warnings about non-leaf `.grad` access, but those runs still passed.

## Conclusion

The originally reported primary findings were legitimate at audit time, and they have now been remediated with executable coverage. The remaining risk has shifted back toward breadth: world-model breadth, end-to-end racing training validation, and broader CI layering rather than the specific contract mismatches identified in the audit.

## Coverage Gaps Worth Tracking

These remain the credible blind spots after remediation:

- There is still no end-to-end trainer smoke proving that `racing` learns successfully through the unified entry; current coverage stops at environment semantics and gate progression.
- DreamerV3/world is still only validated on the state-only `position_control` path; perception-backed variants and broader task coverage remain uncovered.
- Pytest marker tiers exist, but there is still no dedicated CI split that enforces `cpu_smoke`, `gpu_smoke`, and `runtime_preflight` as separate gates.
