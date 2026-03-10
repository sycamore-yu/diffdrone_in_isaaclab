# Newton + IsaacLab DiffAero Reproduction Guide

This workspace exists to reproduce a narrow set of DiffAero capabilities on top of Newton and IsaacLab:

- Differentiable quadrotor dynamics rollout
- Short-horizon risk loss
- Policy training loop
- Obstacle environment task interface

The goal is not to clone DiffAero or IsaacLab wholesale. The goal is to build a small, explicit stack that uses Newton for differentiable physics and IsaacLab for scalable vectorized task orchestration.

## Design Sources

This document is grounded in local reference copies of:

- `reference/newton`
- `reference/IsaacLab`
- `reference/diffaero`
- `reference/DiffPhysDrone`

The most important source patterns are:

- Newton differentiable drone example: `reference/newton/newton/examples/diffsim/example_diffsim_drone.py`
- IsaacLab direct RL environment base: `reference/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py`
- IsaacLab Newton lifecycle bridge: `reference/IsaacLab/source/isaaclab/isaaclab/sim/_impl/newton_manager.py`
- IsaacLab warp direct-task examples: `reference/IsaacLab/source/isaaclab_tasks_experimental/isaaclab_tasks_experimental/direct`
- DiffAero obstacle task and SHAC loop:
  - `reference/diffaero/env/obstacle_avoidance.py`
  - `reference/diffaero/algo/SHAC.py`

## Target Architecture

The project should stay split into four layers.

### 1. Dynamics Layer

Responsibility:

- Own the quadrotor state, control application, and Newton-based integration
- Expose one-step and multi-step rollout functions
- Preserve differentiability across the rollout horizon
- Provide geometry-aware queries needed by risk loss

Rules:

- Keep Newton or Warp kernels close to the dynamics code
- Keep rollout math and control application independent from IsaacLab-specific task bookkeeping
- Treat dynamics as a pure transition interface: current state + control + context -> next state and rollout traces

Expected outputs:

- State tensors needed by the task layer
- Optional cached rollout traces for loss computation
- Collision-distance or nearest-obstacle query hooks when feasible

### 2. Environment Layer

Responsibility:

- Batch environment instances
- Own reset, step cadence, truncation, and episode boundaries
- Convert between IsaacLab environment contracts and differentiable training needs

Recommended base:

- Start with a `DirectRLEnv`-style environment, not a fully manager-based RL task

Why:

- The project needs custom differentiable rollout logic and short-horizon loss accumulation
- `DirectRLEnv` keeps `step()`, reset policy, and tensor ownership explicit
- Manager-based terms are useful later, but they add indirection before the rollout contract is stable

IsaacLab patterns to keep:

- Use `@configclass` configs for environment, scene, solver, and task parameters
- Use `InteractiveSceneCfg` and replicated environments for homogeneous training
- Keep action, observation, and state space declarations explicit in the config

### 3. Training Layer

Responsibility:

- Own rollout horizon, bootstrap logic, optimization step, and logging
- Separate differentiable loss from detached RL-facing reward bookkeeping
- Support short-horizon actor updates without hiding stateful assumptions in the environment

Design rule:

- The training loop owns horizon-level accumulation
- The environment owns per-step physics advance and episode/reset semantics

This split is important because DiffAero-style training uses horizon losses that must remain gradient-carrying, while Gym-like environment returns often need detached tensors for logging, wrappers, or third-party RL code.

### 4. Task Layer

Responsibility:

- Define target sampling, obstacle sampling, observations, success conditions, and risk terms
- Translate geometry into task signals without leaking task policy into the dynamics layer

Rules:

- Obstacle generation and nearest-distance queries belong here or in a task support module
- Sensor encoders may feed observations, but they should not own the final risk or collision penalty definition
- Task logic should work against batched state tensors, not per-env Python loops

## Recommended Repository Layout

Use a small source tree that makes ownership obvious:

```text
source/<package_name>/
  common/
  configs/
  dynamics/
  envs/
  tasks/
  training/
  utils/
```

Suggested responsibilities:

- `common/`: shared tensor types, constants, units, shape helpers
- `configs/`: `@configclass` and training/task config definitions
- `dynamics/`: Newton model setup, control interfaces, differentiable rollout
- `envs/`: IsaacLab environment wrappers and task entrypoints
- `tasks/`: obstacle generation, observation builders, loss/reward terms
- `training/`: SHAC-style or related short-horizon training loops, buffers, bootstrap helpers
- `utils/`: logging, visualization hooks, debug helpers only when they do not hide core behavior

Avoid a layout where tasks, dynamics, and learning code are mixed in one environment file. That is fast for prototyping and expensive to maintain.

## Core Interfaces To Keep Stable

Lock these contracts early.

### Environment Config

Every task config should declare at least:

- `sim.dt`
- `decimation`
- `episode_length_s`
- `num_envs`
- `rollout_horizon`
- `action_space`
- `observation_space`
- `state_space` when asymmetric critic training is used
- obstacle generation parameters
- reset and curriculum parameters

### Environment Outputs

Distinguish these channels explicitly:

- `obs`: actor-facing observation
- `state`: critic-facing global state when needed
- `loss_terms`: differentiable outputs used by the training loop
- `reward`: detached scalar signal for RL accounting
- `extras`: detached diagnostics, success flags, distances, logging payloads

Do not overload a single `reward` tensor to serve both RL accounting and differentiable short-horizon optimization.

### Dynamics Contract

Keep separate entrypoints for:

- applying controls
- integrating one step
- rolling out multiple steps
- querying obstacle or collision context used by risk loss

This prevents the task layer from depending on hidden side effects inside rollout internals.

## Coding Standards

These rules are specific to this workspace.

### Config-First Design

- Use IsaacLab `@configclass` for environment and task configuration
- Put dimensions, timing, noise, and rollout horizon in config, not hardcoded in kernels
- Keep config names stable and literal

### Explicit Tensor Semantics

Every public tensor attribute or function result should be documented with:

- shape
- dtype
- units when physical values are involved
- device expectations when non-obvious

Examples:

- position: `[num_envs, 3]`, `float32`, meters
- body pose: `[num_envs, ...]`, Newton/Warp transform dtype
- per-step risk term: `[num_envs]`, differentiable scalar

### Differentiability Discipline

- Keep gradient-carrying tensors separate from logging or reset masks that do not need gradients
- Do not detach intermediate rollout tensors unless there is a measured reason
- Do not hide detach calls inside utility functions
- Make truncated-horizon bootstrap behavior explicit in the training loop

### Kernel Boundaries

- Warp or Newton kernels should do numeric work only
- Python orchestration code should own reset masks, curriculum, task mode switches, and logging
- Avoid embedding task semantics deeply inside kernels unless the operation must be fused for performance

### Environment Semantics

- Reset behavior must be batch-safe and deterministic for a given seed
- Terminated and truncated must remain distinct in environment internals
- Horizon accumulation must not leak state across reset boundaries

### Naming

- Prefer IsaacLab and Newton vocabulary where possible
- Use `cfg` for config objects
- Use `xform` when following Newton transform naming
- Use `obs`, `state`, `reward`, `terminated`, `truncated`, `extras` consistently across environments
- Do not introduce synonyms like `transform`, `pose_cfg`, or `config` when a Newton- or IsaacLab-aligned term already exists

### Public API Style

- Use `None` defaults for optional configuration or transform arguments instead of constructing mutable defaults inline
- When exposing tensor-valued public attributes, document concrete shape, dtype, and unit semantics in the docstring
- Keep rollout helper signatures literal and stable; avoid convenience overloads that hide device, horizon, or reset semantics
- If an enum or mode switch is part of public configuration, keep the values simple and append-only where possible

## Integration Best Practices

### Newton Integration

- Centralize builder, model, state, control, and solver lifecycle ownership
- Avoid scattering Newton model construction across task code
- Treat CUDA graphing and substep configuration as solver-level concerns, not task-level concerns
- If the project introduces its own Newton wrapper, it should mirror the single-owner lifecycle used by IsaacLab's `NewtonManager`

### IsaacLab Integration

- Use `InteractiveSceneCfg(... replicate_physics=True, clone_in_fabric=True)` for homogeneous batched training unless task heterogeneity makes that impossible
- Keep environment code close to `DirectRLEnv` patterns until the drone task contracts stabilize
- Only move task logic into manager terms when the observation, reward, and reset contracts are already proven stable

### Task And Risk Design

- Keep nearest-distance queries available as first-class geometry signals
- Build risk terms from geometry and motion, not only binary collision events
- Keep reward shaping and differentiable risk loss separate even when they share ingredients
- Prefer short-horizon losses that are decomposed into named terms for debugging and ablation

### Training Loop Design

- Keep the rollout buffer schema aligned with environment outputs
- Make final-step and reset-step handling explicit
- Record the exact point where gradients stop
- Log both task success metrics and optimization-specific diagnostics

## Failure Modes To Avoid

- Mixing IsaacLab reset logic with differentiable horizon accumulation in one opaque helper
- Using manager-based reward terms too early and then trying to recover gradient-carrying tensors after detaches
- Burying obstacle geometry access inside a sensor module that the loss function cannot reuse
- Letting per-env Python loops creep into rollout, reset, or risk evaluation
- Combining detached RL rewards and differentiable losses into a single unnamed scalar
- Spreading Newton solver ownership across multiple modules

## Implementation Order

Implement in this order:

1. Minimal differentiable quadrotor rollout in free space
2. Short-horizon risk loss over obstacle geometry
3. Training loop with explicit horizon accumulation and bootstrap handling
4. Obstacle task interface with reset, observations, and metrics
5. Only after the above is stable: refactor reusable observation or reward pieces into manager-like terms if needed

## Review Checklist

Before merging any feature in this workspace, verify:

- The new code preserves differentiability where intended
- Observation, state, and action spaces are declared in config
- Tensor outputs have clear shape and unit semantics
- Reset logic does not leak horizon state
- Obstacle risk can be inspected independently from the reward scalar
- Newton lifecycle ownership remains centralized
- Warp kernels contain math, not orchestration

## Related Skill

For future Codex sessions, also consult:

- `.agents/skills/newton-isaaclab-diffaero/SKILL.md`

That skill is the concise operational version of this document.
