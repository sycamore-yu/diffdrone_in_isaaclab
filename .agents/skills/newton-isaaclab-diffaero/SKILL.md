---
name: newton-isaaclab-diffaero
description: Use when designing, implementing, or reviewing Newton + IsaacLab code in this workspace, especially for differentiable quadrotor rollout, short-horizon risk loss, policy training loops, and obstacle-task environments.
---

# Newton + IsaacLab DiffAero Conventions

Read `docs/development.md` first when the task touches project architecture. Use this skill for concrete implementation and review decisions.

## Use This Skill When

- adding or refactoring quadrotor dynamics code
- implementing differentiable rollout with Newton or Warp
- building IsaacLab environments for drone tasks
- designing short-horizon risk or collision losses
- wiring a training loop to environment outputs
- reviewing whether a new module matches project conventions

## Core Project Rules

- Prefer IsaacLab `DirectRLEnv` style for custom drone tasks in this repo.
- Keep Newton model, state, control, and solver ownership centralized.
- Keep differentiable losses separate from detached RL reward bookkeeping.
- Keep obstacle geometry queries reusable outside sensor code.
- Keep Warp or Newton kernels numeric and batched. Keep orchestration in Python.

## API And Naming Conventions

- Use `cfg` for configuration objects, not `config`.
- Use `xform` for Newton-style transforms when a transform argument is part of a public API.
- Prefer `None` defaults for optional config or transform inputs.
- Document public tensor outputs with shape, dtype, and units.
- Keep environment-facing names stable: `obs`, `state`, `reward`, `terminated`, `truncated`, `extras`.
- Avoid signature churn in rollout helpers. Add explicit parameters instead of ambiguous convenience wrappers.

## Environment Design Template

Default structure:

1. Config declares timing, spaces, rollout horizon, and task parameters.
2. Environment owns reset, step cadence, truncation, and diagnostics.
3. Dynamics module owns control application and state rollout.
4. Task module owns observations, risk terms, success criteria, and obstacle sampling.
5. Training loop owns horizon accumulation, bootstrap, and optimizer steps.

Choose a manager-based IsaacLab environment only after the rollout and task contracts are stable and clearly reusable.

## Differentiable Rollout Rules

- Expose one-step and multi-step rollout entrypoints.
- Do not hide `.detach()` inside rollout helpers.
- Document tensor shapes, dtypes, and units on public outputs.
- Batch over environments by default. Do not introduce per-env Python loops in hot paths.
- Separate numeric kernels from reset and curriculum logic.

## Risk-Loss Rules

- Use geometry-aware signals, not collision booleans alone.
- Keep individual loss terms named and inspectable.
- Separate differentiable `loss_terms` from detached `reward`.
- Make horizon weighting and terminal handling explicit in the training code.
- Sensor modules may provide inputs to the loss, but they should not own the final risk definition.

## Training-Loop Contract

- Environment step should make it obvious which outputs are differentiable and which are detached.
- Training code owns rollout buffers, bootstrap logic, and gradient truncation boundaries.
- Keep `terminated` and `truncated` distinct internally.
- Reset horizon accumulators exactly at reset boundaries.
- Log optimization diagnostics and task metrics separately.

## Newton And IsaacLab Best Practices

From Newton:

- Follow project vocabulary such as `cfg` and `xform` where applicable.
- Keep default configuration values explicit and simple.
- Prefer `None` defaults over constructed objects for optional inputs.
- Keep lifecycle ownership in one place instead of distributing builder or solver setup across modules.

From IsaacLab:

- Use `@configclass` configs.
- Declare `action_space`, `observation_space`, and `state_space` explicitly.
- Use replicated environments for homogeneous training when possible.
- Do not force manager abstractions before task semantics are proven stable.

## Review Checklist

When reviewing a change, verify:

- Does the environment expose `obs`, optional `state`, detached diagnostics, and differentiable loss inputs clearly?
- Is Newton solver or model ownership centralized?
- Are obstacle-distance queries reusable by both observations and risk loss?
- Does reset logic avoid leaking previous rollout state?
- Are tensor semantics documented?
- Are kernels free of task orchestration or logging logic?

## Reject These Patterns

- A single unnamed scalar mixing risk loss, RL reward, and logging value
- Sensor code that is the only place obstacle geometry can be accessed
- Hidden detach points in rollout helpers
- Environment code that mixes reset bookkeeping, rollout math, and optimizer logic in one function
- Early manager-based abstraction that obscures gradient flow or terminal handling
