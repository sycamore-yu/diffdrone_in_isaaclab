# `diffaero_newton/` Phase 3 Prep For Gemini

Timestamp: `2026-03-10 18:03:33 +0800`

This document isolates the `Phase 3` work that should be handled by Gemini:

- align training semantics with `reference/diffaero/algo/SHAC.py`
- introduce differentiable short-horizon loss accumulation
- stop calling the current PPO-style loop “SHAC” without qualification

## Current Reality

The current training stack is still PPO-style:

- actor update uses clipped surrogate loss
- critic consumes `(obs, action)` rather than a cleaner value-target contract
- environment returns detached reward bookkeeping, not differentiable training loss
- `compute_risk_loss()` exists but is not on the actual actor-update path

Relevant files:

- `diffaero_newton/source/diffaero_newton/training/shac.py`
- `diffaero_newton/source/diffaero_newton/training/buffer.py`
- `diffaero_newton/source/diffaero_newton/envs/drone_env.py`
- `diffaero_newton/source/diffaero_newton/tasks/reward_terms.py`
- `reference/diffaero/algo/SHAC.py`

## Reference Target

Gemini should align with the reference design, not just with the current code.

Core reference semantics from `reference/diffaero/algo/SHAC.py`:

- actor loss is accumulated from differentiable horizon loss, not PPO clipping
- `record_loss()` consumes environment-provided `loss`, `reset`, `truncated`, and `next_obs_before_reset`
- terminal bootstrap uses environment-provided next observations before auto-reset
- reward accounting and differentiable actor loss are separate concerns
- rollout bookkeeping tracks termination and reset semantics explicitly

## Initial Analysis Already Done

These facts should reduce Gemini’s discovery time:

1. The environment now returns policy observation as `obs["policy"]`, and that path is stable enough to build on.
2. The trainer/buffer layer already has some rollout storage improvements from the previous debug pass:
   - next observation storage exists
   - bootstrap value storage exists
3. The current environment auto-resets internally inside `step()`.

That third point is the main blocker for reference-style SHAC semantics.

Reference SHAC needs access to “next observation before reset” so actor loss and terminal bootstrap are computed against the correct state, not against a post-reset state.

## Required Design Changes

### 1. Define a real env-to-trainer contract

The trainer should not guess semantics from the current Gym-style tuple alone.

Recommended target:

```python
next_obs, reward, terminated, truncated, extras = env.step(action)
```

But `extras` must include training-side fields such as:

- `loss_terms`
- `reset`
- `truncated`
- `terminated`
- `next_obs_before_reset`

Alternative acceptable contract:

```python
next_obs, (loss, reward), terminated, env_info = env.step(...)
```

This is closer to `reference/diffaero`.

The important point is semantic fidelity, not exact tuple shape.

### 2. Separate reward from differentiable actor loss

Recommended split:

- `reward`: detached tensor for logging / RL accounting
- `loss_terms`: differentiable tensor or dict used by actor optimization

`compute_risk_loss()` is the natural starting point, but it needs to be wired into the real rollout path.

### 3. Replace PPO clipped actor update

Current PPO-style path in `training/shac.py` should be replaced or isolated behind a different name.

The target actor update should look conceptually like:

1. collect action and policy info
2. step env and receive differentiable `loss`
3. accumulate discounted horizon loss
4. bootstrap with `next_obs_before_reset` when truncated / rollout-end
5. backprop actor through accumulated differentiable objective

### 4. Revisit critic contract

The current critic is a Q-style `(obs, action) -> value` network.

Gemini should decide explicitly whether to:

- keep that interface and justify it against the reference, or
- move to a cleaner value-target design closer to the reference stack

This decision must be documented in the final PR summary.

## Suggested File-Level Plan

### `diffaero_newton/source/diffaero_newton/envs/drone_env.py`

Add:

- differentiable `loss_terms` output
- `next_obs_before_reset`
- explicit `reset`, `terminated`, `truncated` flags in `extras`

Potential implementation note:

- capture pre-reset observation first
- compute task loss from pre-reset state
- only then reset terminated/truncated environments

### `diffaero_newton/source/diffaero_newton/tasks/reward_terms.py`

Refactor into two clearly separate paths:

- detached reward terms
- differentiable loss terms

Avoid mixing logging-only reward with actor-optimization loss.

### `diffaero_newton/source/diffaero_newton/training/buffer.py`

Ensure the buffer can store:

- differentiable losses
- done/reset masks
- terminated masks
- truncated masks
- next observations before reset
- next values for bootstrap

### `diffaero_newton/source/diffaero_newton/training/shac.py`

Main tasks:

- add `record_loss()` style accumulation
- replace PPO clipped actor update
- implement horizon accumulation and bootstrap behavior
- keep reward logging detached from actor loss

## Recommended Migration Strategy

Do this in two small steps rather than one rewrite:

1. Extend env and buffer contracts while keeping the current actor update temporarily runnable.
2. Swap the actor update from PPO-style clipping to reference-style differentiable loss accumulation.

This reduces the chance of breaking the training CLI and makes debugging easier.

## Acceptance Criteria

Gemini’s Phase 3 is done only if all of the following are true:

- actor update depends on differentiable horizon loss accumulation
- actor update does not use PPO clipped surrogate loss
- env exposes pre-reset next observations or equivalent bootstrap-safe data
- trainer can explain how its loss path maps to `reference/diffaero/algo/SHAC.py`
- tests cover truncated and terminated bootstrap edge cases

## Suggested Tests

Run existing smoke tests plus add targeted tests for:

- `next_obs_before_reset` correctness
- actor loss accumulation across horizon
- truncated bootstrap behavior
- terminated path with no invalid post-reset bootstrap leakage
- reward remains detached while loss stays differentiable

## Handoff Note For Gemini

Do not spend time re-litigating whether the current code is PPO-style. It is.

The correct next step is to redesign the training contract around differentiable loss accumulation and bootstrap-safe environment outputs.
