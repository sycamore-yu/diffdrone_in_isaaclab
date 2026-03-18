# IMPL-2 Review

Review framework: `ai-first-engineering`

## Scope Reviewed

- `training/shac.py`
- `training/__init__.py`
- `scripts/registry.py`
- `scripts/train.py`
- `tests/test_shac_training.py`
- `tests/test_train_entry.py`
- `docs/progress.md`
- `docs/regression_matrix.md`

## Executable Evidence

- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_shac_training.py -q`
  - Result: pass (`3 passed`)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q`
  - Result: pass (`7 passed`)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo sha2c --env position_control --dynamics pointmass --max_iter 1 --l_rollout 2 --n_envs 2 --device cpu --log_interval 1`
  - Result: pass

## Findings

- The implementation takes the lowest-risk path: `SHA2C` is added as a state-aware SHAC variant instead of introducing a second divergent short-horizon training stack.
- Registry and CLI wiring are now explicit, so `sha2c` is visible through `--list` and available through the public train entry.
- Coverage is real but narrow: current evidence is for the privileged-state `position_control` point-mass path only.

## Residual Risks

- No GPU-backed `SHA2C` smoke exists yet.
- No task coverage beyond `position_control` is claimed, and docs must keep that limit explicit.
- The asymmetric critic path depends on env state shape remaining stable; future env contract changes must keep `get_flat_state()` semantics aligned.

## Verdict

`IMPL-2` can be treated as complete for the current support envelope.
