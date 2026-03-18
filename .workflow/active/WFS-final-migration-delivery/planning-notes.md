# Planning Notes

## User Intent
GOAL: Finish the remaining DiffAero -> Newton/IsaacLab migration work from a fresh worktree and land it as one PR.

SCOPE:
- Use `.wt/final-migration-delivery` as the only implementation branch.
- Plan first via `workflow-plan` style artifacts.
- Execute in acceptance-driven loops: code -> `isaaclab-newton` test -> `ai-first-engineering` review -> docs update -> commit.
- Submit one PR only after delivery gates are satisfied.

CONTEXT:
- Project standards live in `.workflow/` and `.ccw/specs/`.
- `docs/development.md` is the target architecture.
- `docs/progress.md` is current implementation truth but still contains stale post-PR11 contradictions on merged `main`.
- Root reference repos are populated at:
  - `/home/tong/tongworkspace/diffdroneinisaac_workspace/reference/diffaero`
  - `/home/tong/tongworkspace/diffdroneinisaac_workspace/reference/newton`
  - `/home/tong/tongworkspace/diffdroneinisaac_workspace/reference/IsaacLab`
- The new worktree contains empty `reference/` placeholders, so all reference comparisons must use the absolute root paths above.

## Context Findings
- Branch baseline in this worktree is `ebb408d` (`origin/main` / merged PR #11).
- Verified locally before planning:
  - `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q` -> `7 passed`
  - `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics quadrotor --quadrotor-control-mode body_rate ... --device cpu` -> passes
- Remaining migration gaps confirmed by docs + code search:
  - `SHA2C` is still absent from `scripts/registry.py`.
  - Quadrotor still has partial semantic parity even though the unified-entry regression is fixed.
  - Obstacle/sensor parity is still narrower than reference semantics.
  - DreamerV3/world has perception scaffolding in the code but mainline scope and validation remain state-only.
  - Tooling parity remains missing: Hydra wrapping, export, WandB, Optuna, deploy.
- Conflict risk is high because remaining work spans registry, training, dynamics, envs, docs, and validation gates.

## AI-First Delivery Rules
- No task is complete without executable evidence.
- Each IMPL task must define:
  - explicit acceptance criteria
  - concrete test commands in `isaaclab-newton`
  - review checks focused on regression, contract breakage, and rollout safety
  - required `docs/` updates
- Review bandwidth goes to correctness, behavior regressions, data/physics integrity, and rollout safety, not style cleanup.

## Working Assumptions
- Use `scripts/registry.py` and `scripts/train.py` as the only public assembly points.
- Keep runtime split explicit: `common/isaaclab_launch.py` vs `common/direct_rl_shim.py`.
- Use Context7 only when Newton / IsaacLab runtime behavior needs authoritative reference clarification.
