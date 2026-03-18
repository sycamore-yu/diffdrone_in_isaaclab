# Team Lifecycle v4 Report

**Session**: `tlv4-diffaero-newton-migration-plan-20260318`  
**Requirement**: Audit DiffAero -> Newton/IsaacLab migration status, initialize project standards, define delivery criteria, streamline `docs/`, and break remaining work into `bd` issues.  
**Pipeline**: `spec-only`

## Summary

- Completed capabilities on `main`: APG / APG_sto, PPO / APPO, SHAC, MASHAC, DreamerV3/world (state-only), pointmass dynamics family, position_control, sim2real_position_control, MAPC, obstacle avoidance at coarse parity, racing core on pointmass.
- Partial or narrower-than-reference areas: quadrotor controller semantics, obstacle geometry and reset/randomization depth, sensor realism and IMU support, world breadth beyond state-only `position_control`, Hydra/export/tooling parity.
- Confirmed regression: `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q` currently fails with `1 failed, 6 passed` because `dynamics/registry.py` passes unsupported `action_frame` into `DroneConfig`.

## Spec Setup

Initialized:

- `.workflow/project-tech.json`
- `.ccw/specs/coding-conventions.md`
- `.ccw/specs/architecture-constraints.md`
- `.ccw/specs/quality-rules.md`

Key standards:

- Keep the runtime split explicit between real IsaacLab launch and the Newton-only DirectRLEnv shim.
- Treat `scripts/registry.py` as the single public assembly surface.
- Preserve the 5-tuple env contract `obs, state, loss_terms, reward, extras`.
- Require runtime_preflight / cpu_smoke / gpu_smoke coverage by change type.
- Forbid duplicate public config fields and undocumented tensor semantics.

## Docs Cleanup

Active docs now live at:

- `docs/development.md`
- `docs/progress.md`
- `docs/regression_matrix.md`
- `docs/README.md`

Historical snapshots were moved under `docs/archive/`.

## Delivery Criteria

Final migration delivery should not be declared until all of the following are true:

1. Every claimed registry capability has at least one passing validation gate in `isaaclab-newton`.
2. Claimed scope in `docs/development.md`, `docs/progress.md`, and `bd` is consistent.
3. Quadrotor env-backed unified entry is repaired before body-rate parity is claimed.
4. Remaining gaps are either implemented with tests or explicitly descoped in docs and registry.

## BD Breakdown

Created under epic `diffdroneinisaac_workspace-i8x`:

- `diffdroneinisaac_workspace-i8x.17` Fix quadrotor unified-entry regression and config drift
- `diffdroneinisaac_workspace-i8x.18` Migrate SHA2C or explicitly descope it
- `diffdroneinisaac_workspace-i8x.19` Close obstacle/sensor parity gaps
- `diffdroneinisaac_workspace-i8x.20` Expand DreamerV3/world breadth or explicitly descope it
- `diffdroneinisaac_workspace-i8x.21` Establish final delivery gates and validation matrix
- `diffdroneinisaac_workspace-i8x.22` Reconcile stale epic child issue status with current code/tests

Reused existing open issues for tooling and related gaps:

- `diffdroneinisaac_workspace-i8x.16`
- `diffdroneinisaac_workspace-2kq`
- `diffdroneinisaac_workspace-3w0`
- `diffdroneinisaac_workspace-18r`
- `diffdroneinisaac_workspace-8pt`
