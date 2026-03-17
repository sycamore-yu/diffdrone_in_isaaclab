"""Legacy import bridge.

Internal code should prefer importing from ``direct_rl_shim`` or ``isaaclab_launch``
directly. This module no longer performs runtime fallback.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "diffaero_newton.common.isaaclab_compat is deprecated; import from "
    "direct_rl_shim or isaaclab_launch directly.",
    DeprecationWarning,
    stacklevel=2,
)

from diffaero_newton.common.direct_rl_shim import (
    DirectRLEnv,
    DirectRLEnvCfg,
    FeatherstoneSolverCfg,
    InteractiveSceneCfg,
    NewtonCfg,
    SimulationCfg,
    configclass,
)
from diffaero_newton.common.isaaclab_launch import launch_app
