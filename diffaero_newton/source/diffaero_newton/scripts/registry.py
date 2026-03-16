"""Unified algorithm and environment registry.

Provides factory functions to build environments and training algorithms
by name, enabling the unified training entry point.
"""

from typing import Optional
import torch

from diffaero_newton.common.constants import ACTION_DIM


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def build_algo(name: str, obs_dim: int, action_dim: int = ACTION_DIM,
               device: str = "cuda", **kwargs):
    """Build a training algorithm by name.

    Args:
        name: Algorithm name. One of: shac, apg, apg_sto, ppo, appo.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        device: Torch device string.
        **kwargs: Additional algorithm-specific arguments.

    Returns:
        Algorithm instance.
    """
    name = name.lower()

    if name == "apg":
        from diffaero_newton.training.apg import APG
        return APG(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)

    elif name in ("apg_sto", "apg_stochastic"):
        from diffaero_newton.training.apg import APGStochastic
        return APGStochastic(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)

    elif name == "ppo":
        from diffaero_newton.training.ppo import PPO
        return PPO(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)

    elif name in ("appo", "asymmetric_ppo"):
        from diffaero_newton.training.ppo import AsymmetricPPO
        state_dim = kwargs.pop("state_dim", obs_dim)
        return AsymmetricPPO(obs_dim=obs_dim, state_dim=state_dim,
                             action_dim=action_dim, device=device, **kwargs)

    elif name == "shac":
        # SHAC is more complex (needs critic + buffer), return class for manual init
        from diffaero_newton.training.shac import SHACAgent
        return SHACAgent(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)

    else:
        raise ValueError(f"Unknown algorithm: {name}. "
                         f"Available: apg, apg_sto, ppo, appo, shac")


# ---------------------------------------------------------------------------
# Environment registry (names → config classes)
# ---------------------------------------------------------------------------

ALGO_REGISTRY = {
    "apg": "diffaero_newton.training.apg.APG",
    "apg_sto": "diffaero_newton.training.apg.APGStochastic",
    "ppo": "diffaero_newton.training.ppo.PPO",
    "appo": "diffaero_newton.training.ppo.AsymmetricPPO",
    "shac": "diffaero_newton.training.shac.SHACAgent",
}

ENV_REGISTRY = {
    "position_control": "diffaero_newton.envs.drone_env.DroneEnv",
    "mapc": "diffaero_newton.envs.drone_env.DroneEnv",
    "obstacle_avoidance": "diffaero_newton.envs.obstacle_env.ObstacleAvoidanceEnv",
    "racing": "diffaero_newton.envs.racing_env.RacingEnv",
}

DYNAMICS_REGISTRY = {
    "pointmass": "diffaero_newton.dynamics.pointmass.PointMass",
    "quadrotor": "diffaero_newton.dynamics.quadrotor.Quadrotor",
}


def list_available():
    """Print available algorithms, environments, and dynamics."""
    print("Algorithms:", list(ALGO_REGISTRY.keys()))
    print("Environments:", list(ENV_REGISTRY.keys()))
    print("Dynamics:", list(DYNAMICS_REGISTRY.keys()))
