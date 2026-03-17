"""Unified environment, dynamics, and algorithm registry for training entry."""

from __future__ import annotations

from typing import Any

import torch

from diffaero_newton.common.constants import ACTION_DIM


def build_algo(name: str, obs_dim: int, action_dim: int = ACTION_DIM, device: str = "cuda", **kwargs):
    """Build a training algorithm by name."""
    name = name.lower()

    if name == "apg":
        from diffaero_newton.training.apg import APG

        return APG(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)

    if name in ("apg_sto", "apg_stochastic"):
        from diffaero_newton.training.apg import APGStochastic

        return APGStochastic(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)

    if name == "ppo":
        from diffaero_newton.training.ppo import PPO

        return PPO(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)

    if name in ("appo", "asymmetric_ppo"):
        from diffaero_newton.training.ppo import AsymmetricPPO

        state_dim = kwargs.pop("state_dim", obs_dim)
        return AsymmetricPPO(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs,
        )

    if name == "shac":
        from diffaero_newton.training.shac import SHACAgent
        from diffaero_newton.configs.training_cfg import TrainingCfg

        cfg = kwargs.pop("cfg", None) or TrainingCfg(device=device)
        return SHACAgent(obs_dim=obs_dim, action_dim=action_dim, cfg=cfg)

    if name in ("world", "dreamerv3"):
        from diffaero_newton.training.dreamerv3 import World_Agent

        env = kwargs.pop("env", None)
        if env is None:
            raise ValueError("DreamerV3/world requires an environment instance.")
        cfg = kwargs.pop("cfg", None) or {}
        world_device = env.device if hasattr(env, "device") else device
        return World_Agent(cfg=cfg, env=env, device=torch.device(world_device))

    if name == "mashac":
        from diffaero_newton.training.mashac import MASHACAgent
        from diffaero_newton.configs.training_cfg import TrainingCfg

        state_dim = kwargs.pop("state_dim", obs_dim)
        cfg = kwargs.pop("cfg", None) or TrainingCfg(device=device)
        return MASHACAgent(obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim, cfg=cfg)

    raise ValueError(f"Unknown algorithm: {name}. Available: {', '.join(sorted(ALGO_REGISTRY))}")


ALGO_REGISTRY = {
    "apg": "diffaero_newton.training.apg.APG",
    "apg_sto": "diffaero_newton.training.apg.APGStochastic",
    "ppo": "diffaero_newton.training.ppo.PPO",
    "appo": "diffaero_newton.training.ppo.AsymmetricPPO",
    "shac": "diffaero_newton.training.shac.SHAC",
    "world": "diffaero_newton.training.dreamerv3.World_Agent",
    "mashac": "diffaero_newton.training.mashac.MASHAC",
}

ENV_REGISTRY = {
    "position_control": "diffaero_newton.envs.position_control_env.PositionControlEnv",
    "sim2real_position_control": "diffaero_newton.envs.position_control_env.Sim2RealPositionControlEnv",
    "mapc": "diffaero_newton.envs.mapc_env.MAPCEnv",
    "obstacle_avoidance": "diffaero_newton.envs.obstacle_env.ObstacleAvoidanceEnv",
    "racing": "diffaero_newton.envs.racing_env.RacingEnv",
}

DYNAMICS_REGISTRY = {
    "pointmass": "diffaero_newton.configs.dynamics_cfg.PointMassCfg",
    "continuous_pointmass": "diffaero_newton.configs.dynamics_cfg.ContinuousPointMassCfg",
    "discrete_pointmass": "diffaero_newton.configs.dynamics_cfg.DiscretePointMassCfg",
    "quadrotor": "diffaero_newton.configs.dynamics_cfg.QuadrotorCfg",
}


def build_dynamics_cfg(name: str, num_envs: int, requires_grad: bool, dt: float | None = None):
    """Build a dynamics config object by name."""
    name = name.lower()
    if name == "pointmass":
        from diffaero_newton.configs.dynamics_cfg import PointMassCfg

        cfg = PointMassCfg(num_envs=num_envs, requires_grad=requires_grad)
    elif name == "continuous_pointmass":
        from diffaero_newton.configs.dynamics_cfg import ContinuousPointMassCfg

        cfg = ContinuousPointMassCfg(num_envs=num_envs, requires_grad=requires_grad)
    elif name == "discrete_pointmass":
        from diffaero_newton.configs.dynamics_cfg import DiscretePointMassCfg

        cfg = DiscretePointMassCfg(num_envs=num_envs, requires_grad=requires_grad)
    elif name == "quadrotor":
        from diffaero_newton.configs.dynamics_cfg import QuadrotorCfg

        cfg = QuadrotorCfg(num_envs=num_envs, requires_grad=requires_grad)
    else:
        raise ValueError(
            f"Unknown dynamics model: {name}. Available: {', '.join(sorted(DYNAMICS_REGISTRY))}"
        )

    if dt is not None:
        cfg.dt = dt
    return cfg


def build_env(
    name: str,
    dynamics: str,
    num_envs: int,
    device: str,
    differentiable: bool,
    sensor: str = "relpos",
):
    """Build an environment configured for the selected dynamics backend."""
    name = name.lower()
    dynamics = dynamics.lower()

    if name == "position_control":
        from diffaero_newton.configs.position_control_env_cfg import PositionControlEnvCfg
        from diffaero_newton.envs.position_control_env import PositionControlEnv

        cfg = PositionControlEnvCfg()
        cfg.num_envs = num_envs
        cfg.scene.num_envs = num_envs
        cfg.dynamics = build_dynamics_cfg(dynamics, num_envs=num_envs, requires_grad=differentiable, dt=cfg.sim.dt)
        return PositionControlEnv(cfg=cfg, device=device)

    if name == "sim2real_position_control":
        from diffaero_newton.configs.position_control_env_cfg import Sim2RealPositionControlEnvCfg
        from diffaero_newton.envs.position_control_env import Sim2RealPositionControlEnv

        cfg = Sim2RealPositionControlEnvCfg()
        cfg.num_envs = num_envs
        cfg.scene.num_envs = num_envs
        cfg.dynamics = build_dynamics_cfg(dynamics, num_envs=num_envs, requires_grad=differentiable, dt=cfg.sim.dt)
        return Sim2RealPositionControlEnv(cfg=cfg, device=device)

    if name == "mapc":
        from diffaero_newton.configs.mapc_env_cfg import MAPCEnvCfg
        from diffaero_newton.envs.mapc_env import MAPCEnv

        cfg = MAPCEnvCfg()
        cfg.num_envs = num_envs
        cfg.scene.num_envs = num_envs
        cfg.__post_init__()
        cfg.dynamics = build_dynamics_cfg(
            dynamics,
            num_envs=num_envs * cfg.n_agents,
            requires_grad=differentiable,
            dt=cfg.sim.dt,
        )
        return MAPCEnv(cfg=cfg, device=device)

    if name == "obstacle_avoidance":
        from diffaero_newton.configs.obstacle_env_cfg import ObstacleAvoidanceEnvCfg
        from diffaero_newton.configs.sensor_cfg import build_sensor_cfg
        from diffaero_newton.envs.obstacle_env import ObstacleAvoidanceEnv

        cfg = ObstacleAvoidanceEnvCfg()
        cfg.num_envs = num_envs
        cfg.scene.num_envs = num_envs
        cfg.sensor_cfg = build_sensor_cfg(sensor, cfg.num_obstacles)
        cfg.__post_init__()
        cfg.dynamics = build_dynamics_cfg(dynamics, num_envs=num_envs, requires_grad=differentiable, dt=cfg.sim.dt)
        return ObstacleAvoidanceEnv(cfg=cfg, device=device)

    if name == "racing":
        from diffaero_newton.configs.racing_env_cfg import RacingEnvCfg
        from diffaero_newton.envs.racing_env import RacingEnv

        cfg = RacingEnvCfg()
        cfg.num_envs = num_envs
        cfg.scene.num_envs = num_envs
        cfg.dynamics = build_dynamics_cfg(dynamics, num_envs=num_envs, requires_grad=differentiable, dt=cfg.sim.dt)
        return RacingEnv(cfg=cfg, device=device)

    raise ValueError(f"Unknown environment: {name}. Available: {', '.join(sorted(ENV_REGISTRY))}")


def get_policy_obs(obs: Any):
    """Extract the actor-facing observation tensor from env output."""
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def get_env_state(env) -> Any:
    """Return privileged state flattened per environment when available."""
    if not hasattr(env, "drone"):
        return None

    state = env.drone.get_flat_state()
    if hasattr(env, "n_agents"):
        return state.view(env.num_envs, -1)
    return state.view(env.num_envs, -1)


def list_available():
    """Print available algorithms, environments, and dynamics."""
    print("Algorithms:", ", ".join(sorted(ALGO_REGISTRY)))
    print("Environments:", ", ".join(sorted(ENV_REGISTRY)))
    print("Dynamics:", ", ".join(sorted(DYNAMICS_REGISTRY)))
