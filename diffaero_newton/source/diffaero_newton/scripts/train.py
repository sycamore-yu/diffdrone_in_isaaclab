"""Unified env-backed training entry point for diffaero_newton."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from diffaero_newton.common.constants import ACTION_DIM
from diffaero_newton.configs.training_cfg import TrainingCfg
from diffaero_newton.scripts.registry import (
    build_algo,
    build_env,
    get_env_state,
    get_policy_obs,
    list_available,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training entry for diffaero_newton")
    parser.add_argument("--algo", type=str, default="apg", help="Algorithm: apg, apg_sto, ppo, appo, shac, sha2c, mashac, world")
    parser.add_argument(
        "--env",
        type=str,
        default="position_control",
        help="Environment: position_control, sim2real_position_control, mapc, obstacle_avoidance, racing",
    )
    parser.add_argument(
        "--dynamics",
        type=str,
        default="pointmass",
        help="Dynamics model: pointmass, continuous_pointmass, discrete_pointmass, quadrotor",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="relpos",
        help="Obstacle sensor: relpos, camera, lidar",
    )
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum training iterations")
    parser.add_argument("--l_rollout", type=int, default=16, help="Rollout length per iteration")
    parser.add_argument("--n_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument(
        "--quadrotor-control-mode",
        type=str,
        choices=("motor_thrust", "body_rate"),
        default=None,
        help="Optional quadrotor control mode override for unified env entry.",
    )
    parser.add_argument(
        "--quadrotor-drag-coeff-xy",
        type=float,
        default=None,
        help="Optional quadrotor xy linear drag override.",
    )
    parser.add_argument(
        "--quadrotor-drag-coeff-z",
        type=float,
        default=None,
        help="Optional quadrotor z linear drag override.",
    )
    parser.add_argument(
        "--quadrotor-k-angvel",
        type=float,
        nargs=3,
        default=None,
        metavar=("KX", "KY", "KZ"),
        help="Optional quadrotor body-rate feedback gains override.",
    )
    parser.add_argument(
        "--quadrotor-max-body-rates",
        type=float,
        nargs=3,
        default=None,
        metavar=("WX", "WY", "WZ"),
        help="Optional quadrotor body-rate command limits in rad/s.",
    )
    parser.add_argument(
        "--quadrotor-thrust-ratio",
        type=float,
        default=None,
        help="Optional DiffAero-style collective thrust scaling.",
    )
    parser.add_argument(
        "--quadrotor-torque-ratio",
        type=float,
        default=None,
        help="Optional DiffAero-style body torque scaling.",
    )
    parser.add_argument("--world_warmup_steps", type=int, default=32, help="Replay warmup transitions for world")
    parser.add_argument("--world_min_ready_steps", type=int, default=8, help="Minimum replay steps before world updates")
    parser.add_argument("--world_batch_size", type=int, default=8, help="World-model batch size")
    parser.add_argument("--world_batch_length", type=int, default=8, help="World-model sequence length")
    parser.add_argument("--world_imagine_length", type=int, default=8, help="Dreamer imagination horizon")
    parser.add_argument("--world_update_freq", type=int, default=1, help="World-model updates per env step once ready")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved run config as JSON.")
    parser.add_argument("--config-out", type=str, default=None, help="Optional path to write the resolved run config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the environment and config, then exit before training.")
    parser.add_argument("--list", action="store_true", help="List available algorithms and environments")
    return parser.parse_args()


def _metrics_to_str(metrics: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _reset_env(env):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _jsonify_config(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonify_config(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonify_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if type(value).__module__.startswith("numpy"):
        return value.tolist() if hasattr(value, "tolist") else str(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _jsonify_config(vars(value))
    if hasattr(value, "shape") and hasattr(value, "low") and hasattr(value, "high"):
        low = value.low.tolist() if hasattr(value.low, "tolist") else value.low
        high = value.high.tolist() if hasattr(value.high, "tolist") else value.high
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "low": low,
            "high": high,
        }
    return value


def _build_dynamics_overrides(args) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.dynamics != "quadrotor":
        return overrides

    if args.quadrotor_control_mode is not None:
        overrides["control_mode"] = args.quadrotor_control_mode
    if args.quadrotor_drag_coeff_xy is not None:
        overrides["drag_coeff_xy"] = args.quadrotor_drag_coeff_xy
    if args.quadrotor_drag_coeff_z is not None:
        overrides["drag_coeff_z"] = args.quadrotor_drag_coeff_z
    if args.quadrotor_k_angvel is not None:
        overrides["k_angvel"] = tuple(args.quadrotor_k_angvel)
    if args.quadrotor_max_body_rates is not None:
        overrides["max_body_rates"] = tuple(args.quadrotor_max_body_rates)
    if args.quadrotor_thrust_ratio is not None:
        overrides["thrust_ratio"] = args.quadrotor_thrust_ratio
    if args.quadrotor_torque_ratio is not None:
        overrides["torque_ratio"] = args.quadrotor_torque_ratio
    return overrides


def _build_run_config(args, env, device: str, action_dim: int, initial_obs: Any) -> dict[str, Any]:
    run_config = {
        "args": vars(args).copy(),
        "resolved_device": device,
        "action_dim": action_dim,
        "dynamics_overrides": _build_dynamics_overrides(args),
        "env_cfg": _jsonify_config(getattr(env, "cfg", None)),
    }
    if args.algo in ("world", "dreamerv3"):
        run_config["world_cfg"] = _build_world_cfg(args, action_dim=action_dim, device=device, initial_obs=initial_obs)
    return run_config


def _write_run_config(run_config: dict[str, Any], args) -> str:
    out_path = args.config_out or os.path.join(args.save_dir, "run_config.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonify_config(run_config), handle, indent=2, sort_keys=True)
    return out_path


def _run_apg_iteration(agent, env, obs, l_rollout: int):
    for _ in range(l_rollout):
        policy_obs = get_policy_obs(obs).to(agent.device)
        if hasattr(agent, "entropy_weight"):
            action, _, entropy = agent.act(policy_obs)
            next_obs, _, loss_terms, _, _ = env.step(action)
            agent.record_loss(loss_terms, entropy)
        else:
            action = agent.act(policy_obs)
            next_obs, _, loss_terms, _, _ = env.step(action)
            agent.record_loss(loss_terms)
        obs = next_obs

    metrics = agent.update_actor()
    if hasattr(env, "detach_graph"):
        env.detach_graph()
        obs = env._get_observations()
    return obs, metrics


def _run_ppo_iteration(agent, env, obs, l_rollout: int, algo_name: str):
    agent.buffer.clear()

    for _ in range(l_rollout):
        policy_obs = get_policy_obs(obs).to(agent.device)
        curr_state = get_env_state(env)
        with torch.no_grad():
            action, info = agent.act(policy_obs)
            next_obs, next_state, _, reward, extras = env.step(action)
            next_done = (extras["terminated"] | extras["truncated"]).float().to(agent.device)

            if algo_name == "appo":
                state = curr_state.to(agent.device)
                next_state = next_state.to(agent.device).view(env.num_envs, -1)
                value = agent.agent.get_value(state)
                next_value = agent.agent.get_value(next_state)
                agent.buffer.add(
                    policy_obs,
                    state,
                    info["sample"],
                    info["logprob"],
                    reward.to(agent.device),
                    next_done,
                    value,
                    next_value,
                )
            else:
                agent.buffer.add(
                    policy_obs,
                    info["sample"],
                    info["logprob"],
                    reward.to(agent.device),
                    next_done,
                    info["value"],
                    agent.agent.get_value(get_policy_obs(next_obs).to(agent.device)),
                )
        obs = next_obs

    advantages, target_values = agent.bootstrap()
    losses = {}
    grad_norms = {}
    for _ in range(agent.n_epoch):
        losses, grad_norms = agent.train_epoch(advantages, target_values)
    return obs, {**losses, **grad_norms}


def _run_shac(args, env, device: str):
    from diffaero_newton.training.shac import SHAC

    cfg = TrainingCfg(
        rollout_horizon=args.l_rollout,
        num_iterations=args.max_iter,
        actor_lr=args.lr,
        device=device,
        log_interval=args.log_interval,
        save_interval=max(args.max_iter + 1, 1000),
        save_dir=args.save_dir,
        enable_tensorboard=False,
    )
    trainer = SHAC(env, cfg=cfg)
    trainer.train()
    os.makedirs(args.save_dir, exist_ok=True)
    trainer.agent.save(os.path.join(args.save_dir, "shac_agent.pt"))


def _run_sha2c(args, env, device: str):
    from diffaero_newton.training.shac import SHA2C

    cfg = TrainingCfg(
        rollout_horizon=args.l_rollout,
        num_iterations=args.max_iter,
        actor_lr=args.lr,
        device=device,
        log_interval=args.log_interval,
        save_interval=max(args.max_iter + 1, 1000),
        save_dir=args.save_dir,
        enable_tensorboard=False,
    )
    trainer = SHA2C(env, cfg=cfg)
    trainer.train()
    os.makedirs(args.save_dir, exist_ok=True)
    trainer.agent.save(os.path.join(args.save_dir, "sha2c_agent.pt"))


def _build_world_cfg(args, action_dim: int, device: str, initial_obs: Any | None = None) -> dict[str, Any]:
    rollout = max(args.l_rollout, 1)
    total_transitions = max(args.n_envs * rollout * max(args.max_iter, 1), args.n_envs * 64)
    batch_length = max(2, min(args.world_batch_length, rollout))
    batch_size = max(1, min(args.world_batch_size, args.n_envs))
    use_perception = isinstance(initial_obs, dict) and initial_obs.get("perception") is not None
    image_height = 9
    image_width = 16
    if use_perception:
        image_height = int(initial_obs["perception"].shape[-2])
        image_width = int(initial_obs["perception"].shape[-1])
    return {
        "state_predictor": {
            "action_dim": action_dim,
            "only_state": not use_perception,
            "use_perception": use_perception,
            "enable_rec": False,
            "image_height": image_height,
            "image_width": image_width,
            "use_amp": str(device).startswith("cuda"),
            "worldmodel_update_freq": max(1, args.world_update_freq),
        },
        "replaybuffer": {
            "max_length": max(total_transitions * 2, args.n_envs * (batch_length + 4)),
            "warmup_length": max(0, args.world_warmup_steps),
            "min_ready_steps": max(1, args.world_min_ready_steps),
            "store_on_gpu": str(device).startswith("cuda"),
        },
        "world_state_env": {
            "batch_size": batch_size,
            "batch_length": batch_length,
            "imagine_length": max(2, args.world_imagine_length),
        },
    }


def _run_world(args, env, device: str, action_dim: int):
    if hasattr(env, "detach_graph"):
        env.detach_graph()

    initial_obs = _reset_env(env)
    initial_state = get_env_state(env)
    if initial_state is None:
        raise RuntimeError(f"Environment '{args.env}' does not expose state required for DreamerV3/world.")
    uses_perception = isinstance(initial_obs, dict) and initial_obs.get("perception") is not None

    agent = build_algo(
        "world",
        obs_dim=initial_state.shape[-1],
        action_dim=action_dim,
        device=device,
        env=env,
        cfg=_build_world_cfg(args, action_dim=action_dim, device=device, initial_obs=initial_obs),
    )

    world_input: Any = initial_obs if uses_perception else initial_state.to(agent.device)
    print(f"  Algorithm: {type(agent).__name__}")
    print(f"  Parameters: {sum(param.numel() for param in agent.state_model.parameters()):,} (world model)")
    print(f"\nStarting training for {args.max_iter} iterations...")

    start_time = time.time()
    for iteration in range(1, args.max_iter + 1):
        metrics: dict[str, float] = {}
        last_reward = 0.0
        last_done = 0.0
        for _ in range(args.l_rollout):
            next_state, policy_info, _, reward_mean, done_mean = agent.step(world_input)
            world_input = env._get_observations() if uses_perception else next_state
            last_reward = reward_mean
            last_done = done_mean
            metrics.update({key: float(value) for key, value in policy_info.items()})
        metrics.setdefault("reward_mean", last_reward)
        metrics.setdefault("done_mean", last_done)

        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - start_time
            print(f"  [{iteration}/{args.max_iter}] {elapsed:.1f}s | {_metrics_to_str(metrics)}")

    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(args.save_dir)
    print(f"\nTraining complete. Checkpoint saved to {args.save_dir}/")
    print(f"Total time: {time.time() - start_time:.1f}s")

def _run_mashac(args, env, device: str):
    from diffaero_newton.training.mashac import MASHAC

    cfg = TrainingCfg(
        rollout_horizon=args.l_rollout,
        num_iterations=args.max_iter,
        actor_lr=args.lr,
        device=device,
        log_interval=args.log_interval,
        save_interval=max(args.max_iter + 1, 1000),
        save_dir=args.save_dir,
        enable_tensorboard=False,
    )
    trainer = MASHAC(env, cfg=cfg)
    trainer.train()
    os.makedirs(args.save_dir, exist_ok=True)
    trainer.agent.save(os.path.join(args.save_dir, "mashac_agent.pt"))


def main():
    args = parse_args()
    if args.list:
        list_available()
        return

    from diffaero_newton.common.isaaclab_launch import launch_app

    app = launch_app()
    env = None
    try:
        torch.manual_seed(args.seed)
        requested_device = args.device if torch.cuda.is_available() else "cpu"
        differentiable = args.algo in ("apg", "apg_sto", "shac", "sha2c", "mashac")

        print(f"Training: algo={args.algo}, env={args.env}, dynamics={args.dynamics}")
        print(f"  requested_device={requested_device}, n_envs={args.n_envs}, l_rollout={args.l_rollout}, lr={args.lr}")

        env = build_env(
            name=args.env,
            dynamics=args.dynamics,
            num_envs=args.n_envs,
            device=requested_device,
            differentiable=differentiable,
            sensor=args.sensor,
            dynamics_overrides=_build_dynamics_overrides(args),
        )
        device = str(env.device)
        if device != requested_device:
            print(f"  using_env_device={device} (requested {requested_device})")

        obs = _reset_env(env)
        policy_obs = get_policy_obs(obs)
        obs_dim = policy_obs.shape[-1]
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else ACTION_DIM
        state = get_env_state(env)
        run_config = _build_run_config(args, env, device, action_dim, obs)
        config_path = _write_run_config(run_config, args)
        if args.print_config:
            print(json.dumps(_jsonify_config(run_config), indent=2, sort_keys=True))
        if args.dry_run:
            print(f"Resolved run config written to {config_path}")
            return

        if args.algo == "shac":
            _run_shac(args, env, device)
            print(f"Training complete. Checkpoint saved to {args.save_dir}/")
            return
        if args.algo == "sha2c":
            _run_sha2c(args, env, device)
            print(f"Training complete. Checkpoint saved to {args.save_dir}/")
            return
        if args.algo == "mashac":
            _run_mashac(args, env, device)
            print(f"Training complete. Checkpoint saved to {args.save_dir}/")
            return
        if args.algo in ("world", "dreamerv3"):
            _run_world(args, env, device, action_dim)
            return

        algo_kwargs: dict[str, Any] = {"lr": args.lr, "l_rollout": args.l_rollout}
        if args.algo in ("ppo", "appo"):
            algo_kwargs["n_envs"] = args.n_envs
            if args.algo == "appo":
                if state is None:
                    raise RuntimeError(f"Environment '{args.env}' does not expose privileged state for APPO.")
                algo_kwargs["state_dim"] = state.shape[-1]
        elif args.algo == "sha2c":
            if state is None:
                raise RuntimeError(f"Environment '{args.env}' does not expose privileged state for SHA2C.")
            algo_kwargs["state_dim"] = state.shape[-1]

        agent = build_algo(
            args.algo,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            **algo_kwargs,
        )

        print(f"  Algorithm: {type(agent).__name__}")
        network = getattr(agent, "actor", None) or getattr(agent, "agent", None)
        if network is not None:
            print(f"  Parameters: {sum(param.numel() for param in network.parameters()):,}")
        print(f"\nStarting training for {args.max_iter} iterations...")

        start_time = time.time()
        for iteration in range(1, args.max_iter + 1):
            if args.algo in ("apg", "apg_sto"):
                obs, metrics = _run_apg_iteration(agent, env, obs, args.l_rollout)
            else:
                obs, metrics = _run_ppo_iteration(agent, env, obs, args.l_rollout, args.algo)

            if iteration % args.log_interval == 0 or iteration == 1:
                elapsed = time.time() - start_time
                print(f"  [{iteration}/{args.max_iter}] {elapsed:.1f}s | {_metrics_to_str(metrics)}")

        os.makedirs(args.save_dir, exist_ok=True)
        agent.save(args.save_dir)
        print(f"\nTraining complete. Checkpoint saved to {args.save_dir}/")
        print(f"Total time: {time.time() - start_time:.1f}s")
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
