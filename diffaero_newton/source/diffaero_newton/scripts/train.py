"""Unified env-backed training entry point for diffaero_newton."""

from __future__ import annotations

import argparse
import os
import sys
import time
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
    parser.add_argument("--algo", type=str, default="apg", help="Algorithm: apg, apg_sto, ppo, appo, shac, world")
    parser.add_argument(
        "--env",
        type=str,
        default="position_control",
        help="Environment: position_control, mapc, obstacle_avoidance, racing",
    )
    parser.add_argument("--dynamics", type=str, default="pointmass", help="Dynamics model: pointmass, quadrotor")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum training iterations")
    parser.add_argument("--l_rollout", type=int, default=16, help="Rollout length per iteration")
    parser.add_argument("--n_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--world_warmup_steps", type=int, default=32, help="Replay warmup transitions for world")
    parser.add_argument("--world_min_ready_steps", type=int, default=8, help="Minimum replay steps before world updates")
    parser.add_argument("--world_batch_size", type=int, default=8, help="World-model batch size")
    parser.add_argument("--world_batch_length", type=int, default=8, help="World-model sequence length")
    parser.add_argument("--world_imagine_length", type=int, default=8, help="Dreamer imagination horizon")
    parser.add_argument("--world_update_freq", type=int, default=1, help="World-model updates per env step once ready")
    parser.add_argument("--list", action="store_true", help="List available algorithms and environments")
    return parser.parse_args()


def _metrics_to_str(metrics: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _reset_env(env):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


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


def _build_world_cfg(args, action_dim: int, device: str) -> dict[str, Any]:
    rollout = max(args.l_rollout, 1)
    total_transitions = max(args.n_envs * rollout * max(args.max_iter, 1), args.n_envs * 64)
    batch_length = max(2, min(args.world_batch_length, rollout))
    batch_size = max(1, min(args.world_batch_size, args.n_envs))
    return {
        "state_predictor": {
            "action_dim": action_dim,
            "only_state": True,
            "enable_rec": False,
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

    initial_state = get_env_state(env)
    if initial_state is None:
        raise RuntimeError(f"Environment '{args.env}' does not expose state required for DreamerV3/world.")

    agent = build_algo(
        "world",
        obs_dim=initial_state.shape[-1],
        action_dim=action_dim,
        device=device,
        env=env,
        cfg=_build_world_cfg(args, action_dim=action_dim, device=device),
    )

    state = initial_state.to(agent.device)
    print(f"  Algorithm: {type(agent).__name__}")
    print(f"  Parameters: {sum(param.numel() for param in agent.state_model.parameters()):,} (world model)")
    print(f"\nStarting training for {args.max_iter} iterations...")

    start_time = time.time()
    for iteration in range(1, args.max_iter + 1):
        metrics: dict[str, float] = {}
        last_reward = 0.0
        last_done = 0.0
        for _ in range(args.l_rollout):
            state, policy_info, _, reward_mean, done_mean = agent.step(state)
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


def main():
    args = parse_args()
    if args.list:
        list_available()
        return

    from diffaero_newton.common.isaaclab_compat import launch_app

    app = launch_app()
    env = None
    try:
        torch.manual_seed(args.seed)
        requested_device = args.device if torch.cuda.is_available() else "cpu"
        differentiable = args.algo in ("apg", "apg_sto", "shac")

        print(f"Training: algo={args.algo}, env={args.env}, dynamics={args.dynamics}")
        print(f"  requested_device={requested_device}, n_envs={args.n_envs}, l_rollout={args.l_rollout}, lr={args.lr}")

        env = build_env(
            name=args.env,
            dynamics=args.dynamics,
            num_envs=args.n_envs,
            device=requested_device,
            differentiable=differentiable,
        )
        device = str(env.device)
        if device != requested_device:
            print(f"  using_env_device={device} (requested {requested_device})")

        obs = _reset_env(env)
        policy_obs = get_policy_obs(obs)
        obs_dim = policy_obs.shape[-1]
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else ACTION_DIM
        state = get_env_state(env)

        if args.algo == "shac":
            _run_shac(args, env, device)
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
