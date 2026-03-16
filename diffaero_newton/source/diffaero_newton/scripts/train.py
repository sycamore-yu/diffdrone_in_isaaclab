"""Unified training entry point for diffaero_newton.

Usage:
    python scripts/train.py --algo apg --env position_control --dynamics pointmass --max_iter 100

This script:
1. Selects algorithm, environment, and dynamics by name
2. Sets up the training loop with logging
3. Runs training for the specified number of iterations
"""

import argparse
import sys
import os
import time

# Ensure package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from diffaero_newton.common.isaaclab_compat import launch_app
app = launch_app()

import torch
import torch.nn as nn

from diffaero_newton.scripts.registry import build_algo, list_available
from diffaero_newton.common.constants import ACTION_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training entry for diffaero_newton")
    parser.add_argument("--algo", type=str, default="apg",
                        help="Algorithm: apg, apg_sto, ppo, appo, shac")
    parser.add_argument("--env", type=str, default="position_control",
                        help="Environment: position_control, mapc, obstacle_avoidance, racing")
    parser.add_argument("--dynamics", type=str, default="pointmass",
                        help="Dynamics model: pointmass, quadrotor")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum training iterations")
    parser.add_argument("--l_rollout", type=int, default=16,
                        help="Rollout length per iteration")
    parser.add_argument("--n_envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N iterations")
    parser.add_argument("--list", action="store_true",
                        help="List available algorithms and environments")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_available()
        return

    # Seed
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Training: algo={args.algo}, env={args.env}, dynamics={args.dynamics}")
    print(f"  device={device}, n_envs={args.n_envs}, l_rollout={args.l_rollout}, lr={args.lr}")

    # For now, we create a simple synthetic training loop
    # since full env instantiation requires Omniverse scene setup.
    obs_dim = 18  # typical for position control
    algo_kwargs = {"lr": args.lr, "l_rollout": args.l_rollout}

    if args.algo in ("ppo", "appo"):
        algo_kwargs["n_envs"] = args.n_envs
        if args.algo == "appo":
            algo_kwargs["state_dim"] = 30

    agent = build_algo(args.algo, obs_dim=obs_dim, action_dim=ACTION_DIM,
                       device=device, **algo_kwargs)

    print(f"  Algorithm: {type(agent).__name__}")
    # Parameter counting: APG exposes .actor, PPO/APPO expose .agent, SHAC may vary
    _net = getattr(agent, "actor", None) or getattr(agent, "agent", None)
    if _net is not None:
        print(f"  Parameters: {sum(p.numel() for p in _net.parameters()):,}")
    print(f"\nStarting training for {args.max_iter} iterations...")

    start_time = time.time()

    for iteration in range(1, args.max_iter + 1):
        # Simulate observations (placeholder until full env integration)
        obs = torch.randn(args.n_envs, obs_dim, device=device, requires_grad=True)

        if args.algo in ("apg", "apg_sto", "shac"):
            # Differentiable rollout
            for _ in range(args.l_rollout):
                if args.algo == "apg":
                    action = agent.act(obs)
                    loss = ((action - 0.5) ** 2).sum(dim=-1)
                    agent.record_loss(loss)
                elif args.algo in ("apg_sto",):
                    action, log_prob, entropy = agent.act(obs)
                    loss = ((action - 0.5) ** 2).sum(dim=-1)
                    agent.record_loss(loss, entropy)
                obs = torch.randn(args.n_envs, obs_dim, device=device, requires_grad=True)

            metrics = agent.update_actor()

        elif args.algo in ("ppo", "appo"):
            agent.buffer.clear()
            obs_ppo = torch.randn(args.n_envs, obs_dim, device=device)
            for step in range(args.l_rollout):
                with torch.no_grad():
                    action, info = agent.act(obs_ppo)
                    reward = -((action - 0.5) ** 2).sum(-1)
                    next_done = torch.zeros(args.n_envs, device=device)
                    if args.algo == "appo":
                        state = torch.randn(args.n_envs, 30, device=device)
                        next_value = agent.agent.get_value(state)
                        agent.buffer.add(obs_ppo, state, info["sample"],
                                         info["logprob"], reward, next_done,
                                         next_value, next_value)
                    else:
                        next_value = agent.agent.get_value(obs_ppo)
                        agent.buffer.add(obs_ppo, info["sample"], info["logprob"],
                                         reward, next_done, info["value"], next_value)
                    obs_ppo = torch.randn(args.n_envs, obs_dim, device=device)

            advantages, target_values = agent.bootstrap()
            for _ in range(agent.n_epoch):
                losses, grad_norms = agent.train_epoch(advantages, target_values)
            metrics = {**losses, **grad_norms}

        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - start_time
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  [{iteration}/{args.max_iter}] {elapsed:.1f}s | {metrics_str}")

    # Save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(args.save_dir)
    print(f"\nTraining complete. Checkpoint saved to {args.save_dir}/")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
