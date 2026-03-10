"""Main entry point for DiffAero Newton training."""

import argparse
import torch

from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.configs.training_cfg import TrainingCfg
from diffaero_newton.training.shac import SHAC


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DiffAero Newton Training")

    # Environment
    parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
    parser.add_argument("--episode_length_s", type=float, default=30.0, help="Episode length in seconds")

    # Training
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--rollout_horizon", type=int, default=10, help="Rollout horizon for SHAC")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Critic learning rate")

    # Logging
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint save interval")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    print("=" * 50)
    print("DiffAero Newton Training")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Num envs: {args.num_envs}")
    print(f"Iterations: {args.num_iterations}")
    print()

    # Create environment config
    env_cfg = DroneEnvCfg()
    env_cfg.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length_s

    # Create training config
    training_cfg = TrainingCfg()
    training_cfg.num_iterations = args.num_iterations
    training_cfg.rollout_horizon = args.rollout_horizon
    training_cfg.actor_lr = args.actor_lr
    training_cfg.critic_lr = args.critic_lr
    training_cfg.log_interval = args.log_interval
    training_cfg.save_interval = args.save_interval
    training_cfg.save_dir = args.save_dir
    training_cfg.device = args.device

    # Create environment
    print("Creating environment...")
    env = DroneEnv(cfg=env_cfg)
    print(f"Environment created: {env.num_envs} parallel environments")
    print()

    # Create SHAC trainer
    print("Initializing SHAC agent...")
    trainer = SHAC(env, cfg=training_cfg)
    print("SHAC agent initialized")
    print()

    # Run training
    print("Starting training...")
    print("=" * 50)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    print("Training complete!")


if __name__ == "__main__":
    main()
