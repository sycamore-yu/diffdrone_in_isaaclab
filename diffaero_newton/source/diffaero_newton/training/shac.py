"""SHAC (Scalable High-actor-critic) algorithm implementation.

SHAC is a short-horizon model-free RL algorithm that uses differentiable
dynamics for efficient policy optimization.
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from diffaero_newton.configs.training_cfg import TrainingCfg
from diffaero_newton.training.buffer import RolloutBuffer
from diffaero_newton.common.constants import ACTION_DIM


class Actor(nn.Module):
    """Actor (policy) network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_dims: list = [256, 256, 128],
        log_std_init: float = -0.5,
    ):
        """Initialize actor network.

        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            hidden_dims: Hidden layer dimensions.
            log_std_init: Initial log standard deviation.
        """
        super().__init__()

        # Build network
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Observation [batch, obs_dim].

        Returns:
            Tuple of (action_mean, action_log_std).
        """
        hidden = self.network(obs)
        mean = self.mean(hidden)
        log_std = self.log_std.clamp(-20, 2)
        return mean, log_std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from observation.

        Args:
            obs: Observation [batch, obs_dim].
            deterministic: If True, return mean action.

        Returns:
            Tuple of (action, log_prob, entropy).
        """
        mean, log_std = self.forward(obs)

        if deterministic:
            return mean, torch.zeros_like(mean), torch.zeros_like(mean)

        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        # Squash action to [0, 1]
        action = torch.sigmoid(action)

        return action, log_prob, entropy


class Critic(nn.Module):
    """Critic (value function) network."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list = [256, 256, 128],
    ):
        """Initialize critic network.

        Args:
            obs_dim: Observation dimension.
            hidden_dims: Hidden layer dimensions.
        """
        super().__init__()

        # Q-function: obs + action -> Q value
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Observation [batch, obs_dim].

        Returns:
            V-value [batch, 1].
        """
        return self.network(obs)


class SHACAgent:
    """SHAC agent with actor-critic architecture."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        cfg: Optional[TrainingCfg] = None,
    ):
        """Initialize SHAC agent.

        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            cfg: Training configuration.
        """
        self.cfg = cfg or TrainingCfg()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Networks
        self.actor = Actor(obs_dim, action_dim, self.cfg.actor_hidden_dims, self.cfg.actor_log_std_init)
        self.critic = Critic(obs_dim, self.cfg.critic_hidden_dims)
        self.target_critic = Critic(obs_dim, self.cfg.critic_hidden_dims)

        # Copy weights to target
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad_(False)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.cfg.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.cfg.critic_lr,
        )

        # Device
        self.device = torch.device(self.cfg.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from observation.

        Args:
            obs: Observation.
            deterministic: If True, return mean action.

        Returns:
            Tuple of (action, log_prob, entropy).
        """
        obs = obs.to(self.device)
        return self.actor.get_action(obs, deterministic)

    def update(
        self,
        buffer: RolloutBuffer,
    ) -> Dict[str, float]:
        """Update actor and critic.

        Args:
            buffer: Rollout buffer with experience.

        Returns:
            Dictionary of training metrics.
        """
        # Compute returns and advantages from detached per-step losses.
        returns, advantages = self._compute_gae(buffer)

        # Flatten buffer data
        obs = buffer.obs[:-1].reshape(-1, self.obs_dim)
        old_values = buffer.values.reshape(-1, 1)

        # Update actor from differentiable short-horizon loss.
        actor_loss, entropy = self._update_actor(buffer)

        # Update critic on detached targets.
        critic_loss = self._update_critic(obs, returns)

        # Update target critic (soft update)
        self._soft_update_target()

        metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
            "value_mean": old_values.mean().item(),
            "advantage_mean": advantages.mean().item(),
        }

        return metrics

    def _compute_gae(
        self,
        buffer: RolloutBuffer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE (Generalized Advantage Estimation).

        Args:
            buffer: Rollout buffer.

        Returns:
            Tuple of (returns, advantages).
        """
        gamma = self.cfg.gamma
        lam = self.cfg.lam

        losses = buffer.losses  # [horizon, num_envs]
        values = buffer.values.squeeze(-1)  # [horizon, num_envs]
        next_values = buffer.next_values.squeeze(-1)  # [horizon, num_envs]
        resets = buffer.resets  # [horizon, num_envs]
        terminations = buffer.terminations  # [horizon, num_envs]

        # Compute advantages
        advantages = torch.zeros_like(losses)

        gae = torch.zeros_like(losses[0])  # [num_envs]
        for t in reversed(range(len(losses))):
            nonterminal = 1.0 - terminations[t]
            nonreset = 1.0 - resets[t]
            delta = losses[t] + gamma * next_values[t] * nonterminal - values[t]
            gae = delta + gamma * lam * nonreset * gae
            advantages[t] = gae

        # Compute returns
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns.reshape(-1, 1), advantages.reshape(-1, 1)

    def _update_critic(
        self,
        obs: torch.Tensor,
        returns: torch.Tensor,
    ) -> float:
        """Update critic network.

        Args:
            obs: Observations.
            returns: Returns.

        Returns:
            Critic loss value.
        """
        obs = obs.to(self.device)
        returns = returns.to(self.device)

        values = self.critic(obs)

        # Compute loss
        critic_loss = F.mse_loss(values, returns)

        # Optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, buffer: RolloutBuffer) -> Tuple[float, float]:
        """Update actor from the differentiable short-horizon loss graph."""
        if buffer.actor_loss_graph is None:
            raise RuntimeError("Rollout buffer does not contain an actor loss graph.")

        self.actor_optimizer.zero_grad()
        actor_loss = buffer.actor_loss_graph
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()

        return actor_loss.detach().item(), buffer.mean_entropy.detach().item()

    def _soft_update_target(self):
        """Soft update target critic network."""
        tau = 0.005
        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        """Save agent checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent checkpoint.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])


class SHAC:
    """SHAC training loop."""

    def __init__(
        self,
        env,
        cfg: Optional[TrainingCfg] = None,
    ):
        """Initialize SHAC trainer.

        Args:
            env: Environment instance.
            cfg: Training configuration.
        """
        self.env = env
        self.cfg = cfg or TrainingCfg()

        # Get obs/action dimensions
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        # Create agent
        self.agent = SHACAgent(
            obs_dim=obs_shape[0],
            action_dim=action_shape[0],
            cfg=self.cfg,
        )

        # Create buffer
        self.buffer = RolloutBuffer(
            num_envs=env.num_envs,
            horizon=self.cfg.rollout_horizon,
            obs_dim=obs_shape[0],
            action_dim=action_shape[0],
            device=self.cfg.device,
        )

        # Training state
        self.iteration = 0
        self.last_extras = {}
        self.writer = SummaryWriter(self.cfg.log_dir) if self.cfg.enable_tensorboard else None

    def train(self):
        """Run training loop."""
        obs, _ = self.env.reset()

        try:
            while self.iteration < self.cfg.num_iterations:
                # Collect rollout
                obs = self._collect_rollout(obs)

                # Update agent
                metrics = self.agent.update(self.buffer)

                # Log
                if self.iteration % self.cfg.log_interval == 0:
                    self._log_metrics(metrics)

                # Save
                if self.iteration % self.cfg.save_interval == 0:
                    self._save_checkpoint()

                if hasattr(self.env, "detach_graph"):
                    self.env.detach_graph()

                self.iteration += 1
        finally:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

    @staticmethod
    def _policy_obs(obs: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract the actor-facing observation tensor."""
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def _collect_rollout(self, obs: torch.Tensor | Dict[str, torch.Tensor]):
        """Collect rollout experience natively with BPTT graphs attached.

        Args:
            obs: Initial observation.
        """
        self.buffer.reset()
        
        # Accumulate horizon loss while preserving graph across the rollout.
        cumulated_loss = torch.zeros(self.env.num_envs, device=self.agent.device)
        entropy_accum = torch.zeros(self.env.num_envs, device=self.agent.device)
        
        for step in range(self.cfg.rollout_horizon):
            policy_obs = self._policy_obs(obs)

            # Get action (retains graph to actor params)
            action, log_prob, entropy = self.agent.get_action(policy_obs)

            next_obs, (loss, reward), terminated, truncated, extras = self.env.step(action)
            reset = extras["reset"]
            next_obs_before_reset = extras["next_obs_before_reset"]
            self.last_extras = extras
            
            with torch.no_grad():
                value = self.agent.critic(policy_obs.detach().to(self.agent.device))

            next_value_graph = self.agent.target_critic(next_obs_before_reset.to(self.agent.device))
            next_value = next_value_graph.detach()
                
            # Horizon accumulation
            gamma = self.cfg.gamma
            cumulated_loss = cumulated_loss + loss.to(self.agent.device) * (gamma ** step)
            entropy_accum = entropy_accum + entropy.squeeze(-1)

            is_rollout_end = step == self.cfg.rollout_horizon - 1
            if is_rollout_end:
                should_bootstrap = ~terminated
            else:
                should_bootstrap = truncated & ~terminated
            terminal_bootstrap = (gamma ** (step + 1)) * next_value_graph.squeeze(-1)
            cumulated_loss = cumulated_loss + terminal_bootstrap * should_bootstrap.float().to(self.agent.device)

            next_policy_obs = self._policy_obs(next_obs).detach()

            # Store in buffer
            self.buffer.add(
                obs=policy_obs.detach(), 
                next_obs=next_policy_obs, 
                action=action.detach(), 
                loss=loss.detach(),
                reward=reward.detach(), 
                done=reset.detach(), 
                terminated=terminated.detach(),
                reset=reset.detach(),
                log_prob=log_prob.detach(), 
                value=value.detach(),
                next_value=next_value.detach()
            )
            obs = next_obs

        # Finalize actor graph
        self.buffer.actor_loss_graph = cumulated_loss.mean() - self.cfg.entropy_coef * entropy_accum.mean()
        self.buffer.mean_entropy = entropy_accum.mean() / self.cfg.rollout_horizon

        final_policy_obs = self._policy_obs(obs).detach()
        self.buffer.bootstrap(self.agent.target_critic(final_policy_obs.to(self.agent.device)).detach())
        return {"policy": final_policy_obs} if isinstance(obs, dict) else final_policy_obs

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics.

        Args:
            metrics: Dictionary of metrics.
        """
        print(f"Iteration {self.iteration}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, self.iteration)
            episode = self.last_extras.get("episode", {})
            obstacles = self.last_extras.get("obstacles", {})
            if "r" in episode:
                self.writer.add_scalar("env/episode_reward", episode["r"], self.iteration)
            if "l" in episode:
                self.writer.add_scalar("env/episode_length", episode["l"], self.iteration)
            if "nearest_dist" in obstacles:
                self.writer.add_scalar("env/nearest_obstacle_dist", obstacles["nearest_dist"], self.iteration)
            if "collisions" in obstacles:
                self.writer.add_scalar("env/collisions", obstacles["collisions"], self.iteration)
            self.writer.flush()

    def _save_checkpoint(self):
        """Save training checkpoint."""
        import os
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        path = os.path.join(self.cfg.save_dir, f"checkpoint_{self.iteration}.pt")
        self.agent.save(path)
        print(f"Saved checkpoint to {path}")
