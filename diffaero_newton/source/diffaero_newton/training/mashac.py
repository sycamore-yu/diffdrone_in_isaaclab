"""Multi-agent SHAC with shared actor and centralized critic."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from diffaero_newton.common.constants import ACTION_DIM
from diffaero_newton.configs.training_cfg import TrainingCfg
from diffaero_newton.training.buffer import StateRolloutBuffer
from diffaero_newton.training.shac import Actor


class CentralizedCritic(nn.Module):
    """Critic that consumes flattened global state for each environment."""

    def __init__(self, state_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class MASHACAgent:
    """Shared actor with centralized critic for multi-agent differentiable rollouts."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int = ACTION_DIM,
        cfg: Optional[TrainingCfg] = None,
    ):
        self.cfg = cfg or TrainingCfg()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(obs_dim, action_dim, self.cfg.actor_hidden_dims, self.cfg.actor_log_std_init)
        self.critic = CentralizedCritic(state_dim, self.cfg.critic_hidden_dims)
        self.target_critic = CentralizedCritic(state_dim, self.cfg.critic_hidden_dims)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self.device = torch.device(self.cfg.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor.get_action(obs.to(self.device), deterministic)

    def update(self, buffer: StateRolloutBuffer) -> Dict[str, float]:
        returns, advantages = self._compute_gae(buffer)
        states = buffer.states.reshape(-1, self.state_dim)
        old_values = buffer.values.reshape(-1, 1)

        actor_loss, entropy = self._update_actor(buffer)
        critic_loss = self._update_critic(states, returns)
        self._soft_update_target()

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
            "value_mean": old_values.mean().item(),
            "advantage_mean": advantages.mean().item(),
        }

    def _compute_gae(self, buffer: StateRolloutBuffer) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = self.cfg.gamma
        lam = self.cfg.lam

        losses = buffer.losses
        values = buffer.values.squeeze(-1)
        next_values = buffer.next_values.squeeze(-1)
        resets = buffer.resets
        terminations = buffer.terminations

        advantages = torch.zeros_like(losses)
        gae = torch.zeros_like(losses[0])
        for t in reversed(range(len(losses))):
            nonterminal = 1.0 - terminations[t]
            nonreset = 1.0 - resets[t]
            delta = losses[t] + gamma * next_values[t] * nonterminal - values[t]
            gae = delta + gamma * lam * nonreset * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns.reshape(-1, 1), advantages.reshape(-1, 1)

    def _update_critic(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        values = self.critic(states.to(self.device))
        critic_loss = F.mse_loss(values, returns.to(self.device))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optimizer.step()
        return critic_loss.item()

    def _update_actor(self, buffer: StateRolloutBuffer) -> Tuple[float, float]:
        if buffer.actor_loss_graph is None:
            raise RuntimeError("Rollout buffer does not contain an actor loss graph.")

        self.actor_optimizer.zero_grad()
        actor_loss = buffer.actor_loss_graph
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        return actor_loss.detach().item(), buffer.mean_entropy.detach().item()

    def _soft_update_target(self):
        tau = 0.005
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "target_critic_state_dict": self.target_critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )


class MASHAC:
    """Training loop for shared-actor multi-agent SHAC."""

    def __init__(
        self,
        env,
        cfg: Optional[TrainingCfg] = None,
    ):
        self.env = env
        self.cfg = cfg or TrainingCfg()

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        if hasattr(obs_space, "spaces") and "policy" in obs_space.spaces:
            obs_dim = obs_space.spaces["policy"].shape[0]
        else:
            obs_dim = obs_space.shape[0]

        action_space = getattr(env, "single_action_space", env.action_space)
        action_dim = action_space.shape[0]

        state_dim = self._flatten_state(self._get_env_state()).shape[-1]

        self.agent = MASHACAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            cfg=self.cfg,
        )
        self.buffer = StateRolloutBuffer(
            num_envs=env.num_envs,
            horizon=self.cfg.rollout_horizon,
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.cfg.device,
        )

        self.iteration = 0
        self.last_extras = {}
        self.writer = SummaryWriter(self.cfg.log_dir) if self.cfg.enable_tensorboard else None

    def train(self):
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        try:
            while self.iteration < self.cfg.num_iterations:
                obs = self._collect_rollout(obs)
                metrics = self.agent.update(self.buffer)

                if self.iteration % self.cfg.log_interval == 0:
                    self._log_metrics(metrics)
                if self.iteration % self.cfg.save_interval == 0:
                    self._save_checkpoint()
                if hasattr(self.env, "detach_graph"):
                    self.env.detach_graph()
                self.iteration += 1
        finally:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

    def _collect_rollout(self, obs):
        self.buffer.reset()

        cumulated_loss = torch.zeros(self.env.num_envs, device=self.agent.device)
        entropy_accum = torch.zeros(self.env.num_envs, device=self.agent.device)

        for step in range(self.cfg.rollout_horizon):
            policy_obs = self._policy_obs(obs)
            state = self._flatten_state(self._get_env_state())
            action, log_prob, entropy = self.agent.get_action(policy_obs)

            next_obs, next_state, loss_terms, reward, extras = self.env.step(action)
            next_policy_obs = extras.get("next_obs_before_reset", self._policy_obs(next_obs))
            next_state_graph = self._flatten_state(extras.get("next_state_before_reset", next_state))
            terminated = extras["terminated"]
            truncated = extras["truncated"]
            reset = terminated | truncated
            self.last_extras = extras

            with torch.no_grad():
                value = self.agent.critic(state.detach().to(self.agent.device))

            next_value_graph = self.agent.target_critic(next_state_graph.to(self.agent.device))
            next_value = next_value_graph.detach()

            gamma = self.cfg.gamma
            cumulated_loss = cumulated_loss + loss_terms.to(self.agent.device) * (gamma ** step)
            entropy_accum = entropy_accum + entropy.squeeze(-1)

            is_rollout_end = step == self.cfg.rollout_horizon - 1
            if is_rollout_end:
                should_bootstrap = ~terminated
            else:
                should_bootstrap = truncated & ~terminated
            cumulated_loss = cumulated_loss + (gamma ** (step + 1)) * next_value_graph.squeeze(-1) * should_bootstrap.float().to(self.agent.device)

            self.buffer.add(
                obs=policy_obs.detach(),
                next_obs=next_policy_obs.detach(),
                action=action.detach(),
                loss=loss_terms.detach(),
                reward=reward.detach(),
                done=reset.detach(),
                state=state.detach(),
                next_state=next_state_graph.detach(),
                terminated=terminated.detach(),
                reset=reset.detach(),
                log_prob=log_prob.detach(),
                value=value.detach(),
                next_value=next_value.detach(),
            )
            obs = next_obs

        self.buffer.actor_loss_graph = cumulated_loss.mean() - self.cfg.entropy_coef * entropy_accum.mean()
        self.buffer.mean_entropy = entropy_accum.mean() / self.cfg.rollout_horizon
        final_state = self._flatten_state(self._get_env_state()).detach()
        self.buffer.bootstrap(self.agent.target_critic(final_state.to(self.agent.device)).detach())
        return obs

    def _log_metrics(self, metrics: Dict[str, float]):
        if self.writer is None:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.iteration)

    def _save_checkpoint(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        save_path = f"{self.cfg.save_dir}/mashac_{self.iteration:06d}.pt"
        self.agent.save(save_path)

    @staticmethod
    def _policy_obs(obs: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def _get_env_state(self) -> torch.Tensor:
        if not hasattr(self.env, "drone"):
            raise RuntimeError("MASHAC requires an environment with drone state.")
        return self.env.drone.get_flat_state().view(self.env.num_envs, -1)

    @staticmethod
    def _flatten_state(state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            return state
        return state.view(state.shape[0], -1)
