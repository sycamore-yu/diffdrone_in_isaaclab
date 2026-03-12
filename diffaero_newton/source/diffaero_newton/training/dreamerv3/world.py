"""World_Agent - DreamerV3 world-model training for IsaacLab environments."""

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffaero_newton.training.dreamerv3.models.state_predictor import DepthStateModel, DepthStateModelCfg
from diffaero_newton.training.dreamerv3.models.agent import ActorCriticAgent, ActorCriticConfig
from diffaero_newton.training.dreamerv3.models.blocks import symlog
from diffaero_newton.training.dreamerv3.wmenv.world_state_env import DepthStateEnv, DepthStateEnvConfig
from diffaero_newton.training.dreamerv3.wmenv.replaybuffer import ReplayBuffer, ReplayBufferCfg
from diffaero_newton.training.dreamerv3.wmenv.utils import configure_opt, DictConfig


def collect_imagine_trj(
    env: DepthStateEnv,
    agent: ActorCriticAgent,
    imagine_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Collect imagined trajectories from world model."""
    latents, hiddens, rewards, ends, actions, org_samples = [], [], [], [], [], []
    latent, hidden = env.make_generator_init()

    for i in range(imagine_length):
        latents.append(latent)
        hiddens.append(hidden)
        action, org_sample = agent.sample(torch.cat([latent, hidden], dim=-1))
        latent, reward, end, hidden = env.step(action)
        rewards.append(reward)
        actions.append(action)
        org_samples.append(org_sample)
        ends.append(end)

    latents.append(latent)
    hiddens.append(hidden)
    latents = torch.stack(latents, dim=1)
    hiddens = torch.stack(hiddens, dim=1)
    actions = torch.stack(actions, dim=1)
    org_samples = torch.stack(org_samples, dim=1)
    rewards = torch.stack(rewards, dim=1)
    ends = torch.stack(ends, dim=1)

    return latents, hiddens, actions, rewards, ends, org_samples


def generate_video(
    env: DepthStateEnv,
    agent: ActorCriticAgent,
    imagine_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate imagined video for visualization."""
    latents, hiddens, _, _, _, _ = collect_imagine_trj(env, agent, imagine_length, device)
    videos = env.decode(latents, hiddens)
    videos = videos[:: videos.shape[0] // 16]
    return videos.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # B L 3 H W


def train_agents(
    agent: ActorCriticAgent,
    state_env: DepthStateEnv,
    training_cfg: DictConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Train actor-critic agents on imagined trajectories."""
    latents, hiddens, _, rewards, ends, org_samples = collect_imagine_trj(
        state_env, agent, training_cfg.imagine_length, device
    )
    agent_info = agent.update(
        torch.cat([latents, hiddens], dim=-1), org_samples, rewards, ends
    )
    reward_sum = rewards.sum(dim=-1).mean()
    agent_info["reward_sum"] = reward_sum.item()
    return agent_info


def train_worldmodel(
    world_model: DepthStateModel,
    replaybuffer: ReplayBuffer,
    opt: torch.optim.Optimizer,
    training_cfg: DictConfig,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
) -> Dict[str, float]:
    """Train world model on replay buffer samples."""
    use_amp = training_cfg.get("use_amp", False)
    dtype = torch.float16 if use_amp else torch.float32

    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
        for _ in range(training_cfg.get("worldmodel_update_freq", 1)):
            sample_state, sample_action, sample_reward, sample_termination, sample_perception = (
                replaybuffer.sample(training_cfg.batch_size, training_cfg.batch_length)
            )
            total_loss, rep_loss, dyn_loss, rec_loss, rew_loss, end_loss = world_model.compute_loss(
                sample_state,
                sample_perception,
                sample_action,
                sample_reward,
                sample_termination,
            )

    if scaler is not None:
        scaler.scale(total_loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            world_model.parameters(), training_cfg.get("max_grad_norm", 100.0)
        )
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    else:
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            world_model.parameters(), training_cfg.get("max_grad_norm", 100.0)
        )
        opt.step()
        opt.zero_grad(set_to_none=True)

    world_info = {
        "WorldModel/state_total_loss": total_loss.item(),
        "WorldModel/state_rep_loss": rep_loss.item(),
        "WorldModel/state_dyn_loss": dyn_loss.item(),
        "WorldModel/state_rec_loss": rec_loss.item(),
        "WorldModel/grad_norm": grad_norm.item(),
        "WorldModel/state_rew_loss": rew_loss.item(),
        "WorldModel/state_end_loss": end_loss.item(),
    }

    return world_info


class World_Agent:
    """World model agent for DreamerV3 training with IsaacLab environments."""

    def __init__(
        self,
        cfg: Union[DictConfig, Dict[str, Any]],
        env,  # IsaacLab environment
        device: torch.device,
    ):
        """
        Initialize World_Agent.

        Args:
            cfg: Configuration object
            env: IsaacLab environment (DroneEnv)
            device: torch device
        """
        self.cfg = cfg if isinstance(cfg, DictConfig) else DictConfig(cfg)
        self.env = env
        self.device = device

        # Get number of environments
        if hasattr(env, "num_envs"):
            self.n_envs = env.num_envs
        else:
            self.n_envs = 1

        # Determine state dimension
        if hasattr(env, "cfg") and hasattr(env.cfg, "num_states"):
            state_dim = env.cfg.num_states
        else:
            state_dim = 13  # Default for drone

        # Build configs
        self._build_configs(state_dim)

        # Create models
        self.agent = ActorCriticAgent(self.actorcritic_cfg, env).to(device)
        self.state_model = DepthStateModel(self.statemodel_cfg).to(device)

        # Create replay buffer and world model env
        if not self.cfg.get("is_test", False):
            self.replaybuffer = ReplayBuffer(self.buffercfg)
            self.world_model_env = DepthStateEnv(self.state_model, self.replaybuffer, self.worldcfg)

        # Setup optimizer
        opt_cfg = self.cfg.get("optimizer", {})
        self.opt = configure_opt(self.state_model, **opt_cfg)

        # Setup scaler for AMP
        use_amp = self.training_hyper.get("use_amp", False)
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Load checkpoint if specified
        if self.cfg.get("checkpoint_path") is not None:
            self.load(self.cfg.checkpoint_path)

        self.num_steps = 0

        # Initialize hidden state
        self.hidden = torch.zeros(self.n_envs, self.statemodel_cfg.hidden_dim, device=device)

    def _build_configs(self, state_dim: int):
        """Build model configurations."""
        # State model config
        sm_cfg = self.cfg.get("state_predictor", {})
        self.statemodel_cfg = DepthStateModelCfg(
            state_dim=state_dim,
            image_width=sm_cfg.get("image_width", 16),
            image_height=sm_cfg.get("image_height", 9),
            hidden_dim=sm_cfg.get("hidden_dim", 256),
            action_dim=sm_cfg.get("action_dim", 4),
            latent_dim=sm_cfg.get("latent_dim", 256),
            categoricals=sm_cfg.get("categoricals", 16),
            num_classes=sm_cfg.get("num_classes", 255),
            end_loss_pos_weight=sm_cfg.get("end_loss_pos_weight", 1.0),
            img_recon_loss_weight=sm_cfg.get("img_recon_loss_weight", 1.0),
            use_simnorm=sm_cfg.get("use_simnorm", False),
            only_state=sm_cfg.get("only_state", True),  # Default to state-only for drone
            enable_rec=sm_cfg.get("enable_rec", False),
        )

        # Actor-critic config
        ac_cfg = self.cfg.get("actor_critic", {})
        self.actorcritic_cfg = ActorCriticConfig(
            feat_dim=self.statemodel_cfg.hidden_dim + self.statemodel_cfg.latent_dim,
            num_layers=ac_cfg.get("num_layers", 2),
            hidden_dim=self.statemodel_cfg.hidden_dim,
            action_dim=self.statemodel_cfg.action_dim,
            gamma=ac_cfg.get("gamma", 0.99),
            lambd=ac_cfg.get("lambd", 0.95),
            entropy_coef=ac_cfg.get("entropy_coef", 1e-4),
            device=self.device,
        )

        # Replay buffer config
        rb_cfg = self.cfg.get("replaybuffer", {})
        self.buffercfg = ReplayBufferCfg(
            perception_width=sm_cfg.get("image_width", 16),
            perception_height=sm_cfg.get("image_height", 9),
            state_dim=state_dim,
            action_dim=self.statemodel_cfg.action_dim,
            num_envs=self.n_envs,
            max_length=rb_cfg.get("max_length", 100000),
            warmup_length=rb_cfg.get("warmup_length", 5000),
            store_on_gpu=rb_cfg.get("store_on_gpu", True),
            device=str(self.device),
            use_perception=sm_cfg.get("use_perception", False),
        )

        # World model env config
        wm_cfg = self.cfg.get("world_state_env", {})
        self.worldcfg = DepthStateEnvConfig(
            horizon=wm_cfg.get("horizon", 64),
            batch_size=wm_cfg.get("batch_size", 16),
            batch_length=wm_cfg.get("batch_length", 64),
            use_perception=sm_cfg.get("use_perception", False),
        )

        # Training hyperparams
        self.training_hyper = DictConfig(
            {
                "batch_size": wm_cfg.get("batch_size", 16),
                "batch_length": wm_cfg.get("batch_length", 64),
                "imagine_length": wm_cfg.get("imagine_length", 64),
                "worldmodel_update_freq": sm_cfg.get("worldmodel_update_freq", 1),
                "use_amp": sm_cfg.get("use_amp", False),
                "max_grad_norm": sm_cfg.get("max_grad_norm", 100.0),
            }
        )

    @torch.no_grad()
    def act(self, obs, test: bool = False):
        """Act in environment (for inference)."""
        if not isinstance(obs, torch.Tensor):
            state = obs.get("state", obs.get("observation"))
            perception = obs.get("perception")
            if perception is not None:
                perception = perception.unsqueeze(1)
        else:
            state = obs
            perception = None

        if self.cfg.get("use_symlog", True):
            state = symlog(state)

        latent = self.state_model.sample_with_post(state, perception, self.hidden, True)[0].flatten(1)
        action = self.agent.sample(torch.cat([latent, self.hidden], dim=-1), test)[0]
        prior_sample, _, self.hidden = self.state_model.sample_with_prior(
            latent, action, self.hidden, True
        )
        return action, None

    def step(
        self,
        obs,
        on_step_cb=None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any], float, float]:
        """
        Step the agent in the environment.

        Returns:
            next_obs, policy_info, env_info, reward, done
        """
        policy_info = {}

        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                state = obs.get("state", obs.get("observation"))
                perception = obs.get("perception")
                if perception is not None:
                    perception = perception.unsqueeze(1)
            else:
                state = obs
                perception = None

            if self.cfg.get("use_symlog", True):
                state = symlog(state)

            if self.replaybuffer.ready() or self.cfg.get("checkpoint_path") is not None:
                latent = self.state_model.sample_with_post(state, perception, self.hidden)[0].flatten(1)
                action = self.agent.sample(torch.cat([latent, self.hidden], dim=-1))[0]
                prior_sample, _, self.hidden = self.state_model.sample_with_prior(
                    latent, action, self.hidden
                )
            else:
                # Random action during warmup
                action = torch.randn(self.n_envs, self.statemodel_cfg.action_dim, device=state.device)

            # Step environment
            next_obs, reward, terminated, truncated, env_info = self.env.step(action)
            reward = reward * 10.0

            # Store in replay buffer
            self.replaybuffer.append(state, action, reward, terminated, perception)

            # Reset hidden state for terminated episodes
            if terminated.any():
                zeros = torch.zeros_like(self.hidden)
                self.hidden = torch.where(terminated.unsqueeze(-1), zeros, self.hidden)

        # Train world model and agent if ready
        if self.replaybuffer.ready():
            world_info = train_worldmodel(
                self.state_model,
                self.replaybuffer,
                self.opt,
                self.training_hyper,
                self.scaler,
                self.device,
            )
            agent_info = train_agents(
                self.agent, self.world_model_env, self.training_hyper, self.device
            )
            policy_info.update(world_info)
            policy_info.update(agent_info)

        self.num_steps += 1

        return next_obs, policy_info, env_info, reward.mean().item(), (terminated | truncated).float().mean().item()

    def finetune(self) -> Dict[str, float]:
        """Fine-tune agent on world model (for adaptation)."""
        agent_info = train_agents(self.agent, self.world_model_env, self.training_hyper, self.device)
        return agent_info

    def save(self, path: str):
        """Save model checkpoints."""
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_model.state_dict(), f"{path}/statemodel.pth")
        torch.save(self.agent.state_dict(), f"{path}/agent.pth")

    def load(self, path: str):
        """Load model checkpoints."""
        self.state_model.load_state_dict(torch.load(os.path.join(path, "statemodel.pth")))
        self.agent.load_state_dict(torch.load(os.path.join(path, "agent.pth")))

    @staticmethod
    def build(cfg, env, device):
        """Factory method to build World_Agent."""
        return World_Agent(cfg, env, device)
