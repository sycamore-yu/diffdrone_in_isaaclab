"""World model environment for imagined trajectories."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from diffaero_newton.training.dreamerv3.models.state_predictor import DepthStateModel
from diffaero_newton.training.dreamerv3.wmenv.replaybuffer import ReplayBuffer


ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]


@dataclass
class DepthStateEnvConfig:
    """Configuration for DepthStateEnv."""

    horizon: int
    batch_size: int
    batch_length: int
    use_perception: bool = False
    use_extern: bool = False


class DepthStateEnv:
    """World model environment for imagined rollouts."""

    def __init__(
        self,
        state_model: DepthStateModel,
        replaybuffer: ReplayBuffer,
        cfg: DepthStateEnvConfig,
    ) -> None:
        self.state_model = state_model
        self.replaybuffer = replaybuffer
        self.cfg = cfg
        self.hidden = None
        self.latent = None
        self.use_extern = cfg.use_extern

    @torch.no_grad()
    def make_generator_init(self) -> Tuple[Tensor, Tensor]:
        """Initialize latent state from replay buffer for imagined trajectories."""
        batch_size = self.cfg.batch_size
        batch_length = self.cfg.batch_length
        if self.use_extern:
            states, actions, perceptions = self.replaybuffer.sample_extern(batch_size, batch_length)
        else:
            states, actions, _, _, perceptions = self.replaybuffer.sample(batch_size, batch_length)
        hidden = None

        for i in range(batch_length):
            if perceptions is not None:
                latent, _ = self.state_model.sample_with_post(states[:, i], perceptions[:, i], hidden)
            else:
                latent, _ = self.state_model.sample_with_post(states[:, i], None, hidden)
            latent = self.state_model.flatten(latent)
            latent, _, hidden = self.state_model.sample_with_prior(latent, actions[:, i], hidden)

        latent = self.state_model.flatten(latent)
        self.latent = latent
        self.hidden = hidden
        return latent, hidden

    @torch.no_grad()
    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Step the world model forward."""
        assert action.ndim == 2
        prior_sample, pred_reward, pred_end, hidden = self.state_model.predict_next(
            latent=self.latent, act=action, hidden=self.hidden
        )
        flattened_sample = prior_sample.view(*prior_sample.shape[:-2], -1)
        self.latent = flattened_sample
        self.hidden = hidden
        return flattened_sample, pred_reward, pred_end, hidden

    def decode(self, latents: Tensor, hiddens: Tensor) -> Tensor:
        """Decode latents to images for visualization."""
        _, videos = self.state_model.decode(latents, hiddens)
        assert videos.ndim == 4, f"Expected videos to have 4 dimensions, got {videos.ndim}"
        return videos
