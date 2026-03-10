"""Training configuration for SHAC-style algorithm."""

from dataclasses import dataclass, field
from typing import Optional

from diffaero_newton.common.constants import DEFAULT_ROLLOUT_HORIZON


@dataclass
class TrainingCfg:
    """Configuration for SHAC-style training.

    Attributes:
        # Rollout parameters
        rollout_horizon: Number of steps for short-horizon rollout.
        gamma: Discount factor.
        lam: GAE lambda parameter.

        # Actor parameters
        actor_lr: Actor learning rate.
        actor_hidden_dims: Actor network hidden dimensions.
        actor_log_std_init: Initial log standard deviation.

        # Critic parameters
        critic_lr: Critic learning rate.
        critic_hidden_dims: Critic network hidden dimensions.

        # Training parameters
        num_iterations: Number of training iterations.
        num_envs: Number of parallel environments.
        batch_size: Batch size for updates.
        clip_epsilon: PPO clip epsilon.
        value_loss_coef: Value function loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        use_gae: Whether to use GAE.

        # Logging
        log_interval: Logging interval.
        save_interval: Checkpoint save interval.
    """

    # Rollout parameters
    rollout_horizon: int = DEFAULT_ROLLOUT_HORIZON
    gamma: float = 0.99
    lam: float = 0.95

    # Actor parameters
    actor_lr: float = 3.0e-4
    actor_hidden_dims: list = field(default_factory=lambda: [256, 256, 128])
    actor_log_std_init: float = -0.5

    # Critic parameters
    critic_lr: float = 1.0e-3
    critic_hidden_dims: list = field(default_factory=lambda: [256, 256, 128])

    # Training parameters
    num_iterations: int = 10000
    num_steps_per_iteration: int = 2000  # rollout_steps * num_envs
    batch_size: int = 64
    num_epochs: int = 10
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    use_gae: bool = True

    # Learning rate schedule
    lr_schedule: str = "constant"  # 'constant', 'linear', 'cosine'

    # Device
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 1
    save_interval: int = 100
    eval_interval: int = 10
    eval_episodes: int = 10

    # Checkpoint
    save_dir: str = "checkpoints"
    resume_from: Optional[str] = None


@dataclass
class NetworkCfg:
    """Configuration for neural network architecture."""

    activation: str = "elu"
    use_orthogonal_init: bool = True
    use_layer_norm: bool = False


@dataclass
class OptimizerCfg:
    """Configuration for optimizer."""

    type: str = "adam"
    eps: float = 1.0e-5
    weight_decay: float = 0.0
    momentum: float = 0.9
