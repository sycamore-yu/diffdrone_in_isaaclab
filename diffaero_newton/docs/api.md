# DiffAero Newton API Documentation

## Common Module

### Constants

```python
from diffaero_newton.common.constants import *
```

| Constant | Value | Description |
|----------|-------|-------------|
| GRAVITY | 9.81 | Gravity (m/s²) |
| QUADROTOR_MASS | 1.0 | Drone mass (kg) |
| ARM_LENGTH | 0.2 | Arm length (m) |
| COLLISION_RADIUS | 0.3 | Drone collision radius (m) |
| DEFAULT_DT | 0.01 | Simulation timestep (s) |
| DEFAULT_ROLLOUT_HORIZON | 10 | SHAC rollout horizon |
| STATE_DIM | 13 | State dimension [pos, quat, vel, omega] |
| ACTION_DIM | 4 | Action dimension (4 motors) |

## Dynamics Module

### Drone

```python
from diffaero_newton.dynamics.drone_dynamics import Drone, DroneConfig, create_drone
```

#### DroneConfig

Configuration for drone dynamics:

```python
config = DroneConfig(
    num_envs=256,      # Number of parallel environments
    dt=0.01,          # Simulation timestep
    requires_grad=False,  # Enable gradient tracking
    arm_length=0.2,   # Arm length
    mass=1.0,         # Drone mass
)
```

#### Drone Methods

| Method | Description |
|--------|-------------|
| `reset_states(positions)` | Reset drone to initial state |
| `apply_control(thrust)` | Apply normalized thrust [0, 1] |
| `integrate(dt)` | Step simulation forward |
| `get_state()` | Get state as dict of tensors |
| `get_flat_state()` | Get state as flat tensor [N, 13] |

### Rollout Functions

```python
from diffaero_newton.dynamics.rollout import rollout_onestep, rollout_horizon
```

| Function | Description |
|----------|-------------|
| `rollout_onestep(drone, state, control, dt)` | Single-step state transition |
| `rollout_horizon(env_cfg, states, actions, dt)` | Multi-step differentiable rollout |
| `RolloutBuffer` | Buffer for storing trajectories |

## Environment Module

### DroneEnv

```python
from diffaero_newton.envs.drone_env import DroneEnv, create_env
```

#### DroneEnv Methods

| Method | Description |
|--------|-------------|
| `reset(seed, options)` | Reset environment |
| `step(action)` | Execute one step |
| `_get_observations()` | Compute observations |
| `_get_rewards()` | Compute rewards (detached) |
| `_get_dones()` | Compute done flags |

#### Output Channels

- `obs["policy"]` - Actor-facing observation
- `state` - Critic-facing global state
- `reward` - Detached RL signal
- `extras` - Diagnostics

## Task Module

### ObstacleManager

```python
from diffaero_newton.tasks.obstacle_manager import ObstacleManager
```

| Method | Description |
|--------|-------------|
| `compute_distances(positions)` | Distance to all obstacles |
| `compute_nearest_distances(positions)` | Distance to nearest obstacle |
| `check_collisions(positions)` | Check collision status |
| `reset(env_ids)` | Reset obstacles |

### Reward Terms

```python
from diffaero_newton.tasks.reward_terms import (
    compute_risk_loss,
    compute_rewards,
    compute_goal_progress,
)
```

### Observations

```python
from diffaero_newton.tasks.observations import (
    build_state_observation,
    build_goal_observation,
    build_obstacle_observation,
    build_full_observation,
)
```

## Training Module

### SHAC

```python
from diffaero_newton.training.shac import SHAC, SHACAgent
from diffaero_newton.training.buffer import RolloutBuffer
```

#### Training Configuration

```python
from diffaero_newton.configs.training_cfg import TrainingCfg

cfg = TrainingCfg(
    rollout_horizon=10,     # Short horizon
    gamma=0.99,             # Discount factor
    lam=0.95,               # GAE lambda
    actor_lr=3e-4,          # Actor learning rate
    critic_lr=1e-3,         # Critic learning rate
    num_iterations=10000,   # Training iterations
)
```

#### SHAC Methods

| Method | Description |
|--------|-------------|
| `train()` | Run training loop |
| `agent.get_action(obs)` | Get action from observation |
| `agent.update(buffer)` | Update actor-critic |

## Configuration Classes

### DroneEnvCfg

```python
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

cfg = DroneEnvCfg(
    num_envs=256,
    episode_length_s=30.0,
    observation_space=18,
    action_space=4,
    decimation=4,
)
```

### ObstacleTaskCfg

```python
from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg

cfg = ObstacleTaskCfg(
    num_obstacles=5,
    obstacle_radius=0.5,
    collision_radius=0.3,
    goal_reward_scale=10.0,
    collision_penalty=100.0,
)
```
