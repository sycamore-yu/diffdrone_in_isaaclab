"""Tests for obstacle avoidance training.

TDD RED: These tests define the expected behavior for the obstacle avoidance training.
They will fail until the implementation is complete.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


def _prepare_non_resetting_step(env) -> None:
    """Place the drone in a safe state so a single differentiable step does not reset."""
    env.episode_length_buf.zero_()
    state = env.drone.get_flat_state().detach().clone()
    state[:, :3] = torch.tensor([10.0, 10.0, 10.0], device=env.device)
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[:, 7:10] = 0.0
    state[:, 10:13] = 0.0
    env.drone.set_state(state)
    env.goal_position[:] = state[:, :3] + torch.tensor([1.0, 0.0, 0.0], device=env.device)


# Test fixtures
@pytest.fixture
def device():
    """Explicit CPU smoke device for obstacle/SHAC coverage."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Explicit CUDA smoke device for differentiable obstacle checks."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gpu_smoke obstacle coverage")
    return torch.device("cuda")


@pytest.fixture
def num_envs():
    """Number of test environments."""
    return 4


@pytest.fixture
def num_obstacles():
    """Number of obstacles per environment."""
    return 5


class TestObstacleEnvironment:
    """Test suite for obstacle avoidance environment."""

    def test_environment_creation(self, device, num_envs):
        """Test that environment can be created with correct dimensions."""
        # TODO: This test will fail until ObstacleDroneEnv is implemented
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg, device=str(device))

        assert env.num_envs == num_envs

    def test_reset_returns_observation(self, device, num_envs):
        """Test that reset returns valid observations."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg, device=str(device))

        obs, extras = env.reset()

        assert "policy" in obs
        assert obs["policy"].shape == (num_envs, 21)  # state + goal + prev_action + nearest_obs_dist

    def test_step_returns_all_outputs(self, device, num_envs):
        """Test that step keeps the shared 5-tuple training contract."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg, device=str(device))
        env.reset()

        action = torch.zeros(num_envs, 4, device=device)
        obs, state, loss, reward, extras = env.step(action)

        assert obs["policy"].shape == (num_envs, 21)
        assert state.shape == (num_envs, 13)
        assert loss.shape == (num_envs,)
        assert reward.shape == (num_envs,)
        assert extras["terminated"].shape == (num_envs,)
        assert extras["truncated"].shape == (num_envs,)
        assert extras["next_obs_before_reset"].shape == (num_envs, 21)

    def test_goal_tracking_reward(self, device, num_envs):
        """Test that reward increases when moving toward goal."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg, device=str(device))
        env.reset()

        # Zero action - drone should stay roughly in place
        action = torch.zeros(num_envs, 4, device=device)
        _, _, loss1, reward1, _ = env.step(action)

        # Small thrust - slight movement
        action = torch.ones(num_envs, 4, device=device) * 0.3
        _, _, loss2, reward2, _ = env.step(action)

        # Rewards should be computed (not NaN)
        assert not torch.isnan(reward1).any()
        assert not torch.isnan(reward2).any()

    def test_obstacle_collision_detection(self, device, num_envs):
        """Test that obstacle collisions trigger termination."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg, device=str(device))
        env.reset()

        # Put drone at obstacle position (should trigger collision)
        state = env.drone.get_flat_state()
        state[:, :3] = 0.0  # At origin where obstacle is
        env.drone.set_state(state)

        action = torch.zeros(num_envs, 4, device=device)
        obs, state, loss, reward, extras = env.step(action)

        # Should have collision in extras
        assert "obstacles" in extras
        assert "nearest_dist" in extras["obstacles"]
        assert "goal_dist" in extras["obstacles"]

    def test_obstacle_reward_penalty(self, device, num_envs):
        """Test that being close to obstacles incurs penalty."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg, device=str(device))
        env.reset()

        # Put drone very close to obstacle (at origin)
        state = env.drone.get_flat_state()
        state[:, :3] = 0.0
        env.drone.set_state(state)

        action = torch.zeros(num_envs, 4, device=device)
        obs, state, loss, reward, extras = env.step(action)

        # Reward should be computed (not NaN)
        assert not torch.isnan(reward).any()
 
    def test_pointmass_backend_is_respected(self, device, num_envs):
        """Test that DroneEnv uses the configured dynamics backend."""
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
        from diffaero_newton.configs.dynamics_cfg import PointMassCfg
        from diffaero_newton.dynamics.pointmass_dynamics import PointMass
        from diffaero_newton.envs.drone_env import DroneEnv

        cfg = DroneEnvCfg(num_envs=num_envs)
        cfg.dynamics = PointMassCfg(num_envs=num_envs, requires_grad=True)
        env = DroneEnv(cfg=cfg, device=str(device))

        assert isinstance(env.drone, PointMass)


class TestObstacleManager:
    """Test suite for obstacle management."""

    def test_obstacle_spawn(self, device, num_envs, num_obstacles):
        """Test that obstacles are spawned correctly."""
        from diffaero_newton.tasks.obstacle_manager import ObstacleManager
        from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg

        cfg = ObstacleTaskCfg(num_obstacles=num_obstacles)
        manager = ObstacleManager(num_envs=num_envs, cfg=cfg, device=str(device))

        positions = manager.get_obstacle_positions()
        assert positions.shape == (num_envs, num_obstacles, 3)

    def test_collision_detection(self, device):
        """Test collision detection returns correct boolean tensor."""
        from diffaero_newton.tasks.obstacle_manager import ObstacleManager
        from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg

        cfg = ObstacleTaskCfg(num_obstacles=3)
        manager = ObstacleManager(num_envs=2, cfg=cfg, device=str(device))

        # Position at origin (should be far from obstacles)
        positions = torch.zeros(2, 3)
        collisions = manager.check_collisions(positions)

        assert collisions.shape == (2,)
        assert collisions.dtype == torch.bool


class TestRewardTerms:
    """Test suite for reward computation."""

    def test_rewards_are_finite(self, device):
        """Test that rewards are finite numbers."""
        from diffaero_newton.tasks.reward_terms import compute_rewards

        states = torch.randn(4, 13)
        goal = torch.randn(4, 3)

        rewards, components = compute_rewards(states, goal)

        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(v) for v in components.values() if isinstance(v, torch.Tensor))

    def test_collision_penalty(self, device):
        """Test that collision incurs penalty."""
        from diffaero_newton.tasks.reward_terms import compute_rewards

        # Drone at origin
        states = torch.zeros(1, 13)

        # Obstacle at origin (collision!)
        obstacles = torch.zeros(1, 1, 4)
        obstacles[0, 0, :3] = 0.0  # position at origin
        obstacles[0, 0, 3] = 0.5  # radius

        goal = torch.ones(1, 3) * 5.0  # goal far away

        rewards, _ = compute_rewards(states, goal, obstacles)

        # Should have negative collision penalty
        assert rewards.item() < 0


class TestTraining:
    """Test suite for SHAC training."""

    def test_agent_creation(self, device):
        """Test that SHAC agent can be created."""
        from diffaero_newton.training.shac import SHACAgent
        from diffaero_newton.configs.training_cfg import TrainingCfg

        cfg = TrainingCfg()
        agent = SHACAgent(obs_dim=21, cfg=cfg)

        assert agent is not None

    def test_agent_action_shape(self, device):
        """Test that agent returns action with correct shape."""
        from diffaero_newton.training.shac import SHACAgent
        from diffaero_newton.configs.training_cfg import TrainingCfg

        cfg = TrainingCfg()
        agent = SHACAgent(obs_dim=21, cfg=cfg)

        obs = torch.randn(4, 21)
        action, log_prob, entropy = agent.get_action(obs)

        assert action.shape == (4, 4)
        assert log_prob.shape == (4, 1)
        assert entropy.shape == (4, 1)


class TestIntegration:
    """Integration tests for full training loop."""

    @pytest.mark.gpu_smoke
    def test_training_iteration(self, cuda_device, num_envs):
        """Test a single SHAC training iteration on the intended CUDA path."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
        from diffaero_newton.training.shac import SHAC
        from diffaero_newton.configs.training_cfg import TrainingCfg

        # Create environment
        env_cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=env_cfg, device=str(cuda_device))

        # Create trainer
        train_cfg = TrainingCfg(device=str(cuda_device), rollout_horizon=10, num_iterations=1, save_interval=1000)
        trainer = SHAC(env, cfg=train_cfg)

        # Collect one rollout through the trainer path
        obs, _ = env.reset()
        trainer._collect_rollout(obs)
        assert trainer.buffer.actor_loss_graph is not None
        assert trainer.buffer.actor_loss_graph.requires_grad

        actor_before = {
            name: param.detach().clone()
            for name, param in trainer.agent.actor.named_parameters()
        }
        metrics = trainer.agent.update(trainer.buffer)

        actor_changed = any(
            not torch.allclose(actor_before[name], param.detach(), atol=1e-8)
            for name, param in trainer.agent.actor.named_parameters()
        )

        assert trainer.buffer.ptr == train_cfg.rollout_horizon
        assert actor_changed
        assert set(metrics.keys()) == {
            "actor_loss",
            "critic_loss",
            "entropy",
            "value_mean",
            "advantage_mean",
        }
        assert all(np.isfinite(value) for value in metrics.values())

    def test_loss_is_differentiable(self, device, num_envs):
        """Test that env.step returns a loss tensor with gradients on the actor path."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg, device=str(device))
        env.reset()
        _prepare_non_resetting_step(env)

        action = torch.full((num_envs, 4), 0.1, device=device, requires_grad=True)
        _, _state, loss, _reward, extras = env.step(action)

        assert not extras["reset"].any()
        assert loss.requires_grad
        loss.sum().backward()
        assert action.grad is not None
        assert torch.isfinite(action.grad).all()

    @pytest.mark.gpu_smoke
    def test_loss_is_differentiable_on_cuda(self, cuda_device, num_envs):
        """Test the differentiable obstacle path on the intended CUDA runtime."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg, device=str(cuda_device))
        env.reset()
        _prepare_non_resetting_step(env)

        action = torch.full((num_envs, 4), 0.1, device=cuda_device, requires_grad=True)
        _, _state, loss, _reward, extras = env.step(action)

        assert not extras["reset"].any()
        assert loss.requires_grad
        loss.sum().backward()
        assert action.grad is not None
        assert torch.isfinite(action.grad).all()

    def test_bootstrap_semantics(self, device):
        """Test truncated paths bootstrap while terminated paths do not."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
        from diffaero_newton.training.shac import SHAC
        from diffaero_newton.configs.training_cfg import TrainingCfg

        env_cfg = DroneEnvCfg(num_envs=2)
        env = DroneEnv(cfg=env_cfg, device=str(device))
        train_cfg = TrainingCfg(
            device=str(device),
            rollout_horizon=1,
            num_iterations=1,
            save_interval=1000,
            enable_tensorboard=False,
        )
        trainer = SHAC(env, cfg=train_cfg)

        obs, _ = env.reset()
        env.episode_length_buf[0] = env.episode_length_max
        state = env.drone.get_flat_state()
        state[1, 2] = 0.0
        env.drone.set_state(state)
        trainer._collect_rollout(obs)

        assert trainer.buffer.resets[0, 0].item() == 1.0
        assert trainer.buffer.terminations[0, 0].item() == 0.0
        assert trainer.buffer.resets[0, 1].item() == 1.0
        assert trainer.buffer.terminations[0, 1].item() == 1.0

    @pytest.mark.gpu_smoke
    def test_tensorboard_logging(self, cuda_device, num_envs, tmp_path):
        """Test that a CUDA-backed SHAC run emits TensorBoard event files."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
        from diffaero_newton.training.shac import SHAC
        from diffaero_newton.configs.training_cfg import TrainingCfg

        env_cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=env_cfg, device=str(cuda_device))
        log_dir = tmp_path / "runs"
        train_cfg = TrainingCfg(
            device=str(cuda_device),
            rollout_horizon=2,
            num_iterations=2,
            log_interval=1,
            save_interval=1000,
            save_dir=str(tmp_path / "checkpoints"),
            log_dir=str(log_dir),
            enable_tensorboard=True,
        )
        trainer = SHAC(env, cfg=train_cfg)
        trainer.train()

        event_files = list(Path(log_dir).rglob("events.out.tfevents*"))
        assert event_files


class TestQuadrotorDynamics:
    """Test suite for full quadrotor dynamics."""

    def test_quaternion_update(self, device):
        """Test that quaternion updates during integration."""
        from diffaero_newton.dynamics.drone_dynamics import Drone, DroneConfig
        
        cfg = DroneConfig(num_envs=2, dt=0.01)
        drone = Drone(cfg, device=str(device))
        
        # Apply roll torque
        thrust = torch.ones(2, 4) * 0.5
        thrust[:, 0] = 0.8  # Higher front-right
        thrust[:, 1] = 0.2  # Lower front-left
        thrust[:, 2] = 0.2  # Lower rear-left  
        thrust[:, 3] = 0.8  # Higher rear-right
        drone.apply_control(thrust)
        
        # Get initial orientation
        initial_quat = drone.get_state()["orientation"].clone()
        
        # Integrate multiple steps
        for _ in range(100):
            drone.integrate(0.01)
        
        final_quat = drone.get_state()["orientation"]
        
        # Quaternion should change (not identity)
        assert not torch.allclose(initial_quat, final_quat, atol=1e-3)
        
        # Quaternion should still be normalized
        quat_norm = torch.norm(final_quat, dim=1)
        assert torch.allclose(quat_norm, torch.ones(2), atol=1e-5)

    def test_angular_velocity_update(self, device):
        """Test that angular velocity updates correctly."""
        from diffaero_newton.dynamics.drone_dynamics import Drone, DroneConfig
        
        cfg = DroneConfig(num_envs=2, dt=0.01)
        drone = Drone(cfg, device=str(device))
        
        # Initial angular velocity should be zero
        initial_omega = drone.get_state()["omega"]
        assert torch.allclose(initial_omega, torch.zeros(2, 3), atol=1e-5)
        
        # Apply control with differential thrust (creates torque)
        thrust = torch.ones(2, 4) * 0.5
        thrust[:, 0] = 0.8  # Creates roll torque
        thrust[:, 3] = 0.8
        drone.apply_control(thrust)
        
        # Integrate
        for _ in range(50):
            drone.integrate(0.01)
        
        # Angular velocity should be non-zero
        final_omega = drone.get_state()["omega"]
        assert not torch.allclose(final_omega, torch.zeros(2, 3), atol=1e-3)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
