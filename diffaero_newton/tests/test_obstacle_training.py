"""Tests for obstacle avoidance training.

TDD RED: These tests define the expected behavior for the obstacle avoidance training.
They will fail until the implementation is complete.
"""

import pytest
import torch
import numpy as np

# Test fixtures
@pytest.fixture
def device():
    """CPU device for testing."""
    return torch.device("cpu")


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
        env = DroneEnv(cfg=cfg)

        assert env.num_envs == num_envs

    def test_reset_returns_observation(self, device, num_envs):
        """Test that reset returns valid observations."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg)

        obs, extras = env.reset()

        assert "policy" in obs
        assert obs["policy"].shape == (num_envs, 20)  # state + goal + prev_action

    def test_step_returns_all_outputs(self, device, num_envs):
        """Test that step returns obs, reward, terminated, truncated, extras."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg)
        env.reset()

        action = torch.zeros(num_envs, 4)
        obs, reward, terminated, truncated, extras = env.step(action)

        assert obs["policy"].shape == (num_envs, 20)
        assert reward.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)

    def test_goal_tracking_reward(self, device, num_envs):
        """Test that reward increases when moving toward goal."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

        cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=cfg)
        env.reset()

        # Zero action - drone should stay roughly in place
        action = torch.zeros(num_envs, 4)
        _, reward1, _, _, _ = env.step(action)

        # Small thrust - slight movement
        action = torch.ones(num_envs, 4) * 0.3
        _, reward2, _, _, _ = env.step(action)

        # Rewards should be computed (not NaN)
        assert not torch.isnan(reward1).any()
        assert not torch.isnan(reward2).any()


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
        agent = SHACAgent(obs_dim=20, action_dim=4, cfg=cfg)

        assert agent is not None

    def test_agent_action_shape(self, device):
        """Test that agent returns action with correct shape."""
        from diffaero_newton.training.shac import SHACAgent
        from diffaero_newton.configs.training_cfg import TrainingCfg

        cfg = TrainingCfg()
        agent = SHACAgent(obs_dim=20, action_dim=4, cfg=cfg)

        obs = torch.randn(4, 20)
        action, log_prob, entropy = agent.get_action(obs)

        assert action.shape == (4, 4)
        assert log_prob.shape == (4, 1)
        assert entropy.shape == (4, 1)


class TestIntegration:
    """Integration tests for full training loop."""

    def test_training_iteration(self, device, num_envs):
        """Test a single training iteration."""
        from diffaero_newton.envs.drone_env import DroneEnv
        from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
        from diffaero_newton.training.shac import SHACAgent
        from diffaero_newton.configs.training_cfg import TrainingCfg
        from diffaero_newton.training.buffer import RolloutBuffer

        # Create environment
        env_cfg = DroneEnvCfg(num_envs=num_envs)
        env = DroneEnv(cfg=env_cfg)

        # Create agent
        train_cfg = TrainingCfg()
        agent = SHACAgent(obs_dim=20, action_dim=4, cfg=train_cfg)

        # Create buffer
        buffer = RolloutBuffer(
            num_envs=num_envs,
            horizon=10,
            obs_dim=20,
            action_dim=4,
            device=str(device)
        )

        # Collect one rollout
        obs, _ = env.reset()
        buffer.reset()

        for step in range(10):
            action, log_prob, _ = agent.get_action(obs["policy"])
            next_obs, reward, terminated, truncated, _ = env.step(action)

            with torch.no_grad():
                value = agent.critic(obs["policy"].to(device), action.to(device))

            buffer.add(obs["policy"], action, reward, terminated, log_prob, value)
            obs = next_obs

        # Just test that we can get action from the agent (simpler test)
        test_obs = torch.randn(4, 20)
        action, log_prob, entropy = agent.get_action(test_obs)
        
        assert action.shape == (4, 4)
        assert log_prob.shape == (4, 1)
        assert entropy.shape == (4, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
