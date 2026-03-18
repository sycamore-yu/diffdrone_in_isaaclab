"""Obstacle manager for generating and managing obstacles in the environment."""

from typing import Optional, Tuple
import torch

from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg, ObstacleConfig
from diffaero_newton.common.constants import MAX_OBSTACLES


class ObstacleManager:
    """Manages obstacles for the obstacle avoidance task.

    This class handles obstacle generation, position queries, and collision detection.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: Optional[ObstacleTaskCfg] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the obstacle manager.

        Args:
            num_envs: Number of parallel environments.
            cfg: Obstacle task configuration.
            device: Device for tensor operations.
        """
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg or ObstacleTaskCfg()

        # Obstacle buffers: [num_envs, num_obstacles, 4] = [pos(3), radius]
        self.obstacles = torch.zeros(
            num_envs, self.cfg.num_obstacles, 4,
            device=device
        )  # x, y, z, radius

        # Cube buffers: [num_envs, num_cubes, 3], [num_envs, num_cubes, 3], [num_envs, num_cubes, 3]
        self.p_cubes = torch.zeros(num_envs, self.cfg.num_cubes, 3, device=device)  # position
        self.lwh_cubes = torch.full((num_envs, self.cfg.num_cubes, 3), self.cfg.cube_size, device=device)  # length, width, height
        self.rpy_cubes = torch.zeros(num_envs, self.cfg.num_cubes, 3, device=device)  # roll, pitch, yaw

        # Initialize obstacles
        self._spawn_obstacles()

    def _spawn_obstacles(self):
        """Spawn obstacles based on the configured strategy."""
        strategy = self.cfg.spawn_strategy

        if strategy == "random":
            self._spawn_random()
        elif strategy == "grid":
            self._spawn_grid()
        elif strategy == "fixed":
            self._spawn_fixed()
        else:
            raise ValueError(f"Unknown spawn strategy: {strategy}")

    def _spawn_random(self):
        """Spawn obstacles randomly within bounds."""
        bounds = self.cfg.obstacle_bounds
        num_obs = self.cfg.num_obstacles

        for env_idx in range(self.num_envs):
            for obs_idx in range(num_obs):
                # Random position within bounds
                x = torch.rand(1) * (bounds[3] - bounds[0]) + bounds[0]
                y = torch.rand(1) * (bounds[4] - bounds[1]) + bounds[1]
                z = torch.rand(1) * (bounds[5] - bounds[2]) + bounds[2]

                self.obstacles[env_idx, obs_idx, 0] = x
                self.obstacles[env_idx, obs_idx, 1] = y
                self.obstacles[env_idx, obs_idx, 2] = z
                self.obstacles[env_idx, obs_idx, 3] = self.cfg.obstacle_radius

    def _spawn_grid(self):
        """Spawn obstacles in a grid pattern."""
        bounds = self.cfg.obstacle_bounds
        num_obs = self.cfg.num_obstacles

        # Calculate grid dimensions
        grid_size = int(num_obs ** (1/3)) + 1

        x_range = torch.linspace(bounds[0], bounds[3], grid_size)
        y_range = torch.linspace(bounds[1], bounds[4], grid_size)
        z_range = torch.linspace(bounds[2], bounds[5], grid_size)

        obs_idx = 0
        for env_idx in range(self.num_envs):
            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        if obs_idx >= num_obs:
                            break
                        self.obstacles[env_idx, obs_idx, 0] = x
                        self.obstacles[env_idx, obs_idx, 1] = y
                        self.obstacles[env_idx, obs_idx, 2] = z
                        self.obstacles[env_idx, obs_idx, 3] = self.cfg.obstacle_radius
                        obs_idx += 1
                    obs_idx = 0

    def _spawn_fixed(self):
        """Spawn obstacles at fixed positions."""
        # Fixed obstacle positions (5 obstacles in typical setup)
        fixed_positions = [
            (2.0, 0.0, 2.0),
            (-2.0, 0.0, 3.0),
            (0.0, 2.0, 1.5),
            (0.0, -2.0, 2.5),
            (1.0, 1.0, 3.5),
        ]

        for env_idx in range(self.num_envs):
            for obs_idx, pos in enumerate(fixed_positions):
                if obs_idx >= self.cfg.num_obstacles:
                    break
                self.obstacles[env_idx, obs_idx, 0] = pos[0]
                self.obstacles[env_idx, obs_idx, 1] = pos[1]
                self.obstacles[env_idx, obs_idx, 2] = pos[2]
                self.obstacles[env_idx, obs_idx, 3] = self.cfg.obstacle_radius

    def get_obstacle_positions(self) -> torch.Tensor:
        """Get obstacle positions.

        Returns:
            Tensor of shape [num_envs, num_obstacles, 3].
        """
        return self.obstacles[:, :, :3]

    def get_obstacle_radii(self) -> torch.Tensor:
        """Get obstacle radii.

        Returns:
            Tensor of shape [num_envs, num_obstacles].
        """
        return self.obstacles[:, :, 3]

    def get_cube_positions(self) -> torch.Tensor:
        """Get cube positions.

        Returns:
            Tensor of shape [num_envs, num_cubes, 3].
        """
        return self.p_cubes

    def get_cube_lwh(self) -> torch.Tensor:
        """Get cube dimensions (length, width, height).

        Returns:
            Tensor of shape [num_envs, num_cubes, 3].
        """
        return self.lwh_cubes

    def get_cube_rpy(self) -> torch.Tensor:
        """Get cube rotations (roll, pitch, yaw).

        Returns:
            Tensor of shape [num_envs, num_cubes, 3].
        """
        return self.rpy_cubes

    @property
    def num_cubes(self) -> int:
        """Number of cube obstacles."""
        return self.cfg.num_cubes

    def compute_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute distances from positions to all obstacles.

        Args:
            positions: Positions [num_envs, 3] or [num_envs, num_samples, 3].

        Returns:
            Distances [num_envs, num_obstacles] or [num_envs, num_samples, num_obstacles].
        """
        if positions.dim() == 2:
            # [num_envs, 3] -> [num_envs, 1, 3]
            positions = positions.unsqueeze(1)
        elif positions.dim() == 3:
            # Already [num_envs, num_samples, 3]
            pass
        else:
            raise ValueError(f"Expected 2 or 3D positions, got {positions.dim()}D")

        # Compute Euclidean distances
        obs_pos = self.get_obstacle_positions()  # [num_envs, num_obstacles, 3]
        distances = torch.norm(positions.unsqueeze(1) - obs_pos.unsqueeze(2), dim=-1)

        return distances

    def compute_nearest_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute distance to nearest obstacle for each position.

        Args:
            positions: Positions [num_envs, 3].

        Returns:
            Nearest distances [num_envs].
        """
        distances = self.compute_distances(positions)  # [num_envs, num_obstacles]
        return distances.min(dim=1)[0].squeeze(-1)

    def check_collisions(
        self,
        positions: torch.Tensor,
        radii: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Check for collisions between positions and obstacles.

        Args:
            positions: Positions [num_envs, 3].
            radii: Optional radii for collision [num_envs]. Uses cfg.collision_radius if None.

        Returns:
            Collision flag [num_envs].
        """
        if radii is None:
            radii = torch.full(
                (self.num_envs,),
                self.cfg.collision_radius,
                device=self.device
            )

        # Get combined radii (obstacle + drone)
        obs_radii = self.get_obstacle_radii()  # [num_envs, num_obstacles]
        combined_radii = obs_radii + radii.unsqueeze(1)  # [num_envs, num_obstacles]

        # Compute distances
        distances = self.compute_distances(positions)  # [num_envs, num_obstacles, 1]

        # Squeeze to 2D
        distances = distances.squeeze(-1)  # [num_envs, num_obstacles]

        # Check if any obstacle is within combined radius
        collisions = (distances < combined_radii).any(dim=1)

        return collisions

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset obstacles for specified environments.

        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        if env_ids is None:
            self._spawn_obstacles()
        else:
            # Reset only specified environments
            bounds = self.cfg.obstacle_bounds
            num_obs = self.cfg.num_obstacles

            for env_idx in env_ids:
                for obs_idx in range(num_obs):
                    x = torch.rand(1, device=self.device) * (bounds[3] - bounds[0]) + bounds[0]
                    y = torch.rand(1, device=self.device) * (bounds[4] - bounds[1]) + bounds[1]
                    z = torch.rand(1, device=self.device) * (bounds[5] - bounds[2]) + bounds[2]

                    self.obstacles[env_idx, obs_idx, 0] = x
                    self.obstacles[env_idx, obs_idx, 1] = y
                    self.obstacles[env_idx, obs_idx, 2] = z
                    self.obstacles[env_idx, obs_idx, 3] = self.cfg.obstacle_radius
