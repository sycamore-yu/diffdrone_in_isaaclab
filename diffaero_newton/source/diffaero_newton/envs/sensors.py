import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from diffaero_newton.tasks.obstacle_manager import ObstacleManager

def quat_rotate(q_xyzw: Tensor, v: Tensor) -> Tensor:
    """Rotate a vector by a quaternion [qx, qy, qz, qw]."""
    qw = q_xyzw[..., 3]
    q_vec = q_xyzw[..., :3]
    a = v * (2.0 * qw ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * qw.unsqueeze(-1) * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a + b + c

def quat_rotate_inverse(q_xyzw: Tensor, v: Tensor) -> Tensor:
    """Rotate a vector by the inverse of a quaternion [qx, qy, qz, qw]."""
    q_inv = torch.cat([-q_xyzw[..., :3], q_xyzw[..., 3:]], dim=-1)
    return quat_rotate(q_inv, v)

def quaternion_apply(quaternion: Tensor, point: Tensor) -> Tensor:
    """Apply quaternion rotation to a point."""
    return quat_rotate(quaternion, point)

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion [qx, qy, qz, qw]."""
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = cy * sr * cp - sy * cr * sp
    qy = cr * sp * cy + sr * cp * sy
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two quaternions [qx, qy, qz, qw]."""
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    qx = x1 * (qq + xx) + y1 * zz - z1 * yy
    qy = y1 * (qq + yy) + z1 * xx - x1 * zz
    qz = z1 * (qq + zz) + x1 * yy - y1 * xx
    qw = qq - xx - yy - zz

    return torch.stack([qx, qy, qz, qw], dim=-1).view(a.shape)


def euler_angles_to_matrix(euler_angles: Tensor, convention: str) -> Tensor:
    """Convert Euler angles to rotation matrix. Simplified version of PyTorch3D's."""
    if convention == "XYZ":
        x, y, z = euler_angles.unbind(-1)
        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)
        
        rx_r0 = torch.stack([torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
        rx_r1 = torch.stack([torch.zeros_like(x), cx, -sx], dim=-1)
        rx_r2 = torch.stack([torch.zeros_like(x), sx, cx], dim=-1)
        rx = torch.stack([rx_r0, rx_r1, rx_r2], dim=-2)
        
        ry_r0 = torch.stack([cy, torch.zeros_like(y), sy], dim=-1)
        ry_r1 = torch.stack([torch.zeros_like(y), torch.ones_like(y), torch.zeros_like(y)], dim=-1)
        ry_r2 = torch.stack([-sy, torch.zeros_like(y), cy], dim=-1)
        ry = torch.stack([ry_r0, ry_r1, ry_r2], dim=-2)
        
        rz_r0 = torch.stack([cz, -sz, torch.zeros_like(z)], dim=-1)
        rz_r1 = torch.stack([sz, cz, torch.zeros_like(z)], dim=-1)
        rz_r2 = torch.stack([torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z)], dim=-1)
        rz = torch.stack([rz_r0, rz_r1, rz_r2], dim=-2)
        
        return rx @ ry @ rz
    raise NotImplementedError("Convention not implemented")

@torch.jit.script
def raydist3d_sphere(
    obst_pos: Tensor, # [m_spheres, 3]
    obst_r: Tensor, # [m_spheres]
    start: Tensor, # [m_spheres, n_rays, 3]
    direction: Tensor, # [m_spheres, n_rays, 3]
    max_dist: float
) -> Tensor:
    rel_pos = obst_pos.unsqueeze(1) - start # [m_spheres, n_rays, 3]
    rel_dist = torch.norm(rel_pos, dim=-1) # [m_spheres, n_rays]
    costheta = torch.cosine_similarity(rel_pos, direction, dim=-1) # [m_spheres, n_rays]
    sintheta = torch.where(costheta>0, torch.sqrt(1 - costheta**2), torch.tensor(0.9, device=costheta.device)) # [m_spheres, n_rays]
    dist_center2ray = rel_dist * sintheta # [m_spheres, n_rays]
    obst_r = obst_r.unsqueeze(1) # [m_spheres, 1]
    raydist = rel_dist * costheta - torch.sqrt(torch.clamp(torch.pow(obst_r, 2) - torch.pow(dist_center2ray, 2), min=0.0)) # [m_spheres, n_rays]
    valid = torch.logical_and(dist_center2ray < obst_r, costheta > 0) # [m_spheres, n_rays]
    raydist_valid = torch.where(valid, raydist, max_dist) # [m_spheres, n_rays]
    return raydist_valid

@torch.jit.script
def raydist3d_cube(
    p_cubes: Tensor, # [m_cubes, 3]
    lwh_cubes: Tensor, # [m_cubes, 3]
    rpy_cubes: Tensor, # [m_cubes, 3]
    start: Tensor, # [m_cubes, n_rays, 3]
    direction: Tensor, # [m_cubes, n_rays, 3]
    max_dist: float
) -> Tensor:
    if not torch.all(rpy_cubes == 0):
        rotmat = euler_angles_to_matrix(rpy_cubes, convention='XYZ').transpose(-1, -2) # [m_cubes, 3, 3]
        start = (rotmat.unsqueeze(1) @ (start - p_cubes.unsqueeze(1)).unsqueeze(-1)).squeeze(-1) # [m_cubes, n_rays, 3]
        direction = (rotmat.unsqueeze(1) @ direction.unsqueeze(-1)).squeeze(-1) # [m_cubes, n_rays, 3]
        box_min = -lwh_cubes / 2. # [m_cubes, 3]
        box_max =  lwh_cubes / 2. # [m_cubes, 3]
    else: 
        box_min = (p_cubes - lwh_cubes / 2.) # [m_cubes, 3]
        box_max = (p_cubes + lwh_cubes / 2.) # [m_cubes, 3]
    
    _tmin = (box_min.unsqueeze(1) - start) / direction # [m_cubes, n_rays, 3]
    _tmax = (box_max.unsqueeze(1) - start) / direction # [m_cubes, n_rays, 3]
    tmin = torch.where(direction < 0, _tmax, _tmin) # [m_cubes, n_rays, 3]
    tmax = torch.where(direction < 0, _tmin, _tmax) # [m_cubes, n_rays, 3]
    tentry = torch.max(tmin, dim=-1).values # [m_cubes, n_rays]
    texit = torch.min(tmax, dim=-1).values # [m_cubes, n_rays]
    valid = torch.logical_and(tentry <= texit, texit >= 0) # [m_cubes, n_rays]
    raydist = torch.where(valid, tentry, max_dist) # [m_cubes, n_rays]
    return raydist

@torch.jit.script
def raydist3d_ground_plane(
    z_ground_plane: Tensor, # [n_envs]
    start: Tensor, # [n_envs, n_rays, 3]
    direction: Tensor, # [n_envs, n_rays, 3]
    max_dist: float
) -> Tensor:
    z_ground_plane = z_ground_plane.unsqueeze(-1) # [n_envs, 1]
    valid = (start[..., 2] - z_ground_plane) * direction[..., 2] < 0 # [n_envs, n_rays]
    raydist = torch.where(valid, (z_ground_plane - start[..., 2]) / (direction[..., 2] + 1e-6), max_dist) # [n_envs, n_rays]
    return raydist

@torch.jit.script
def ray_directions_body2world(
    ray_directions: Tensor,
    quat_xyzw: Tensor,
    H: int,
    W: int
) -> Tensor: # [n_envs, n_rays, 3]
    quat_wxyz = quat_xyzw.roll(1, dims=-1) # [n_envs, 4]
    quat_wxyz = quat_wxyz.unsqueeze(1).expand(-1, H*W, -1) # [n_envs, n_rays, 4]
    return quaternion_apply(quat_wxyz, ray_directions.view(quat_wxyz.size(0), H*W, 3)) # [n_envs, n_rays, 3]

@torch.jit.script
def get_ray_dist(
    sphere_ray_dists: Tensor, # [n_envs, n_spheres, n_rays]
    sphere_env_ids: Tensor,   # [m_spheres]
    sphere_ids: Tensor,       # [m_spheres]
    p_spheres: Tensor,        # [n_envs, n_spheres, 3]
    r_spheres: Tensor,        # [n_envs, n_spheres]
    cube_ray_dists: Tensor,   # [n_envs, n_cubes, n_rays]
    cube_env_ids: Tensor,     # [m_cubes]
    cube_ids: Tensor,         # [m_cubes]
    p_cubes: Tensor,          # [n_envs, n_cubes, 3]
    lwh_cubes: Tensor,        # [n_envs, n_cubes, 3]
    rpy_cubes: Tensor,        # [n_envs, n_cubes, 3]
    start: Tensor,            # [n_envs, n_rays, 3]
    ray_directions_b: Tensor, # [n_envs, n_rays, 3]
    quat_xyzw: Tensor,        # [n_envs, 4]
    max_dist: float,
    H: int,
    W: int,
    z_ground_plane: Optional[Tensor] = None, # [n_envs]
) -> Tuple[Tensor, Tensor]: # [n_envs, H, W], [n_envs, n_rays, 3]
    ray_directions_w = ray_directions_body2world(ray_directions_b, quat_xyzw, H, W) # [n_envs, n_rays, 3]
    
    n_spheres = p_spheres.shape[1]
    if n_spheres > 0 and len(sphere_env_ids) > 0:
        sphere_ray_starts = start[sphere_env_ids] # [m_spheres, n_rays, 3]
        sphere_ray_directions_w = ray_directions_w[sphere_env_ids] # [m_spheres, n_rays, 3]
        p_spheres_subset = p_spheres[sphere_env_ids, sphere_ids] # [m_spheres, 3]
        r_spheres_subset = r_spheres[sphere_env_ids, sphere_ids] # [m_spheres]
        raydist_sphere = raydist3d_sphere(p_spheres_subset, r_spheres_subset, sphere_ray_starts, sphere_ray_directions_w, max_dist)
        sphere_ray_dists[sphere_env_ids, sphere_ids] = raydist_sphere
    
    n_cubes = p_cubes.shape[1]
    if n_cubes > 0 and len(cube_env_ids) > 0:
        cube_ray_starts = start[cube_env_ids] # [m_cubes, n_rays, 3]
        cube_ray_directions_w = ray_directions_w[cube_env_ids] # [m_cubes, n_rays, 3]
        p_cubes_subset = p_cubes[cube_env_ids, cube_ids] # [m_cubes, 3]
        lwh_cubes_subset = lwh_cubes[cube_env_ids, cube_ids] # [m_cubes, 3]
        rpy_cubes_subset = rpy_cubes[cube_env_ids, cube_ids] # [m_cubes, 3]
        raydist_cube = raydist3d_cube(p_cubes_subset, lwh_cubes_subset, rpy_cubes_subset, cube_ray_starts, cube_ray_directions_w, max_dist)
        cube_ray_dists[cube_env_ids, cube_ids] = raydist_cube
    
    raydist = torch.cat([sphere_ray_dists, cube_ray_dists], dim=1).min(dim=1).values # [n_envs, n_rays]
    if z_ground_plane is not None:
        raydist_ground_plane: Tensor = raydist3d_ground_plane(z_ground_plane, start, ray_directions_w, max_dist) # [n_envs, n_rays]
        raydist = torch.minimum(raydist, raydist_ground_plane) # [n_envs, n_rays]
    raydist = raydist.clamp(max=max_dist)
    contact_points = ray_directions_w * raydist.unsqueeze(-1) + start # [n_envs, n_rays, 3]
    depth = 1. - raydist.reshape(-1, H, W) / max_dist # [n_envs, H, W]
    return depth, contact_points # [n_envs, H, W], [n_envs, n_rays, 3]


class RayCastingSensorBase:
    def __init__(self, cfg, num_envs: int, device: torch.device):
        self.H: int
        self.W: int
        self.num_envs: int = num_envs
        self.max_dist: float = cfg.max_dist
        self.device = device
        self.ray_directions: Tensor # [num_envs, H, W, 3]
        
    def sensor2body(self, vec_s: Tensor):
        # Assuming no sensor rotation from body for now
        return vec_s.reshape(self.num_envs, self.H*self.W, 3)
    
    def body2sensor(self, vec_b: Tensor):
        return vec_b.reshape(self.num_envs, self.H*self.W, 3)

    def __call__(
        self,
        obstacle_manager: ObstacleManager,
        pos: Tensor, # [num_envs, 3]
        quat_xyzw: Tensor, # [num_envs, 4]
        z_ground_plane: Optional[Tensor] = None
    ) -> Tensor: # [num_envs, H, W]
        ray_starts = pos.unsqueeze(1).expand(-1, self.H * self.W, -1) # [num_envs, n_rays, 3]

        n_spheres = getattr(obstacle_manager, "cfg", None)
        n_spheres = n_spheres.num_obstacles if n_spheres else getattr(obstacle_manager, "num_obstacles", 0)
        n_cubes = getattr(obstacle_manager, "num_cubes", 0)

        sphere_ray_dists = torch.full(
            (pos.shape[0], n_spheres, self.H*self.W),
            fill_value=self.max_dist, dtype=torch.float, device=self.device)
        cube_ray_dists = torch.full(
            (pos.shape[0], n_cubes, self.H*self.W),
            fill_value=self.max_dist, dtype=torch.float, device=self.device)

        distances = obstacle_manager.compute_distances(pos)  # [num_envs, n_spheres+n_cubes, 1]
        while distances.dim() > 2:
            distances = distances.squeeze(-1)
        env_ids, obstacle_ids = torch.where(distances.le(self.max_dist))

        sphere_env_ids = env_ids
        sphere_ids = obstacle_ids
        cube_env_ids = torch.empty(0, dtype=torch.long)
        cube_ids = torch.empty(0, dtype=torch.long)

        p_spheres = obstacle_manager.get_obstacle_positions() # [num_envs, n_spheres, 3]
        r_spheres = obstacle_manager.get_obstacle_radii()     # [num_envs, n_spheres]
        p_cubes = obstacle_manager.get_cube_positions()
        lwh_cubes = obstacle_manager.get_cube_lwh()
        rpy_cubes = obstacle_manager.get_cube_rpy()

        depth, _ = get_ray_dist(
            sphere_ray_dists=sphere_ray_dists,
            sphere_env_ids=sphere_env_ids,
            sphere_ids=sphere_ids,
            p_spheres=p_spheres,
            r_spheres=r_spheres,

            cube_ray_dists=cube_ray_dists,
            cube_env_ids=cube_env_ids,
            cube_ids=cube_ids,
            p_cubes=p_cubes,
            lwh_cubes=lwh_cubes,
            rpy_cubes=rpy_cubes,

            start=ray_starts,
            ray_directions_b=self.sensor2body(self.ray_directions), # [num_envs, n_rays, 3]
            quat_xyzw=quat_xyzw,
            max_dist=self.max_dist,
            H=self.H,
            W=self.W,
            z_ground_plane=z_ground_plane
        )
        return depth


class CameraSensor(RayCastingSensorBase):
    def __init__(self, cfg, num_envs: int, device: torch.device):
        super().__init__(cfg, num_envs, device)
        self.H: int = cfg.height
        self.W: int = cfg.width
        self.hfov: float = cfg.horizontal_fov
        self.vfov: float = self.hfov * self.H / self.W
        self.ray_directions = F.normalize(self._get_ray_directions_plane(), dim=-1) # [H, W, 3]
        self.ray_directions = self.ray_directions.unsqueeze(0).expand(self.num_envs, -1, -1, -1)

    def _get_ray_directions_plane(self):
        forward = torch.tensor([[[1., 0., 0.]]], device=self.device).expand(self.H, self.W, -1) # [H, W, 3]
        
        vangle = 0.5 * self.vfov * torch.pi / 180
        vertical_offset = torch.linspace(math.tan(vangle), -math.tan(vangle), self.H, device=self.device).reshape(-1, 1, 1) # [H, 1, 1]
        zero = torch.zeros_like(vertical_offset)
        vertical_offset = torch.cat([zero, zero, vertical_offset], dim=-1) # [H, 1, 3]
        
        hangle = 0.5 * self.hfov * torch.pi / 180
        horizontal_offset = torch.linspace(math.tan(hangle), -math.tan(hangle), self.W, device=self.device).reshape(1, -1, 1) # [1, W, 1]
        zero = torch.zeros_like(horizontal_offset)
        horizontal_offset = torch.cat([zero, horizontal_offset, zero], dim=-1) # [1, W, 3]
        
        return forward + vertical_offset + horizontal_offset # [H, W, 3]


class LiDARSensor(RayCastingSensorBase):
    def __init__(self, cfg, num_envs: int, device: torch.device):
        super().__init__(cfg, num_envs, device)
        self.H: int = cfg.n_rays_vertical
        self.W: int = cfg.n_rays_horizontal
        self.dep_angle_rad: float = cfg.depression_angle * torch.pi / 180
        self.ele_angle_rad: float = cfg.elevation_angle * torch.pi / 180
        self.ray_directions = F.normalize(self._get_ray_directions(), dim=-1) # [H, W, 3]
        self.ray_directions = self.ray_directions.unsqueeze(0).expand(self.num_envs, -1, -1, -1)
    
    def _get_ray_directions(self):
        forward = torch.tensor([[[1., 0., 0.]]], device=self.device).expand(self.H, self.W, -1) # [H, W, 3]
        
        yaw = torch.arange(0, self.W, device=self.device) / self.W * 2 * torch.pi
        pitch = torch.linspace(self.ele_angle_rad, self.dep_angle_rad, self.H, device=self.device)
        pitch, yaw = torch.meshgrid(pitch, yaw, indexing="ij")
        roll = torch.zeros_like(pitch)
        rpy = torch.stack([roll, pitch, yaw], dim=-1)
        rotmat = euler_angles_to_matrix(rpy, convention='XYZ') # [H, W, 3, 3]
        directions = rotmat.transpose(-1, -2) @ forward.unsqueeze(-1) # [H, W, 3, 1]
        return directions.squeeze(-1) # [H, W, 3]


class RelativePositionSensor:
    def __init__(self, cfg, num_envs: int, device: torch.device):
        self.H: int = cfg.n_obstacles + int(cfg.ceiling) + 4 * int(cfg.walls)
        self.W: int = 3
        self.device = device
    
    def __call__(
        self,
        obstacle_manager: ObstacleManager,
        pos: Tensor, # [num_envs, 3]
        quat_xyzw: Tensor, # [num_envs, 4]
        z_ground_plane: Optional[float] = None
    ) -> Tensor: # [num_envs, H, W]
        p_spheres = obstacle_manager.get_obstacle_positions() # [num_envs, n_obstacles, 3]
        # Calculate nearest points simply as points along the vector from pos to sphere center
        dist_vec = p_spheres - pos.unsqueeze(1)
        dist2obstacles = torch.norm(dist_vec, dim=-1)
        r_spheres = obstacle_manager.get_obstacle_radii()
        dist2obstacles = torch.clamp(dist2obstacles - r_spheres, min=0.0)
        
        nearest_points2obstacles = pos.unsqueeze(1) + F.normalize(dist_vec, dim=-1) * dist2obstacles.unsqueeze(-1)
        obst_relpos = nearest_points2obstacles - pos.unsqueeze(1)
        sorted_idx = dist2obstacles.argsort(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
        sorted_obst_relpos = obst_relpos.gather(dim=1, index=sorted_idx) # [num_envs, n_obstacles, 3]
        
        # We need to pad with walls/ceiling if needed to match H
        pad_size = self.H - sorted_obst_relpos.shape[1]
        if pad_size > 0:
            padding = torch.zeros(pos.shape[0], pad_size, 3, device=self.device)
            sorted_obst_relpos = torch.cat([sorted_obst_relpos, padding], dim=1)
            
        return sorted_obst_relpos


class IMUSensor:
    """IMU sensor with complete noise and drift model.

    Models:
    - Accelerometer: drift (bias random walk) + noise (white noise)
    - Gyroscope: drift (bias random walk) + noise (white noise)
    - Pose drift: integrated gyro error
    - Mounting error: small rotation offset
    """
    def __init__(self, cfg, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.dt = 0.01  # Will be overridden if provided
        self.sqrt_dt = 0.1

        # Drift parameters (per sqrt timestep)
        self._acc_drift_base = cfg.acc_drift_std
        self._gyro_drift_base = cfg.gyro_drift_std
        self._acc_noise_base = cfg.acc_noise_std
        self._gyro_noise_base = cfg.gyro_noise_std
        self.acc_drift_std = self._acc_drift_base * self.sqrt_dt
        self.gyro_drift_std = self._gyro_drift_base * self.sqrt_dt
        # Noise parameters (per timestep / sqrt timestep)
        self.acc_noise_std = self._acc_noise_base / self.sqrt_dt
        self.gyro_noise_std = self._gyro_noise_base / self.sqrt_dt

        # Mounting error range
        self.mounting_range_rad = cfg.imu_mounting_error_range_deg * math.pi / 180.0

        factory_kwargs = {"device": device, "dtype": torch.float32}

        # State: drift (bias random walk states)
        self.acc_drift_b = torch.zeros((num_envs, 3), **factory_kwargs)  # accelerometer bias drift
        self.gyro_drift_b = torch.zeros((num_envs, 3), **factory_kwargs)  # gyro bias drift
        self.pose_drift_b = torch.zeros((num_envs, 3), **factory_kwargs)  # integrated pose drift from gyro

        # State: noise (white noise, uncorrelated each step)
        self.acc_noise_b = torch.zeros((num_envs, 3), **factory_kwargs)
        self.gyro_noise_b = torch.zeros((num_envs, 3), **factory_kwargs)

        # Mounting quaternion (small random rotation)
        self.mounting_quat_xyzw = torch.zeros((num_envs, 4), **factory_kwargs)

        self.enable_drift = int(cfg.enable_drift)
        self.enable_noise = int(cfg.enable_noise)

    def set_dt(self, dt: float):
        """Set simulation timestep after initialization."""
        self.dt = dt
        self.sqrt_dt = math.sqrt(dt)
        # Recalculate scaled stds from base cfg values
        self.acc_drift_std = self._acc_drift_base * self.sqrt_dt
        self.gyro_drift_std = self._gyro_drift_base * self.sqrt_dt
        self.acc_noise_std = self._acc_noise_base / self.sqrt_dt
        self.gyro_noise_std = self._gyro_noise_base / self.sqrt_dt

    def sensor2body(self, vec_s: Tensor) -> Tensor:
        """Transform vector from sensor frame to body frame."""
        return quat_rotate(self.mounting_quat_xyzw, vec_s)

    def body2sensor(self, vec_b: Tensor) -> Tensor:
        """Transform vector from body frame to sensor frame."""
        return quat_rotate_inverse(self.mounting_quat_xyzw, vec_b)

    def reset_idx(self, env_idx: Tensor):
        """Reset IMU state for specified environments."""
        n_selected = env_idx.shape[0]
        # Sample new mounting error
        mounting_euler = (torch.rand(n_selected, 3, device=self.device) * 2 - 1) * self.mounting_range_rad
        self.mounting_quat_xyzw[env_idx] = euler_to_quaternion(
            mounting_euler[..., 0], mounting_euler[..., 1], mounting_euler[..., 2]
        )
        # Reset all drift and noise states
        self.acc_drift_b[env_idx] = torch.zeros(n_selected, 3, device=self.device, dtype=torch.float)
        self.gyro_drift_b[env_idx] = torch.zeros(n_selected, 3, device=self.device, dtype=torch.float)
        self.pose_drift_b[env_idx] = torch.zeros(n_selected, 3, device=self.device, dtype=torch.float)
        self.acc_noise_b[env_idx] = torch.zeros(n_selected, 3, device=self.device, dtype=torch.float)
        self.gyro_noise_b[env_idx] = torch.zeros(n_selected, 3, device=self.device, dtype=torch.float)

    def step(self):
        """Update drift and noise states for one timestep."""
        # Gyro drift random walk + noise
        self.gyro_drift_b += self.sensor2body(torch.randn_like(self.gyro_drift_b) * self.gyro_drift_std)
        self.gyro_noise_b = self.sensor2body(torch.randn_like(self.gyro_noise_b) * self.gyro_noise_std)

        # Pose drift = integrated gyro error
        gyro_error = self.enable_drift * self.gyro_drift_b + self.enable_noise * self.gyro_noise_b
        self.pose_drift_b += gyro_error * self.dt

        # Accelerometer drift random walk + noise
        self.acc_drift_b += self.sensor2body(torch.randn_like(self.acc_drift_b) * self.acc_drift_std)
        self.acc_noise_b = self.sensor2body(torch.randn_like(self.acc_noise_b) * self.acc_noise_std)

    def __call__(
        self,
        pos_w: Tensor,       # [num_envs, 3] true position in world
        quat_xyzw: Tensor,   # [num_envs, 4] true quaternion
        vel_w: Tensor,       # [num_envs, 3] true velocity in world
        acc_w: Tensor,       # [num_envs, 3] true acceleration in world
    ) -> dict:
        """Compute IMU measurements from true dynamics state.

        Returns:
            dict with keys:
                'acc_b': accelerometer reading in body frame [num_envs, 3]
                'gyro_b': gyroscope reading in body frame [num_envs, 3]
                'pos_w': position in world frame (with drift) [num_envs, 3]
                'quat': quaternion (with pose drift) [num_envs, 4]
        """
        # True acceleration in body frame
        acc_b_true = quat_rotate_inverse(quat_xyzw, acc_w)

        # Accelerometer: true + drift (bias random walk) + noise (white)
        acc_b_measured = (
            acc_b_true
            + self.enable_drift * self.acc_drift_b
            + self.enable_noise * self.acc_noise_b
        )

        # Gyroscope: assume no true angular rate input, just drift + noise
        # In real IMU this would include body rotation, but here we model sensor noise only
        gyro_b_measured = (
            self.enable_drift * self.gyro_drift_b
            + self.enable_noise * self.gyro_noise_b
        )

        # Position with drift
        pos_w_measured = pos_w + self.pose_drift_b

        # Quaternion with pose drift applied
        pose_drift_euler = self.pose_drift_b
        pose_drift_quat = euler_to_quaternion(pose_drift_euler[..., 0], pose_drift_euler[..., 1], pose_drift_euler[..., 2])
        quat_measured = quat_mul(quat_xyzw, pose_drift_quat)

        return {
            "acc_b": acc_b_measured,
            "gyro_b": gyro_b_measured,
            "pos_w": pos_w_measured,
            "quat": quat_measured,
        }


def create_sensor(cfg, num_envs: int, device: torch.device):
    if cfg is None:
        return None

    sensor_alias = {
        "camera": CameraSensor,
        "lidar": LiDARSensor,
        "relpos": RelativePositionSensor,
        "imu": IMUSensor,
    }
    return sensor_alias[cfg.name](cfg, num_envs, device)
