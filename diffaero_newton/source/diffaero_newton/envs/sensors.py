import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn.functional as F
from torch import Tensor
from diffaero_newton.tasks.obstacle_manager import ObstacleManager

def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate a vector by a quaternion."""
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a + b + c

def quat_rotate_inverse(q: Tensor, v: Tensor) -> Tensor:
    """Rotate a vector by the inverse of a quaternion."""
    q_inv = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return quat_rotate(q_inv, v)

def quaternion_apply(quaternion: Tensor, point: Tensor) -> Tensor:
    """Apply quaternion rotation to a point."""
    return quat_rotate(quaternion, point)

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    q = torch.zeros((*roll.shape, 4), device=roll.device)
    q[..., 0] = cr * cp * cy + sr * sp * sy
    q[..., 1] = sr * cp * cy - cr * sp * sy
    q[..., 2] = cr * sp * cy + sr * cp * sy
    q[..., 3] = cr * cp * sy - sr * sp * cy
    return q


@torch.jit.script
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
        
        # All obstacles in current diffaero_newton.tasks.ObstacleManager are just spherical
        # We will use raydist3d_sphere for all of them
        n_spheres = getattr(obstacle_manager, "cfg", None)
        n_spheres = n_spheres.num_obstacles if n_spheres else getattr(obstacle_manager, "num_obstacles", 0)
        
        sphere_ray_dists = torch.full( # [num_envs, n_obstacles, n_rays]
            (pos.shape[0], n_spheres, self.H*self.W),
            fill_value=self.max_dist, dtype=torch.float, device=self.device)
        cube_ray_dists = torch.empty((pos.shape[0], 0, self.H*self.W), dtype=torch.float, device=self.device)
            
        distances = obstacle_manager.compute_distances(pos)  # may be 3D
        while distances.dim() > 2:
            distances = distances.squeeze(-1)
        env_ids, obstacle_ids = torch.where(distances.le(self.max_dist))
        
        sphere_env_ids, sphere_ids = env_ids, obstacle_ids
        cube_env_ids, cube_ids = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        p_spheres = obstacle_manager.get_obstacle_positions() # [num_envs, n_spheres, 3]
        r_spheres = obstacle_manager.get_obstacle_radii()     # [num_envs, n_spheres]
        
        depth, _ = get_ray_dist(
            sphere_ray_dists=sphere_ray_dists,
            sphere_env_ids=sphere_env_ids,
            sphere_ids=sphere_ids,
            p_spheres=p_spheres, 
            r_spheres=r_spheres, 
            
            cube_ray_dists=cube_ray_dists,
            cube_env_ids=cube_env_ids,
            cube_ids=cube_ids,
            p_cubes=torch.empty((pos.shape[0], 0, 3), device=self.device), # Dummy
            lwh_cubes=torch.empty((pos.shape[0], 0, 3), device=self.device), # Dummy
            rpy_cubes=torch.empty((pos.shape[0], 0, 3), device=self.device), # Dummy

            
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


def create_sensor(cfg, num_envs: int, device: torch.device):
    if cfg is None:
        return None
        
    sensor_alias = {
        "camera": CameraSensor,
        "lidar": LiDARSensor,
        "relpos": RelativePositionSensor,
    }
    return sensor_alias[cfg.name](cfg, num_envs, device)
