import torch
from dataclasses import dataclass
from diffaero_newton.common.constants import DEFAULT_DT

@dataclass
class DynamicsCfg:
    """Base config for dynamics."""
    num_envs: int = 1
    dt: float = DEFAULT_DT
    requires_grad: bool = False
    model_type: str = "quadrotor"


@dataclass
class QuadrotorCfg(DynamicsCfg):
    """Config for Quadrotor dynamics."""
    model_type: str = "quadrotor"
    arm_length: float = 0.04
    mass: float = 0.027
    inertia: tuple[float, float, float] = (1.4e-5, 1.4e-5, 2.17e-5)
    solver_type: str = "semi_implicit"
    n_substeps: int = 1


@dataclass
class PointMassCfg(DynamicsCfg):
    """Config for PointMass dynamics."""
    model_type: str = "pointmass"
    mass: float = 1.0
    drag_coeff: float = 0.1
    solver_type: str = "semi_implicit"
    n_substeps: int = 1

