import torch
from dataclasses import dataclass
from diffaero_newton.common.constants import DEFAULT_DT


POINTMASS_MODEL_TYPES = ("pointmass", "continuous_pointmass", "discrete_pointmass")


def is_pointmass_model_type(model_type: str) -> bool:
    """Return whether a model type uses the point-mass interface."""

    return model_type in POINTMASS_MODEL_TYPES

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
    """Backward-compatible alias for the continuous point-mass model."""
    model_type: str = "pointmass"
    mass: float = 1.0
    drag_coeff: float = 0.1
    solver_type: str = "semi_implicit"
    n_substeps: int = 1


@dataclass
class ContinuousPointMassCfg(DynamicsCfg):
    """Config for continuous point-mass dynamics."""

    model_type: str = "continuous_pointmass"
    mass: float = 1.0
    drag_coeff: float = 0.1
    solver_type: str = "semi_implicit"
    n_substeps: int = 1


@dataclass
class DiscretePointMassCfg(DynamicsCfg):
    """Config for discrete point-mass dynamics."""

    model_type: str = "discrete_pointmass"
    mass: float = 1.0
    drag_coeff: float = 0.1
    solver_type: str = "semi_implicit"
    n_substeps: int = 1
    action_frame: str = "world"
