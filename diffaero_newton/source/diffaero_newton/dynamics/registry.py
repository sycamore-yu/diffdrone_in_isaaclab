from diffaero_newton.configs.dynamics_cfg import (
    ContinuousPointMassCfg,
    DiscretePointMassCfg,
    DynamicsCfg,
    PointMassCfg,
    QuadrotorCfg,
)

def create_dynamics(cfg: DynamicsCfg, device: str = "cpu"):
    """Factory to create the dynamics model based on configuration."""
    if cfg.model_type == "quadrotor":
        from diffaero_newton.dynamics.drone_dynamics import Drone, DroneConfig
        
        # Cast our generic QuadrotorCfg config into DroneConfig
        drone_cfg = DroneConfig(
            num_envs=cfg.num_envs,
            dt=cfg.dt,
            requires_grad=cfg.requires_grad,
            arm_length=getattr(cfg, "arm_length", 0.04),
            mass=getattr(cfg, "mass", 0.027),
            inertia=getattr(cfg, "inertia", (1.4e-5, 1.4e-5, 2.17e-5)),
            control_mode=getattr(cfg, "control_mode", "motor_thrust"),
            torque_coeff=getattr(cfg, "torque_coeff", 0.01),
            max_thrust=getattr(cfg, "max_thrust", 20.0),
            drag_coeff_xy=getattr(cfg, "drag_coeff_xy", 0.0),
            drag_coeff_z=getattr(cfg, "drag_coeff_z", 0.0),
            k_angvel=getattr(cfg, "k_angvel", (6.0, 6.0, 2.5)),
            max_body_rates=getattr(cfg, "max_body_rates", (3.14, 3.14, 3.14)),
            solver_type=getattr(cfg, "solver_type", "semi_implicit"),
            n_substeps=getattr(cfg, "n_substeps", 1),

        )
        return Drone(drone_cfg, device=device)
        
    elif cfg.model_type in ("pointmass", "continuous_pointmass"):
        from diffaero_newton.dynamics.pointmass_dynamics import ContinuousPointMass, PointMassConfig

        pm_cfg = PointMassConfig(
            num_envs=cfg.num_envs,
            dt=cfg.dt,
            requires_grad=cfg.requires_grad,
            mass=getattr(cfg, "mass", 1.0),
            drag_coeff=getattr(cfg, "drag_coeff", 0.1),
            max_acc_xy=getattr(cfg, "max_acc_xy", 20.0),
            max_acc_z=getattr(cfg, "max_acc_z", 40.0),
            solver_type=getattr(cfg, "solver_type", "semi_implicit"),
            n_substeps=getattr(cfg, "n_substeps", 1),

        )
        return ContinuousPointMass(pm_cfg, device=device)

    elif cfg.model_type == "discrete_pointmass":
        from diffaero_newton.dynamics.pointmass_dynamics import DiscretePointMass, DiscretePointMassConfig

        pm_cfg = DiscretePointMassConfig(
            num_envs=cfg.num_envs,
            dt=cfg.dt,
            requires_grad=cfg.requires_grad,
            mass=getattr(cfg, "mass", 1.0),
            drag_coeff=getattr(cfg, "drag_coeff", 0.1),
            max_acc_xy=getattr(cfg, "max_acc_xy", 20.0),
            max_acc_z=getattr(cfg, "max_acc_z", 40.0),
            solver_type=getattr(cfg, "solver_type", "semi_implicit"),
            n_substeps=getattr(cfg, "n_substeps", 1),

        )
        return DiscretePointMass(pm_cfg, device=device)

    else:
        raise ValueError(f"Unknown dynamics model type: {cfg.model_type}")
