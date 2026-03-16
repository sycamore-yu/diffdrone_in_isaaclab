from diffaero_newton.configs.dynamics_cfg import DynamicsCfg, QuadrotorCfg, PointMassCfg

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
            solver_type=getattr(cfg, "solver_type", "semi_implicit"),
            n_substeps=getattr(cfg, "n_substeps", 1),
        )
        return Drone(drone_cfg, device=device)
        
    elif cfg.model_type == "pointmass":
        from diffaero_newton.dynamics.pointmass_dynamics import PointMass, PointMassConfig
        
        pm_cfg = PointMassConfig(
            num_envs=cfg.num_envs,
            dt=cfg.dt,
            requires_grad=cfg.requires_grad,
            mass=getattr(cfg, "mass", 1.0),
            drag_coeff=getattr(cfg, "drag_coeff", 0.1),
            solver_type=getattr(cfg, "solver_type", "semi_implicit"),
            n_substeps=getattr(cfg, "n_substeps", 1),
        )
        return PointMass(pm_cfg, device=device)
        
    else:
        raise ValueError(f"Unknown dynamics model type: {cfg.model_type}")
