import torch
from diffaero_newton.dynamics.drone_dynamics import create_drone
drone = create_drone(num_envs=1, device="cpu")
thrust = torch.ones(1, 4) * 0.5
drone.apply_control(thrust)
drone.integrate(dt=0.01)
print("Success!")
