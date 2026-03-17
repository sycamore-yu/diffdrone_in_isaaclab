import pytest
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
from diffaero_newton.configs.dynamics_cfg import (
    ContinuousPointMassCfg,
    DiscretePointMassCfg,
    PointMassCfg,
)
from diffaero_newton.dynamics.registry import create_dynamics
import warp as wp

wp.init()

@pytest.mark.parametrize(
    ("cfg_cls", "expected_type"),
    [
        (PointMassCfg, "ContinuousPointMass"),
        (ContinuousPointMassCfg, "ContinuousPointMass"),
        (DiscretePointMassCfg, "DiscretePointMass"),
    ],
)
def test_pointmass_dynamics_forward_and_backward(cfg_cls, expected_type):
    """Test point-mass integration and gradient flow for all supported variants."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 2
    
    cfg = cfg_cls(
        num_envs=num_envs,
        dt=0.01,
        requires_grad=True,
        mass=1.0,
        drag_coeff=0.1,
        n_substeps=2
    )
    
    pm = create_dynamics(cfg, device=device)
    assert type(pm).__name__ == expected_type
    
    # Check initial state
    state = pm.get_state()
    assert state['position'].shape == (num_envs, 3)
    assert state['velocity'].shape == (num_envs, 3)
    
    # Create actions that require gradients
    # PointMass action is 3D thrust vector
    actions = torch.ones(num_envs, 3, device=device, requires_grad=True)
    
    pm.apply_control(actions)
    pm.integrate()
    
    # Get unrolled state
    next_state = pm.get_flat_state()
    
    # Compute dummy loss and backprop
    loss = torch.sum(next_state ** 2)
    loss.backward()
    
    # Verify gradients propagate to input actions
    assert actions.grad is not None
    assert torch.any(actions.grad != 0.0)

    # Detach graph
    pm.detach_graph()
    state_after_detach = pm.get_state()
    assert not state_after_detach["position"].requires_grad


def test_discrete_pointmass_respects_mass_scaling():
    """The discrete point-mass variant should slow down under the same force when mass increases."""
    dt = 0.1
    light = create_dynamics(
        DiscretePointMassCfg(num_envs=1, dt=dt, requires_grad=False, mass=1.0, drag_coeff=0.0),
        device="cpu",
    )
    heavy = create_dynamics(
        DiscretePointMassCfg(num_envs=1, dt=dt, requires_grad=False, mass=2.0, drag_coeff=0.0),
        device="cpu",
    )

    control = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    light.apply_control(control)
    heavy.apply_control(control)

    light.integrate()
    heavy.integrate()

    light_vx = light.get_flat_state()[0, 7].item()
    heavy_vx = heavy.get_flat_state()[0, 7].item()

    assert light_vx == pytest.approx(dt * 1.0, rel=1e-5)
    assert heavy_vx == pytest.approx(dt * 0.5, rel=1e-5)
    assert heavy_vx < light_vx

if __name__ == "__main__":
    test_pointmass_dynamics_forward_and_backward()
    print("PointMass integration and gradient test passed.")
