import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
import numpy as np
import torch

from gravitas.ai.pinn import ReactionDiffusionPINN, physics_loss
from gravitas.solver.reaction_diffusion import solve_steady_state_pinn


def test_pinn_architecture_and_forward_pass():
    pinn = ReactionDiffusionPINN(hidden_layers=2, hidden_neurons=16)
    R = torch.linspace(0, 1, 10).unsqueeze(1)
    alpha = torch.full_like(R, 1.0)
    K = torch.full_like(R, 0.5)
    H = torch.full_like(R, 2.0)
    
    out = pinn(R, alpha, K, H)
    assert out.shape == (10, 1)
    # Output is bounded between 0 and 1 due to Sigmoid
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_pinn_physics_loss_gradient_flow():
    pinn = ReactionDiffusionPINN(hidden_layers=2, hidden_neurons=16)
    R = torch.rand(15, 1, requires_grad=True)
    alpha = torch.full_like(R, 1.5)
    K = torch.full_like(R, 0.2)
    H = torch.full_like(R, 5.0)
    
    total_loss, pde_loss, bc_core, bc_surf = physics_loss(pinn, R, alpha, K, H)
    assert total_loss.item() > 0.0
    assert not torch.isnan(total_loss)
    
    total_loss.backward()
    # Check that gradients flow to network weights
    for param in pinn.parameters():
        assert param.grad is not None


def test_solve_steady_state_pinn_wrapper():
    if not os.path.exists("surrogate_pinn.pth"):
        pytest.skip("surrogate_pinn.pth not found")
        
    R, C = solve_steady_state_pinn(alpha=1.0, K=0.5, H=2.0, n_cells=50)
    assert len(R) == 50
    assert len(C) == 50
    assert np.all(C >= 0.0) and np.all(C <= 1.0)
