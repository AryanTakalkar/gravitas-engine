import torch
import torch.nn as nn

class ReactionDiffusionPINN(nn.Module):
    """
    Parametric Physics-Informed Neural Network (PINN) for 1D Spherical Reaction-Diffusion.
    It takes physical parameters as inputs so it acts as a universal surrogate solver.
    """
    def __init__(self, hidden_layers=4, hidden_neurons=64):
        super().__init__()
        
        # Input layer takes 4 variables: R (radius), alpha (reaction rate), K (Michaelis), H (permeability)
        layers = [nn.Linear(4, hidden_neurons), nn.Tanh()]
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
            
        # Output layer gives 1 variable: C (Concentration)
        layers.append(nn.Linear(hidden_neurons, 1))
        
        # Sigmoid output (forces concentration to be strictly between 0 and 1, matching our physics scaling)
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, R, alpha, K, H):
        # Concatenate inputs into a single tensor
        inputs = torch.cat([R, alpha, K, H], dim=1)
        return self.network(inputs)

def physics_loss(pinn, R, alpha, K, H):
    """
    Computes the loss based entirely on the physics of the system (PDE + Boundary Conditions).
    We use automatic differentiation (autograd) to find the exact derivatives of the neural network.
    """
    # 1. Evaluate the Neural Network to get Concentration (C)
    C = pinn(R, alpha, K, H)
    
    # 2. Compute first derivative of C with respect to R (dC/dR)
    dC_dR = torch.autograd.grad(
        C, R, 
        grad_outputs=torch.ones_like(C), 
        create_graph=True
    )[0]
    
    # 3. Compute second derivative (d2C/dR2)
    d2C_dR2 = torch.autograd.grad(
        dC_dR, R, 
        grad_outputs=torch.ones_like(dC_dR), 
        create_graph=True
    )[0]
    
    # 4. The PDE Residual (The Physics Equation)
    # Laplacian in spherical coords: d2C/dR2 + (2/R)*dC/dR
    # We add a tiny epsilon to R to prevent division by zero at the core.
    laplacian = d2C_dR2 + (2.0 / (R + 1e-8)) * dC_dR
    
    # Reaction term (Michaelis-Menten)
    reaction = alpha * (C / (K + C + 1e-8))
    
    # Residual should be zero if physics is perfectly obeyed
    pde_residual = laplacian - reaction
    
    # 5. Compute the Loss Components (Mean Squared Error)
    loss_pde = torch.mean(pde_residual**2)
    
    # --- Boundary Conditions ---
    
    # BC 1: Symmetry at the core (R=0) -> dC/dR should be 0
    R_core = torch.zeros_like(R, requires_grad=True)
    C_core = pinn(R_core, alpha, K, H)
    dC_dR_core = torch.autograd.grad(C_core, R_core, grad_outputs=torch.ones_like(C_core), create_graph=True)[0]
    loss_bc_core = torch.mean(dC_dR_core**2)
    
    # BC 2: Robin boundary condition at the surface (R=1) -> dC/dR = H * (1 - C)
    R_surf = torch.ones_like(R, requires_grad=True)
    C_surf = pinn(R_surf, alpha, K, H)
    dC_dR_surf = torch.autograd.grad(C_surf, R_surf, grad_outputs=torch.ones_like(C_surf), create_graph=True)[0]
    
    bc_surf_residual = dC_dR_surf - H * (1.0 - C_surf)
    loss_bc_surf = torch.mean(bc_surf_residual**2)
    
    # Total Physics Loss
    # We highly weight the boundary conditions to ensure the network obeys the constraints
    total_loss = loss_pde + 10.0 * loss_bc_core + 10.0 * loss_bc_surf
    
    return total_loss, loss_pde, loss_bc_core, loss_bc_surf
