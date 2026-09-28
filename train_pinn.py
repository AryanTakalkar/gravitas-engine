import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np
from src.gravitas.ai.pinn import ReactionDiffusionPINN, physics_loss

def train():
    print("Initializing Physics-Informed Neural Network (PINN)...")
    
    pinn = ReactionDiffusionPINN(hidden_layers=4, hidden_neurons=64)
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    
    epochs = 2000
    batch_size = 1000
    
    history_total = []
    history_pde = []
    
    # ---------------------------------------------------------
    # STATE-OF-THE-ART SAMPLING: Latin Hypercube Sampling (LHS)
    # ---------------------------------------------------------
    # We sample 4 dimensions: R, alpha, K, H
    sampler = qmc.LatinHypercube(d=4)
    
    # Define strictly realistic biological bounds for our dimensionless parameters
    # R (radius): strictly 0 to 1 (center to edge)
    # alpha (reaction rate scale): 0.01 to 100.0
    # K (Michaelis constant scale): 0.01 to 10.0
    # H (Mass transfer / Gravity permeability): 0.1 to 100.0
    l_bounds = [0.0, 0.01, 0.01, 0.1]
    u_bounds = [1.0, 100.0, 10.0, 100.0]
    
    print(f"Starting training for {epochs} epochs using LHS...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Generate Latin Hypercube Samples for this batch
        sample = sampler.random(n=batch_size)
        scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
        
        # 2. Convert to PyTorch Tensors
        # We only need gradients with respect to R (column 0)
        R_np = scaled_sample[:, 0:1]
        alpha_np = scaled_sample[:, 1:2]
        K_np = scaled_sample[:, 2:3]
        H_np = scaled_sample[:, 3:4]
        
        R = torch.tensor(R_np, dtype=torch.float32, requires_grad=True)
        alpha = torch.tensor(alpha_np, dtype=torch.float32)
        K = torch.tensor(K_np, dtype=torch.float32)
        H = torch.tensor(H_np, dtype=torch.float32)
        
        # 3. Calculate Physics Loss
        loss, loss_pde, loss_bc_core, loss_bc_surf = physics_loss(pinn, R, alpha, K, H)
        
        # 4. Backpropagation
        loss.backward()
        optimizer.step()
        
        history_total.append(loss.item())
        history_pde.append(loss_pde.item())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.4e} | PDE Loss: {loss_pde.item():.4e} | Surf BC Loss: {loss_bc_surf.item():.4e}")
            
    print("Training Complete!")
    
    # Save the trained surrogate model
    torch.save(pinn.state_dict(), "surrogate_pinn.pth")
    print("Model saved to 'surrogate_pinn.pth'")
    
    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(history_total, label="Total Physics Loss", color="blue")
    plt.plot(history_pde, label="PDE (Equation) Loss", color="orange", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Convergence (Latin Hypercube Sampling)")
    plt.legend()
    plt.savefig("pinn_training_curve.png")
    print("Training curve saved to 'pinn_training_curve.png'")
    
if __name__ == "__main__":
    train()
