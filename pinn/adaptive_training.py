"""
Adaptive training utilities for better PINN convergence.
Based on the paper's recommendations for loss balancing.
"""

import jax.numpy as jnp

def adaptive_loss_weights(step, loss_values, phase="pretrain"):
    """
    Adaptive loss weight adjustment based on loss magnitudes.
    This helps balance different loss components as suggested in the paper.
    """
    loss_pde, loss_bc, loss_data = loss_values
    
    if phase == "pretrain":
        # During pretraining, focus on data but maintain physics
        base_weights = jnp.array([0.1, 0.1, 1.0])
        
        # If data loss becomes very small, increase physics weights
        if loss_data < 1e-4:
            base_weights = base_weights.at[0].set(0.5)  # Increase PDE weight
            base_weights = base_weights.at[1].set(0.5)  # Increase BC weight
            
    else:  # optimization phase
        # During optimization, focus on physics
        base_weights = jnp.array([10.0, 10.0, 0.1])
        
        # If physics losses are very small, can reduce their weights slightly
        if loss_pde < 1e-2 and loss_bc < 1e-2:
            base_weights = base_weights.at[0].set(5.0)
            base_weights = base_weights.at[1].set(5.0)
            base_weights = base_weights.at[2].set(0.5)  # Increase data weight
    
    return base_weights

def check_convergence(loss_history, patience=500, tolerance=1e-6):
    """
    Check if training has converged based on loss history.
    """
    if len(loss_history) < patience:
        return False
        
    recent_losses = loss_history[-patience:]
    loss_change = abs(recent_losses[-1] - recent_losses[0]) / abs(recent_losses[0])
    
    return loss_change < tolerance

def parameter_regularization_loss(params, target_E=None, target_nu=None):
    """
    Add regularization to prevent parameters from drifting too far.
    This can help when we have some prior knowledge.
    """
    reg_loss = 0.0
    
    E, nu = params.get_constrained_params()
    
    # Soft constraints to typical engineering material ranges
    if target_E is not None:
        reg_loss += 1e-6 * (E - target_E)**2
    
    if target_nu is not None:
        reg_loss += 1e-3 * (nu - target_nu)**2
        
    # Penalize extreme values
    reg_loss += 1e-8 * jnp.maximum(0, E - 150e3)**2  # Penalize if E > 150 GPa
    reg_loss += 1e-3 * jnp.maximum(0, nu - 0.45)**2  # Penalize if nu > 0.45
    
    return reg_loss
