# Simplified Physics-Informed Neural Network for Material Parameter Identification
# This is a minimal example to demonstrate PINN concepts without external FEM dependencies

import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial


def generate_synthetic_data():
    """
    Generate synthetic displacement data for a simple 1D elastic problem.
    This simulates experimental or reference data that we want to match.
    """
    # True material parameters (what we want to recover)
    E_true = 210.0  # Young's modulus (GPa)
    
    # Create simple 1D displacement field: u = F*x/E (linear elasticity)
    x_points = np.linspace(0, 1, 20)  # 1D domain from 0 to 1
    applied_force = 1000.0  # Applied force
    
    # True displacement using linear elasticity: u = F*x/E
    u_true = applied_force * x_points / E_true
    
    return x_points, u_true, E_true, applied_force


def physics_model(x, E, applied_force):
    """
    Simple 1D elasticity model: u(x) = F*x/E
    This represents our physics understanding of the problem.
    
    Args:
        x: spatial coordinates
        E: Young's modulus (parameter to optimize)
        applied_force: applied load
    
    Returns:
        displacement field
    """
    return applied_force * x / E


def compute_physics_loss(E, x_data, u_target, applied_force):
    """
    Physics-informed loss function.
    Measures how well our current E parameter reproduces the target displacements.
    
    Args:
        E: Current estimate of Young's modulus
        x_data: spatial coordinates
        u_target: target displacement data
        applied_force: applied force
    
    Returns:
        loss value
    """
    # Predict displacements using current E
    u_predicted = physics_model(x_data, E, applied_force)
    
    # Mean squared error between predicted and target
    data_loss = jnp.mean((u_predicted - u_target)**2)
    
    # Physics constraint: ensure E stays positive (regularization)
    physics_loss = jnp.maximum(0.0, -E + 1.0)  # Penalty if E < 1
    
    total_loss = data_loss + 0.01 * physics_loss
    return total_loss


def pinn_optimization():
    """
    Main PINN optimization for material parameter identification.
    This demonstrates the core PINN concept in a minimal example.
    """
    print("=" * 60)
    print("SIMPLIFIED PINN FOR MATERIAL PARAMETER IDENTIFICATION")
    print("=" * 60)
    print("Learning objective: Identify Young's modulus E from displacement data")
    print()
    print("PINN CONCEPT:")
    print("- We have displacement measurements u(x) at different positions x")
    print("- We know the physics: u(x) = F*x/E (linear elasticity)")
    print("- We want to find E that makes our physics model match the data")
    print("- This is 'physics-informed' because we use domain knowledge")
    print()
    
    # Generate synthetic target data
    x_data, u_target, E_true, applied_force = generate_synthetic_data()
    print(f"Generated target data:")
    print(f"  - True Young's modulus: {E_true:.1f} GPa")
    print(f"  - Applied force: {applied_force:.1f} N")
    print(f"  - Number of data points: {len(x_data)}")
    print(f"  - Physics equation: u(x) = {applied_force:.0f} * x / E")
    print()
    
    # Initial guess for Young's modulus (what the PINN starts with)
    E_initial = 100.0  # Start with a different value
    print(f"Initial guess: E = {E_initial:.1f} GPa")
    print(f"Initial error: {abs(E_initial - E_true):.1f} GPa")
    
    # Show what the initial guess predicts
    u_initial = physics_model(x_data, E_initial, applied_force)
    initial_loss = jnp.mean((u_initial - u_target)**2)
    print(f"Initial prediction error (MSE): {initial_loss:.6f}")
    print()
    
    # Setup JAX optimization
    learning_rate = 1.0  # Larger learning rate for this simple problem
    optimizer = optax.adam(learning_rate)
    
    # Initialize parameter (just a single scalar in this case)
    E = E_initial
    opt_state = optimizer.init(E)
    
    # JIT compile the loss and gradient computation for speed
    @jax.jit
    def loss_and_grad(E_param):
        loss_fn = lambda E: compute_physics_loss(E, x_data, u_target, applied_force)
        return jax.value_and_grad(loss_fn)(E_param)
    
    # Optimization loop
    num_epochs = 100
    print("Starting optimization...")
    print("Epoch |    Loss    |     E     | Error   | % Error")
    print("-" * 50)
    
    # Store history for analysis
    loss_history = []
    E_history = []
    
    for epoch in range(num_epochs + 1):
        # Compute loss and gradient
        loss, grad = loss_and_grad(E)
        loss_history.append(float(loss))
        E_history.append(float(E))
        
        # Update parameter
        if epoch < num_epochs:  # Don't update on final iteration (just print)
            updates, opt_state = optimizer.update(grad, opt_state)
            E = optax.apply_updates(E, updates)
        
        # Print progress
        error = abs(E - E_true)
        error_percent = (error/E_true)*100
        if epoch % 20 == 0 or epoch == num_epochs:
            print(f"{epoch:5d} | {loss:10.6f} | {E:9.2f} | {error:7.2f} | {error_percent:6.2f}%")
    
    print()
    print("OPTIMIZATION RESULTS:")
    print(f"  True value:      E = {E_true:.2f} GPa")
    print(f"  Optimized value: E = {E:.2f} GPa")
    print(f"  Final error:     {error:.4f} GPa")
    print(f"  Error percentage: {error_percent:.2f}%")
    print(f"  Improvement:     {((E_initial - E_true)/E_true*100) - error_percent:.2f} percentage points")
    
    # Demonstrate the fitted model
    print()
    print("VERIFICATION:")
    u_fitted = physics_model(x_data, E, applied_force)
    rmse = jnp.sqrt(jnp.mean((u_fitted - u_target)**2))
    print(f"  RMSE of fitted displacements: {rmse:.6f}")
    print(f"  Max displacement (target): {jnp.max(u_target):.6f}")
    print(f"  Max displacement (fitted): {jnp.max(u_fitted):.6f}")
    
    # Show convergence behavior
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    print(f"  Loss reduction: {initial_loss:.6f} → {final_loss:.6f} ({(1-final_loss/initial_loss)*100:.1f}% decrease)")
    
    print()
    print("KEY PINN CONCEPTS DEMONSTRATED:")
    print("1. Physics model: u(x) = F*x/E (linear elasticity)")
    print("2. Parameter optimization: Finding E that best fits data")
    print("3. Physics constraints: Regularization to keep E > 0")
    print("4. JAX compilation: Fast gradient computation")
    print("5. Data-driven learning: Using displacement measurements")
    print("6. Inverse problem: Finding material properties from observations")
    
    print()
    print("NEXT STEPS FOR LEARNING:")
    print("- Try with nonlinear physics models (hyperelasticity)")
    print("- Add noise to the target data to test robustness")
    print("- Optimize multiple parameters simultaneously (E and ν)")
    print("- Use neural networks instead of analytical models")
    print("- Apply to partial differential equations (PDEs)")
    
    return float(E), E_true


if __name__ == "__main__":
    # Run the simplified PINN demonstration
    optimized_E, true_E = pinn_optimization()
