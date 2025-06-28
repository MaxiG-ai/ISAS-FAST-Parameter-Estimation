# Physics-Informed Neural Network for hyperelastic Material Parameter Identification
# Minimal example based on the hyperelastic beam problem

import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial


class PINN:
    """
    Physics-Informed Neural Network for hyperelastic material identification.
    This PINN learns material parameters (E, nu) from displacement measurements.
    """
    
    def __init__(self, hidden_dims=[32, 32], learning_rate=1e-3):
        """
        Initialize the PINN with neural network architecture.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimization
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Initialize neural network parameters
        self.params = self._init_network_params()
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
    def _init_network_params(self):
        """Initialize neural network parameters for displacement prediction."""
        key = jax.random.PRNGKey(42)
        layers = [3] + self.hidden_dims + [3]  # 3D input (x,y,z) -> 3D output (ux,uy,uz)
        
        params = []
        for i in range(len(layers) - 1):
            key, subkey = jax.random.split(key)
            w = jax.random.normal(subkey, (layers[i], layers[i+1])) * 0.1
            b = jnp.zeros(layers[i+1])
            params.append({'w': w, 'b': b})
        
        return params
    
    def neural_network(self, params, coords):
        """
        Neural network that predicts displacement given coordinates.
        
        Args:
            params: Network parameters
            coords: Input coordinates [x, y, z]
        
        Returns:
            Predicted displacement [ux, uy, uz]
        """
        x = coords
        for i, layer in enumerate(params):
            x = x @ layer['w'] + layer['b']
            if i < len(params) - 1:  # Apply activation to all but last layer
                x = jnp.tanh(x)
        return x
    
    def hyperelastic_energy(self, F, E, nu):
        """
        Neo-Hookean hyperelastic strain energy density.
        This is the same physics as in the original hyperelastic_beam.py
        
        Args:
            F: Deformation gradient tensor
            E: Young's modulus
            nu: Poisson's ratio
        
        Returns:
            Strain energy density
        """
        # Convert to Lamé parameters
        mu = E / (2.0 * (1.0 + nu))
        kappa = E / (3.0 * (1.0 - 2.0 * nu))
        
        # Compute invariants
        J = jnp.linalg.det(F)
        Jinv = J**(-2.0 / 3.0)
        I1 = jnp.trace(F.T @ F)
        
        # Neo-Hookean energy
        energy = (mu / 2.0) * (Jinv * I1 - 3.0) + (kappa / 2.0) * (J - 1.0)**2
        return energy
    
    def first_pk_stress(self, F, E, nu):
        """
        Compute first Piola-Kirchhoff stress using automatic differentiation.
        """
        energy_fn = lambda F_: self.hyperelastic_energy(F_, E, nu)
        P = jax.grad(energy_fn)(F)
        return P
    
    def compute_physics_loss(self, params, material_params, coords, target_displacements):
        """
        Compute physics-informed loss combining data fitting and physics constraints.
        
        Args:
            params: Neural network parameters
            material_params: Material parameters [E, nu]
            coords: Input coordinates
            target_displacements: Target displacement measurements
        
        Returns:
            Total loss value
        """
        E, nu = material_params
        
        # Neural network predictions
        pred_displacements = jax.vmap(self.neural_network, in_axes=(None, 0))(params, coords)
        
        # Data fitting loss
        data_loss = jnp.mean((pred_displacements - target_displacements)**2)
        
        # Physics loss: compute equilibrium residual
        def displacement_field(x):
            return self.neural_network(params, x)
        
        # Compute displacement gradients (simplified for demonstration)
        grad_u = jax.vmap(jax.jacfwd(displacement_field))(coords)
        
        # Deformation gradient F = I + grad_u
        I = jnp.eye(3)
        F = grad_u + I[None, :, :]
        
        # First Piola-Kirchhoff stress
        P = jax.vmap(self.first_pk_stress, in_axes=(0, None, None))(F, E, nu)
        
        # Equilibrium: div(P) = 0 (simplified as norm of P for demonstration)
        physics_loss = jnp.mean(jnp.linalg.norm(P, axis=(1, 2))**2)
        
        # Material parameter constraints
        E_penalty = jnp.maximum(0.0, -E + 1.0)  # E > 1
        nu_penalty = jnp.maximum(0.0, jnp.abs(nu) - 0.49)  # |nu| < 0.5
        
        total_loss = data_loss + 0.1 * physics_loss + 0.01 * (E_penalty + nu_penalty)
        
        return total_loss, {
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'total_loss': total_loss
        }


def generate_synthetic_data():
    """
    Generate synthetic displacement data for a hyperelastic beam problem.
    This simulates the result of the original hyperelastic_beam.py example.
    """
    print("Generating synthetic hyperelastic beam data...")
    
    # True material parameters (what we want to recover)
    E_true = 10.0
    nu_true = 0.3
    
    # Create 3D mesh points for a beam
    nx, ny, nz = 5, 5, 5
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny) 
    z = jnp.linspace(0, 1, nz)
    
    coords = jnp.array(jnp.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T
    
    # Simulate realistic hyperelastic displacements
    # This is a simplified analytical approximation of the FEM solution
    def analytical_displacement(coord):
        x, y, z = coord
        # Simulate beam bending and twisting (simplified)
        ux = 0.1 * x**2 * (y - 0.5) / E_true
        uy = 0.1 * x * (0.5 + (y - 0.5) * jnp.cos(jnp.pi/3) - (z - 0.5) * jnp.sin(jnp.pi/3) - y) / 2
        uz = 0.1 * x * (0.5 + (y - 0.5) * jnp.sin(jnp.pi/3) + (z - 0.5) * jnp.cos(jnp.pi/3) - z) / 2
        return jnp.array([ux, uy, uz])
    
    target_displacements = jax.vmap(analytical_displacement)(coords)
    
    print(f"Generated data for {len(coords)} points")
    print(f"True parameters: E = {E_true:.1f}, nu = {nu_true:.2f}")
    
    return coords, target_displacements, E_true, nu_true


def run_pinn_optimization():
    """
    Main PINN optimization for hyperelastic material parameter identification.
    """
    print("=" * 70)
    print("hyperelastic PINN: MATERIAL PARAMETER IDENTIFICATION")
    print("=" * 70)
    print("Objective: Learn hyperelastic material properties from displacement data")
    print()
    
    # Generate synthetic data
    coords, target_displacements, E_true, nu_true = generate_synthetic_data()
    
    # Initialize PINN
    pinn = PINN(hidden_dims=[16, 16], learning_rate=1e-3)
    
    # Initial material parameter guess
    material_params = jnp.array([5.0, 0.4])  # [E, nu] - different from true values
    print(f"Initial guess: E = {material_params[0]:.2f}, nu = {material_params[1]:.3f}")
    print(f"True values:   E = {E_true:.2f}, nu = {nu_true:.3f}")
    print()
    
    # JIT compile the loss function
    @jax.jit
    def loss_and_grad(nn_params, mat_params):
        loss_fn = lambda nn_p, mat_p: pinn.compute_physics_loss(
            nn_p, mat_p, coords, target_displacements)[0]
        
        loss_val, (nn_grad, mat_grad) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
            nn_params, mat_params)
        
        return loss_val, nn_grad, mat_grad
    
    # Optimization loop
    num_epochs = 200
    print("Starting optimization...")
    print("Epoch | Total Loss | Data Loss  | Physics Loss | E      | nu     | E Error | nu Error")
    print("-" * 85)
    
    for epoch in range(num_epochs + 1):
        # Compute loss and gradients
        total_loss, nn_grad, mat_grad = loss_and_grad(pinn.params, material_params)
        
        # Get detailed loss components for reporting
        _, loss_dict = pinn.compute_physics_loss(
            pinn.params, material_params, coords, target_displacements)
        
        # Update neural network parameters
        if epoch < num_epochs:
            updates, pinn.opt_state = pinn.optimizer.update(nn_grad, pinn.opt_state)
            pinn.params = optax.apply_updates(pinn.params, updates)
            
            # Update material parameters with smaller learning rate
            material_params = material_params - 0.01 * mat_grad
            
            # Clamp material parameters to physical ranges
            E_clamped = jnp.clip(material_params[0], 1.0, 50.0)
            nu_clamped = jnp.clip(material_params[1], -0.49, 0.49)
            material_params = jnp.array([E_clamped, nu_clamped])
        
        # Print progress
        E_error = abs(material_params[0] - E_true)
        nu_error = abs(material_params[1] - nu_true)
        
        if epoch % 40 == 0 or epoch == num_epochs:
            print(f"{epoch:5d} | {total_loss:10.6f} | {loss_dict['data_loss']:10.6f} | "
                  f"{loss_dict['physics_loss']:12.6f} | {material_params[0]:6.3f} | "
                  f"{material_params[1]:6.3f} | {E_error:7.4f} | {nu_error:8.4f}")
    
    print()
    print("OPTIMIZATION RESULTS:")
    print(f"  True parameters:      E = {E_true:.3f}, nu = {nu_true:.4f}")
    print(f"  Optimized parameters: E = {material_params[0]:.3f}, nu = {material_params[1]:.4f}")
    print(f"  Final errors:         E = {E_error:.6f}, nu = {nu_error:.6f}")
    print(f"  E error percentage:   {(E_error/E_true)*100:.2f}%")
    print(f"  nu error percentage:  {(nu_error/abs(nu_true))*100:.2f}%")
    
    # Test the trained model
    print()
    print("MODEL VERIFICATION:")
    pred_displacements = jax.vmap(pinn.neural_network, in_axes=(None, 0))(
        pinn.params, coords)
    rmse = jnp.sqrt(jnp.mean((pred_displacements - target_displacements)**2))
    max_displacement = jnp.max(jnp.linalg.norm(target_displacements, axis=1))
    
    print(f"  RMSE of predicted displacements: {rmse:.8f}")
    print(f"  Maximum displacement magnitude:  {max_displacement:.8f}")
    print(f"  Relative RMSE:                  {(rmse/max_displacement)*100:.3f}%")
    
    print()
    print("KEY hyperelastic PINN CONCEPTS DEMONSTRATED:")
    print("1. Neo-Hookean hyperelastic constitutive model")
    print("2. Deformation gradient and finite strain theory")
    print("3. First Piola-Kirchhoff stress computation via autodiff")
    print("4. Physics-informed loss combining data and equilibrium")
    print("5. Simultaneous neural network and material parameter optimization")
    print("6. Material parameter constraints (E > 0, |nu| < 0.5)")
    
    return material_params, (E_true, nu_true)


if __name__ == "__main__":
    # Run the hyperelastic PINN demonstration
    try:
        optimized_params, true_params = run_pinn_optimization()
        print(f"\nSUCCESS: PINN optimization completed!")
        print(f"Recovered material parameters: E = {optimized_params[0]:.3f}, nu = {optimized_params[1]:.4f}")
    except Exception as e:
        print(f"Error during PINN optimization: {e}")
        import traceback
        traceback.print_exc()
