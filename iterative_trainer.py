import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt
from collections import namedtuple

# Import from our project
from pinn.model import PINN, MaterialParameters, train_step, calculate_total_loss
from LinearElasticity.simulation import LinearElasticitySimulation

def main():
    """Main script to run the iterative PINN-FEM training."""
    
    # --- Configuration ---
    # Beam dimensions
    Lx, Ly, Lz = 10., 2., 2.
    
    # Training settings
    num_iterations = 10 # Number of outer loops (FEM recalculations)
    num_pinn_steps = 2000 # Number of PINN training steps per iteration
    learning_rate_model = 1e-4
    learning_rate_E = 2e2 # Adjusted learning rate for E
    learning_rate_nu = 5e-4 # Adjusted learning rate for nu
    
    # Loss weights
    loss_weights = (1.0, 10.0, 100.0) # (w_pde, w_bc, w_data)

    # Number of points to sample for each loss term
    N_pde = 2000
    N_bc = 250
    
    # --- Initialization ---
    key = jax.random.PRNGKey(42)
    model_key, params_key, train_key = jax.random.split(key, 3)
    
    # Initialize the PINN and Material Parameters
    model = PINN(model_key)
    material_params = MaterialParameters(E_init=60e3, nu_init=0.25)
    trainable = (model, material_params)
    
    # Setup optimizer
    # We assign different labels to the model weights, E, and nu to use different learning rates.
    model_leaves, material_leaves = eqx.filter(trainable, eqx.is_array)
    
    # Create a pytree for labels that exactly matches the material_leaves structure.
    # We map over the leaves, replacing the array values with string labels.
    def get_label(leaf):
        # This is a simple way to distinguish E and nu, assuming they are scalars.
        # A more robust solution might inspect names if the model gets more complex.
        return 'E' if leaf.shape == () else 'nu'
    
    # Since material_leaves is a MaterialParameters object containing two arrays (for E and nu),
    # and we know their order, we can create a new MaterialParameters object for the labels.
    # This is more explicit and robust than inspecting shapes.
    label_material_params = MaterialParameters(E_init='E', nu_init='nu')

    param_labels = (
        jax.tree_util.tree_map(lambda _: 'model', model_leaves),
        label_material_params
    )

    optimizer = optax.multi_transform(
        {
            'model': optax.adam(learning_rate_model),
            'E': optax.adam(learning_rate_E),
            'nu': optax.adam(learning_rate_nu)
        },
        param_labels
    )
    
    # Initialize FEM simulation
    fem_simulation = LinearElasticitySimulation(Lx, Ly, Lz)

    # --- Logging Setup ---
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"pinn/results/{now_str}-iterative-run"
    os.makedirs(log_dir, exist_ok=True)
    csv_filename = f"{log_dir}/training_log.csv"
    csv_fields = ["iteration", "step", "total_loss", "loss_pde", "loss_bc", "loss_data", "E_pred", "nu_pred"]
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        history = {field: [] for field in csv_fields}

        print("--- Starting Iterative PINN-FEM Training ---")
        
        # --- Iterative Training Loop ---
        for iteration in range(num_iterations):
            print(f"\n===== Iteration {iteration+1}/{num_iterations} =====")
            
            # 1. Get current material parameter predictions
            current_E = float(trainable[1].E)
            current_nu = float(trainable[1].nu)
            print(f"Running FEM with E={current_E:.2f}, nu={current_nu:.4f}")

            # 2. Run FEM simulation to get new "ground truth" data
            fem_coords, fem_displacements = fem_simulation.run(E=current_E, nu=current_nu)
            data_points = (jnp.array(fem_coords), jnp.array(fem_displacements))
            
            # Re-initialize optimizer state for the PINN training phase
            opt_state = optimizer.init(eqx.filter(trainable, eqx.is_array))

            # 3. Train the PINN with the new data
            print(f"Training PINN for {num_pinn_steps} steps...")
            for step in range(num_pinn_steps + 1):
                # Create a new batch of random points for this step
                iter_key, train_key = jax.random.split(train_key)
                pde_key, dir_key, neu_key = jax.random.split(iter_key, 3)
                
                pde_points = jax.random.uniform(pde_key, shape=(N_pde, 3)) * jnp.array([Lx, Ly, Lz])
                dirichlet_points = jax.random.uniform(dir_key, shape=(N_bc, 3)) * jnp.array([0., Ly, Lz])
                neumann_points = jax.random.uniform(neu_key, shape=(N_bc, 3)) * jnp.array([Lx, Ly, Lz])
                neumann_points = neumann_points.at[:, 0].set(Lx)
                
                batch = (pde_points, dirichlet_points, neumann_points, data_points)

                # Perform one training step
                trainable, opt_state, loss_val, individual_losses, _ = train_step(
                    trainable, opt_state, batch, loss_weights, optimizer
                )

                if step % 500 == 0:
                    loss_pde, loss_bc, loss_data = individual_losses
                    E_pred = float(trainable[1].E)
                    nu_pred = float(trainable[1].nu)
                    
                    print(f"  Step: {step:5d}, Loss: {loss_val:.4e}, E: {E_pred:.2f}, nu: {nu_pred:.4f}")

                    # Log data
                    log_entry = {
                        "iteration": iteration, "step": step,
                        "total_loss": float(loss_val), "loss_pde": float(loss_pde),
                        "loss_bc": float(loss_bc), "loss_data": float(loss_data),
                        "E_pred": E_pred, "nu_pred": nu_pred
                    }
                    writer.writerow(log_entry)
                    for key, val in log_entry.items():
                        history[key].append(val)

        print("\n--- Training Complete ---")
        final_E = float(trainable[1].E)
        final_nu = float(trainable[1].nu)
        print(f"Final Predicted Parameters -> E: {final_E:.2f}, nu: {final_nu:.4f}")
        print(f"Logs and plots saved to: {log_dir}")

    # --- Plotting ---
    plot_training_progress(history, log_dir)

def plot_training_progress(history, save_dir):
    """Plots the training history."""
    
    # Plot 1: Losses over time
    plt.figure(figsize=(12, 8))
    
    # Combine iteration and step for a continuous x-axis
    total_steps = [i * 5001 + s for i, s in zip(history['iteration'], history['step'])]

    plt.subplot(2, 1, 1)
    plt.plot(total_steps, history['total_loss'], label='Total Loss', color='black')
    plt.plot(total_steps, history['loss_pde'], label='PDE Loss', linestyle='--')
    plt.plot(total_steps, history['loss_bc'], label='BC Loss', linestyle='--')
    plt.plot(total_steps, history['loss_data'], label='Data Loss', linestyle='--')
    plt.yscale('log')
    plt.title('Training Losses')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # Plot 2: Parameter evolution
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax1.plot(total_steps, history['E_pred'], label='E (Young\'s Modulus)', color='tab:blue')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Young\'s Modulus (E)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(total_steps, history['nu_pred'], label='nu (Poisson\'s Ratio)', color='tab:red')
    ax2.set_ylabel('Poisson\'s Ratio (nu)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Evolution of Material Parameters')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()


if __name__ == '__main__':
    main()
