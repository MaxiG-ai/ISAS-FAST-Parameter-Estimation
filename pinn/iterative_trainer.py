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
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our project
from pinn.model import PINN, MaterialParameters, train_step, calculate_data_loss, calculate_physics_loss
from LinearElasticity.simulation import LinearElasticitySimulation

def main():
    """Main script to run the PINN inverse problem training."""
    
    # --- Configuration ---
    # Beam dimensions
    Lx, Ly, Lz = 10., 2., 2.

    # Ground truth parameters to be discovered by the PINN
    E_true = 70e3
    nu_true = 0.3
    
    # Training settings
    num_pinn_pretrain_steps = 10000 # Steps to train PINN on data
    num_param_optim_steps = 3000   # Steps to optimize material params
    
    learning_rate_model = 1e-4
    # Adjusted for stability
    learning_rate_E = 5e1 
    learning_rate_nu = 5e-4
    
    # Loss weights
    loss_weights = (1.0, 1.0, 1.0) # (w_pde, w_bc, w_data) - used by different stages

    # Number of points to sample for each loss term
    N_pde = 2000
    N_bc = 250
    
    # --- Initialization ---
    key = jax.random.PRNGKey(42)
    model_key, params_key, train_key = jax.random.split(key, 3)
    
    # Initialize the PINN with wrong guesses for the material parameters
    model = PINN(model_key)
    # Start with incorrect material parameters, but closer to the true values
    material_params = MaterialParameters(E_init=50e3, nu_init=0.2)
    
    # --- Data Generation & Verification ---
    print(f"--- Generating ground truth data with E={E_true}, nu={nu_true} ---")
    fem_simulation = LinearElasticitySimulation(Lx, Ly, Lz)
    fem_coords, fem_displacements = fem_simulation.run(E=E_true, nu=nu_true)
    data_points = (jnp.array(fem_coords), jnp.array(fem_displacements))
    print("--- Ground truth data generated. ---")

    # --- Logging Setup ---
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"pinn/results/{now_str}-iterative-run"
    os.makedirs(log_dir, exist_ok=True)
    
    # --- STAGE 1: PINN Pre-training (fitting to data) ---
    print("\n--- Stage 1: Pre-training PINN on FEM data ---")
    
    # Optimizer for the PINN model ONLY
    pinn_params, _ = eqx.partition((model, material_params), eqx.is_array)
    optimizer_model = optax.adam(learning_rate_model)
    opt_state_model = optimizer_model.init(pinn_params)
    
    history_pretrain = run_training_stage(
        "pretrain",
        num_pinn_pretrain_steps,
        (model, material_params),
        opt_state_model,
        optimizer_model,
        calculate_data_loss,
        data_points,
        loss_weights,
        (Lx, Ly, Lz, N_pde, N_bc),
        train_key,
        log_dir
    )
    
    # The model is now pre-trained
    model = history_pretrain['final_model']
    
    # --- Verification after Pre-training ---
    print("\n--- Verifying Pre-trained PINN with Physics Loss ---")
    pde_points = jax.random.uniform(train_key, shape=(N_pde, 3)) * jnp.array([Lx, Ly, Lz])
    dirichlet_points = jax.random.uniform(train_key, shape=(N_bc, 3)) * jnp.array([0., Ly, Lz])
    neumann_points = jax.random.uniform(train_key, shape=(N_bc, 3)) * jnp.array([Lx, Ly, Lz])
    neumann_points = neumann_points.at[:, 0].set(Lx)
    verification_batch = (pde_points, dirichlet_points, neumann_points, data_points)

    physics_loss_val, (pde_loss, bc_loss, _) = calculate_physics_loss(
        (model, material_params), verification_batch, (1., 1., 0.)
    )
    print(f"  Physics Loss after pre-training: {physics_loss_val:.2e} (PDE: {pde_loss:.2e}, BC: {bc_loss:.2e})")
    print("--- Verification Complete ---")

    # Plot the pre-training history
    plot_pretraining_progress(history_pretrain, log_dir)
        
    print("\n--- Stage 1 Complete ---")
    print(f"Final Data Loss: {history_pretrain['total_loss'][-1]:.4e}")


    # --- STAGE 2: Parameter Optimization (fitting to physics) ---
    print("\n--- Stage 2: Optimizing material parameters using the pre-trained PINN ---")

    # The PINN model is now static. Only material_params will be trained.
    static_model = model 

    # Setup optimizer for E and nu
    optimizer_params = optax.multi_transform(
        {
            'E': optax.adam(learning_rate_E),
            'nu': optax.adam(learning_rate_nu)
        },
        MaterialParameters(E_init='E', nu_init='nu')
    )
    # Initialize the optimizer state with only the material parameters
    opt_state_params = optimizer_params.init(eqx.filter(material_params, eqx.is_array))

    history_optimize = run_training_stage(
        "optimize",
        num_param_optim_steps,
        material_params, # Pass only the trainable material parameters
        opt_state_params,
        optimizer_params,
        calculate_physics_loss,
        data_points,
        loss_weights,
        (Lx, Ly, Lz, N_pde, N_bc),
        train_key,
        log_dir,
        static_model=static_model # Pass the frozen PINN model separately
    )

    final_params = history_optimize['final_params']
    
    print("\n--- Stage 2 Complete ---")
    final_E = float(final_params.E)
    final_nu = float(final_params.nu)
    print(f"Final Predicted Parameters -> E: {final_E:.2f} (True: {E_true})")
    print(f"Final Predicted Parameters -> nu: {final_nu:.4f} (True: {nu_true})")
    print(f"Logs and plots saved to: {log_dir}")

    # --- Plotting ---
    plot_optimization_progress(history_optimize, log_dir, (E_true, nu_true))


def run_training_stage(stage_name, num_steps, trainable_params, opt_state, optimizer, loss_fn, data_points, loss_weights, sim_dims, key, log_dir, static_model=None):
    """Generic function to run a training stage."""
    
    Lx, Ly, Lz, N_pde, N_bc = sim_dims
    
    # This is the part of the model that gets updated.
    params_to_train, static_part_of_trainable = eqx.partition(trainable_params, eqx.is_array)

    csv_filename = f"{log_dir}/{stage_name}_log.csv"
    csv_fields = ["step", "total_loss", "loss_pde", "loss_bc", "loss_data", "E_pred", "nu_pred"]
    
    history = {field: [] for field in csv_fields}

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for step in range(num_steps + 1):
            key, iter_key = jax.random.split(key)
            pde_key, dir_key, neu_key = jax.random.split(iter_key, 3)
            
            pde_points = jax.random.uniform(pde_key, shape=(N_pde, 3)) * jnp.array([Lx, Ly, Lz])
            dirichlet_points = jax.random.uniform(dir_key, shape=(N_bc, 3)) * jnp.array([0., Ly, Lz])
            neumann_points = jax.random.uniform(neu_key, shape=(N_bc, 3)) * jnp.array([Lx, Ly, Lz])
            neumann_points = neumann_points.at[:, 0].set(Lx)
            
            batch = (pde_points, dirichlet_points, neumann_points, data_points)

            # Reconstruct the full set of parameters for the loss function
            if stage_name == "optimize":
                # For optimization, combine the static PINN with the current material params
                current_material_params = eqx.combine(params_to_train, static_part_of_trainable)
                loss_params = (static_model, current_material_params)
                step_params = current_material_params
            else:
                # For pre-training, the whole model is trainable
                loss_params = eqx.combine(params_to_train, static_part_of_trainable)
                step_params = loss_params

            # Perform one training step
            updated_params, opt_state, loss_val, individual_losses, _ = train_step(
                step_params, opt_state, batch, loss_weights, optimizer, loss_fn, loss_params, stage_name
            )
            params_to_train, _ = eqx.partition(updated_params, eqx.is_array)


            if step % 500 == 0:
                # Recombine to get the full model/params for logging
                if stage_name == "optimize":
                    logged_material_params = eqx.combine(params_to_train, static_part_of_trainable)
                    full_params_for_log = (static_model, logged_material_params)
                else:
                    full_params_for_log = eqx.combine(params_to_train, static_part_of_trainable)
                
                model, material_params = full_params_for_log

                loss_pde, loss_bc, loss_data = individual_losses
                E_pred = float(material_params.E)
                nu_pred = float(material_params.nu)
                
                if stage_name == "pretrain":
                    print(f"  [{stage_name}] Step: {step:5d}, Loss: {loss_val:.4e}")
                else:
                    print(f"  [{stage_name}] Step: {step:5d}, Loss: {loss_val:.4e}, E: {E_pred:.2f}, nu: {nu_pred:.4f}")

                log_entry = {
                    "step": step, "total_loss": float(loss_val),
                    "loss_pde": float(loss_pde), "loss_bc": float(loss_bc), "loss_data": float(loss_data),
                    "E_pred": E_pred, "nu_pred": nu_pred
                }
                writer.writerow(log_entry)
                for h_key, val in log_entry.items():
                    history[h_key].append(val)
    
    # Reconstruct final model/params
    final_trainable_part = eqx.combine(params_to_train, static_part_of_trainable)
    if stage_name == "optimize":
        history['final_model'] = static_model
        history['final_params'] = final_trainable_part
    else:
        final_model, final_params = final_trainable_part
        history['final_model'] = final_model
        history['final_params'] = final_params
    
    return history


def plot_pretraining_progress(history, save_dir):
    """Plots the pre-training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['step'], history['loss_data'], label='Data Loss (Pre-training)', color='green')
    plt.yscale('log')
    plt.title('Stage 1: PINN Pre-training on Data')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pretraining_progress.png'))
    plt.close()


def plot_optimization_progress(history, save_dir, true_params):
    """Plots the optimization history."""
    E_true, nu_true = true_params
    
    plt.figure(figsize=(12, 10))
    
    # Plot losses
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(history['step'], history['total_loss'], label='Total Physics Loss', color='black')
    ax1.plot(history['step'], history['loss_pde'], label='PDE Loss', linestyle='--')
    ax1.plot(history['step'], history['loss_bc'], label='BC Loss', linestyle='--')
    ax1.set_yscale('log')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Physics Loss (log scale)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Plot parameters
    ax2 = ax1.twinx()
    ax2.plot(history['step'], history['E_pred'], label='E (Young\'s Modulus)', color='tab:blue')
    ax2.axhline(y=E_true, color='tab:blue', linestyle='--', label='True E')
    ax2.plot(history['step'], history['nu_pred'], label='nu (Poisson\'s Ratio)', color='tab:red')
    ax2.axhline(y=nu_true, color='tab:red', linestyle='--', label='True nu')
    ax2.set_ylabel('Material Parameters')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.title('Stage 2: Material Parameter Optimization')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimization_progress.png'))
    plt.close()


if __name__ == '__main__':
    main()
