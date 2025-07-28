import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our project
from pinn.model import PINN, MaterialParameters, calculate_total_loss
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
    num_pinn_pretrain_steps = 5000 # Increased for better physics learning
    num_param_optim_steps = 2000   # Increased for better convergence
    
    learning_rate_model = 1e-4
    # Improved learning rates based on parameter scales
    learning_rate_E = 50.0  # Reduced for better stability
    learning_rate_nu = 1e-3
    
    # Loss weights for different stages
    # Stage 1: Focus on data fitting but include physics for consistency
    # Following the paper: data loss should dominate but physics must be present
    pretrain_loss_weights = (0.1, 0.1, 1.0)  # (w_pde, w_bc, w_data)
    # Stage 2: Focus on physics but keep some data constraint
    # Following the paper: physics should dominate but data provides constraint
    optimize_loss_weights = (10.0, 10.0, 0.1)  # (w_pde, w_bc, w_data)

    # Number of points to sample for each loss term
    N_pde = 2000
    N_bc = 250
    
    # --- Initialization ---
    key = jax.random.PRNGKey(42)
    model_key, params_key, train_key = jax.random.split(key, 3)
    
    # Initialize the PINN with reasonable guesses for the material parameters
    model = PINN(model_key)
    # Start with better initial guesses - closer to typical engineering materials
    # This follows the paper's recommendation for better initialization
    material_params = MaterialParameters(E_init=50e3, nu_init=0.25)
    
    # --- Data Generation & Verification ---
    print(f"--- Generating ground truth data with E={E_true}, nu={nu_true} ---")
    fem_simulation = LinearElasticitySimulation(Lx, Ly, Lz)
    fem_coords, fem_displacements = fem_simulation.run(E=E_true, nu=nu_true)
    
    # Use more comprehensive data sampling - include internal points, not just surface
    # This follows the paper's emphasis on using full displacement field
    print(f"FEM simulation generated {fem_coords.shape[0]} data points")
    
    # Sample a subset of FEM points for training (to avoid overfitting)
    n_data_points = min(1000, fem_coords.shape[0])  # Use up to 1000 data points
    data_indices = jax.random.choice(train_key, fem_coords.shape[0], shape=(n_data_points,), replace=False)
    selected_coords = fem_coords[data_indices]
    selected_displacements = fem_displacements[data_indices]
    
    u_true = (jnp.array(selected_coords), jnp.array(selected_displacements))
    print(f"Using {n_data_points} data points for training")
    print("--- Ground truth data generated. ---")

    # --- Logging Setup ---
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"pinn/results/{now_str}-iterative-run"
    os.makedirs(log_dir, exist_ok=True)
    
    # --- Sample all collocation and boundary points ONCE ---
    pde_points = jax.random.uniform(train_key, shape=(N_pde, 3)) * jnp.array([Lx, Ly, Lz])
    dirichlet_points = jax.random.uniform(train_key, shape=(N_bc, 3)) * jnp.array([0., Ly, Lz])
    neumann_points = jax.random.uniform(train_key, shape=(N_bc, 3)) * jnp.array([Lx, Ly, Lz])
    neumann_points = neumann_points.at[:, 0].set(Lx)
    batch = (pde_points, dirichlet_points, neumann_points, u_true)

    # --- STAGE 1: PINN Pre-training (fitting to data) ---
    print("\n--- Stage 1: Pre-training PINN on FEM data ---")

    # Optimizer for the PINN model ONLY
    pinn_params, _ = eqx.partition(model, eqx.is_array)
    optimizer_model = optax.adam(learning_rate_model)
    opt_state_model = optimizer_model.init(pinn_params)

    history_pretrain = run_pinn_pretraining(
        num_pinn_pretrain_steps,
        model,
        material_params,
        opt_state_model,
        optimizer_model,
        batch,
        pretrain_loss_weights,
        log_dir=log_dir
    )
    
    # The model is now pre-trained
    model = history_pretrain['final_model']
    
    # --- Verification after Pre-training ---
    print("\n--- Verifying Pre-trained PINN with All Loss Components ---")
    total_loss_val, (pde_loss, bc_loss, data_loss) = calculate_total_loss(
        (model, material_params), batch, (1., 1., 1.)
    )
    print(f"  Total Loss after pre-training: {total_loss_val:.2e}")
    print(f"  Individual losses - PDE: {pde_loss:.2e}, BC: {bc_loss:.2e}, Data: {data_loss:.2e}")
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

    history_optimize = run_material_optimization(
        num_param_optim_steps,
        static_model,
        material_params,
        opt_state_params,
        optimizer_params,
        batch,
        optimize_loss_weights,
        log_dir=log_dir
    )

    final_params = history_optimize['final_params']
    
    print("\n--- Stage 2 Complete ---")
    final_E = float(final_params.E)
    final_nu = float(final_params.nu)
    print(f"Final Predicted Parameters -> E: {final_E:.2f} (True: {E_true})")
    print(f"Final Predicted Parameters -> nu: {final_nu:.4f} (True: {nu_true})")
    print(f"Logs and plots saved to: {log_dir}")

    # --- Plotting ---
    plot_pretraining_progress(history_optimize, log_dir, true_params=(E_true, nu_true), phase="optimize")


def run_pinn_pretraining(num_steps, model, material_params, opt_state, optimizer, batch, loss_weights, log_dir):
    """
    Pre-train the PINN model on data only.
    Material parameters are kept constant and not used in the loss.
    """
    from pinn.model import calculate_data_loss, train_step_pretraining
    
    csv_filename = f"{log_dir}/pretrain_log.csv"
    csv_fields = ["step", "total_loss", "loss_pde", "loss_bc", "loss_data", "E_pred", "nu_pred"]
    history = {field: [] for field in csv_fields}

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for step in range(num_steps + 1):
            # Only train the PINN model, material params stay constant
            updated_model, opt_state, loss_val, individual_losses, _ = train_step_pretraining(
                model, opt_state, batch, loss_weights, optimizer, calculate_total_loss, 
                (model, material_params)
            )
            model = updated_model

            if step % 200 == 0:
                loss_pde, loss_bc, loss_data = individual_losses
                E_pred = float(material_params.E)
                nu_pred = float(material_params.nu)

                print(f"  [pretrain] Step: {step:5d}, PDE: {loss_pde:.2e}, BC: {loss_bc:.2e}, Data: {loss_data:.2e}, Total: {loss_val:.4e}")

                log_entry = {
                    "step": step, "total_loss": float(loss_val),
                    "loss_pde": float(loss_pde), "loss_bc": float(loss_bc), "loss_data": float(loss_data),
                    "E_pred": E_pred, "nu_pred": nu_pred
                }
                writer.writerow(log_entry)
                for h_key, val in log_entry.items():
                    history[h_key].append(val)

    history['final_model'] = model
    history['final_params'] = material_params
    return history


def run_material_optimization(num_steps, static_model, material_params, opt_state, optimizer, batch, loss_weights, log_dir):
    """
    Optimize material parameters using the pre-trained (frozen) PINN model.
    Only material parameters are updated, the PINN model stays frozen.
    """
    from pinn.model import calculate_total_loss, train_step_optimization
    
    csv_filename = f"{log_dir}/optimize_log.csv"
    csv_fields = ["step", "total_loss", "loss_pde", "loss_bc", "loss_data", "E_pred", "nu_pred"]
    history = {field: [] for field in csv_fields}

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for step in range(num_steps + 1):
            # Only train material parameters, PINN model is frozen
            updated_material_params, opt_state, loss_val, individual_losses, _ = train_step_optimization(
                material_params, opt_state, batch, loss_weights, optimizer, calculate_total_loss,
                (static_model, material_params)
            )
            material_params = updated_material_params

            if step % 200 == 0:
                loss_pde, loss_bc, loss_data = individual_losses
                E_pred = float(material_params.E)
                nu_pred = float(material_params.nu)

                print(f"  [optimize] Step: {step:5d}, PDE-Loss: {loss_pde:.2e}, BC-Loss: {loss_bc:.2e}, Data-Loss: {loss_data:.2e}, Total Loss: {loss_val:.4e}, E: {E_pred:.2f}, nu: {nu_pred:.4f}")

                log_entry = {
                    "step": step, "total_loss": float(loss_val),
                    "loss_pde": float(loss_pde), "loss_bc": float(loss_bc), "loss_data": float(loss_data),
                    "E_pred": E_pred, "nu_pred": nu_pred
                }
                writer.writerow(log_entry)
                for h_key, val in log_entry.items():
                    history[h_key].append(val)

    history['final_model'] = static_model
    history['final_params'] = material_params
    return history


def plot_pretraining_progress(history, save_dir, true_params=None, phase="pretrain"):
    """
    Plots the training/optimization history showing all loss components and material parameters.
    If true_params is provided, overlays true E and nu values.
    """

    plt.figure(figsize=(12, 8))

    # First subplot: Loss components
    plt.subplot(2, 1, 1)
    plt.plot(history['step'], history['total_loss'], label='Total Loss', color='black', linewidth=2)
    plt.plot(history['step'], history['loss_data'], label='Data Loss (weighted)', color='green', linestyle='--')
    plt.plot(history['step'], history['loss_pde'], label='PDE Loss (weighted)', color='blue', linestyle='--')
    plt.plot(history['step'], history['loss_bc'], label='BC Loss (weighted)', color='red', linestyle='--')
    plt.yscale('log')
    if phase == "pretrain":
        plt.title('Stage 1: PINN Pre-training with Physics Constraints')
    else:
        plt.title('Stage 2: Material Parameter Optimization')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    # Second subplot: Material parameters with dual y-axes
    ax_E = plt.subplot(2, 1, 2)
    ax_nu = ax_E.twinx()
    color_E = 'tab:blue'
    color_nu = 'tab:red'
    p1, = ax_E.plot(history['step'], history['E_pred'], label="E (Young's Modulus)", color=color_E)
    p2, = ax_nu.plot(history['step'], history['nu_pred'], label="nu (Poisson's Ratio)", color=color_nu)
    if true_params is not None:
        E_true, nu_true = true_params
        l1 = ax_E.axhline(y=E_true, color=color_E, linestyle='--', label='True E')
        l2 = ax_nu.axhline(y=nu_true, color=color_nu, linestyle='--', label='True nu')
    ax_E.set_xlabel('Training Step')
    ax_E.set_ylabel("E (Young's Modulus) [Pa]", color=color_E)
    ax_nu.set_ylabel("nu (Poisson's Ratio)", color=color_nu)
    ax_E.tick_params(axis='y', labelcolor=color_E)
    ax_nu.tick_params(axis='y', labelcolor=color_nu)
    # Set y-ticks and limits for E and nu
    ax_E.set_ylim(40000, 85000)
    ax_E.set_yticks([40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000])
    ax_nu.set_ylim(0.1, 0.4)
    ax_nu.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    # Combine legends from both axes
    lines_E, labels_E = ax_E.get_legend_handles_labels()
    lines_nu, labels_nu = ax_nu.get_legend_handles_labels()
    ax_nu.legend(lines_E + lines_nu, labels_E + labels_nu, loc='best')
    if phase == "pretrain":
        ax_E.set_title('Material Parameters During Pre-training (Should Remain Constant)')
    else:
        ax_E.set_title('Material Parameters During Optimization')
    ax_E.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'{phase}_progress.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()

if __name__ == '__main__':
    main()
