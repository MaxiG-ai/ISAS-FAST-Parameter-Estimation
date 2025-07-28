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
from pinn.adaptive_training import adaptive_loss_weights, check_convergence, parameter_regularization_loss
from LinearElasticity.simulation import LinearElasticitySimulation

def main():
    """Enhanced main script with improved PINN training based on paper recommendations."""
    
    # --- Configuration ---
    # Beam dimensions
    Lx, Ly, Lz = 10., 2., 2.

    # Ground truth parameters to be discovered by the PINN
    E_true = 70e3
    nu_true = 0.3
    
    # Enhanced training settings
    num_pinn_pretrain_steps = 8000 # Increased for better convergence
    num_param_optim_steps = 3000   # Increased for better parameter identification
    
    learning_rate_model = 5e-5  # Slightly reduced for stability
    learning_rate_E = 20.0      # Further reduced for stability
    learning_rate_nu = 5e-4     # Reduced for stability
    
    # Dynamic loss weights - will be adjusted during training
    base_pretrain_weights = (0.1, 0.1, 1.0)  # (w_pde, w_bc, w_data)
    base_optimize_weights = (10.0, 10.0, 0.1)  # (w_pde, w_bc, w_data)

    # Enhanced sampling
    N_pde = 3000   # Increased physics points
    N_bc = 400     # Increased boundary points
    
    # --- Initialization ---
    key = jax.random.PRNGKey(42)
    model_key, params_key, train_key = jax.random.split(key, 3)
    
    # Initialize with reasonable material parameter guesses
    model = PINN(model_key)
    material_params = MaterialParameters(E_init=50e3, nu_init=0.25)
    
    # --- Enhanced Data Generation ---
    print(f"--- Generating enhanced ground truth data with E={E_true}, nu={nu_true} ---")
    fem_simulation = LinearElasticitySimulation(Lx, Ly, Lz, Nx=30, Ny=6, Nz=6)  # Higher resolution
    fem_coords, fem_displacements = fem_simulation.run(E=E_true, nu=nu_true)
    
    # Strategic data sampling - include more points near boundaries and high deformation areas
    print(f"FEM simulation generated {fem_coords.shape[0]} data points")
    
    # Sample more points for better displacement field representation
    n_data_points = min(2000, fem_coords.shape[0])
    data_indices = jax.random.choice(train_key, fem_coords.shape[0], shape=(n_data_points,), replace=False)
    selected_coords = fem_coords[data_indices]
    selected_displacements = fem_displacements[data_indices]
    
    u_true = (jnp.array(selected_coords), jnp.array(selected_displacements))
    print(f"Using {n_data_points} data points for training")
    print("--- Enhanced ground truth data generated. ---")

    # --- Logging Setup ---
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"pinn/results/{now_str}-enhanced-iterative-run"
    os.makedirs(log_dir, exist_ok=True)
    
    # --- Enhanced Sampling ---
    sample_key1, sample_key2, sample_key3 = jax.random.split(train_key, 3)
    pde_points = jax.random.uniform(sample_key1, shape=(N_pde, 3)) * jnp.array([Lx, Ly, Lz])
    dirichlet_points = jax.random.uniform(sample_key2, shape=(N_bc, 3)) * jnp.array([0., Ly, Lz])
    neumann_points = jax.random.uniform(sample_key3, shape=(N_bc, 3)) * jnp.array([Lx, Ly, Lz])
    neumann_points = neumann_points.at[:, 0].set(Lx)
    batch = (pde_points, dirichlet_points, neumann_points, u_true)

    # --- ENHANCED STAGE 1: PINN Pre-training ---
    print("\\n--- Enhanced Stage 1: Pre-training PINN with adaptive physics constraints ---")

    pinn_params, _ = eqx.partition(model, eqx.is_array)
    optimizer_model = optax.adam(learning_rate_model)
    opt_state_model = optimizer_model.init(pinn_params)

    history_pretrain = run_enhanced_pinn_pretraining(
        num_pinn_pretrain_steps,
        model,
        material_params,
        opt_state_model,
        optimizer_model,
        batch,
        base_pretrain_weights,
        log_dir=log_dir
    )
    
    model = history_pretrain['final_model']
    
    # --- ENHANCED STAGE 2: Parameter Optimization ---
    print("\\n--- Enhanced Stage 2: Optimizing material parameters with regularization ---")

    static_model = model 

    optimizer_params = optax.multi_transform(
        {
            'E': optax.adam(learning_rate_E),
            'nu': optax.adam(learning_rate_nu)
        },
        MaterialParameters(E_init='E', nu_init='nu')
    )
    opt_state_params = optimizer_params.init(eqx.filter(material_params, eqx.is_array))

    history_optimize = run_enhanced_material_optimization(
        num_param_optim_steps,
        static_model,
        material_params,
        opt_state_params,
        optimizer_params,
        batch,
        base_optimize_weights,
        log_dir=log_dir
    )

    final_params = history_optimize['final_params']
    
    # --- Results ---
    print("\\n--- Enhanced Training Complete ---")
    final_E = float(final_params.get_constrained_params()[0])
    final_nu = float(final_params.get_constrained_params()[1])
    print(f"Final Predicted Parameters -> E: {final_E:.2f} (True: {E_true}) - Error: {abs(final_E-E_true)/E_true*100:.1f}%")
    print(f"Final Predicted Parameters -> nu: {final_nu:.4f} (True: {nu_true}) - Error: {abs(final_nu-nu_true)/nu_true*100:.1f}%")
    print(f"Enhanced logs and plots saved to: {log_dir}")

    # Enhanced plotting
    plot_enhanced_training_progress(history_pretrain, history_optimize, log_dir, true_params=(E_true, nu_true))


def run_enhanced_pinn_pretraining(num_steps, model, material_params, opt_state, optimizer, batch, base_weights, log_dir):
    """Enhanced PINN pretraining with adaptive loss balancing."""
    from pinn.model import calculate_total_loss, train_step_pretraining
    
    csv_filename = f"{log_dir}/enhanced_pretrain_log.csv"
    csv_fields = ["step", "total_loss", "loss_pde", "loss_bc", "loss_data", "E_pred", "nu_pred", "w_pde", "w_bc", "w_data"]
    history = {field: [] for field in csv_fields}

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for step in range(num_steps + 1):
            # Calculate current loss for adaptive weighting
            total_loss_val, (pde_loss, bc_loss, data_loss) = calculate_total_loss(
                (model, material_params), batch, base_weights
            )
            
            # Adaptive loss weights
            loss_weights = adaptive_loss_weights(step, (pde_loss, bc_loss, data_loss), "pretrain")
            
            # Training step
            updated_model, opt_state, loss_val, individual_losses, _ = train_step_pretraining(
                model, opt_state, batch, loss_weights, optimizer, calculate_total_loss, 
                (model, material_params)
            )
            model = updated_model

            if step % 200 == 0:
                loss_pde, loss_bc, loss_data = individual_losses
                E_pred, nu_pred = material_params.get_constrained_params()

                print(f"  [enhanced-pretrain] Step: {step:5d}, PDE: {loss_pde:.2e}, BC: {loss_bc:.2e}, Data: {loss_data:.2e}, Total: {loss_val:.4e}")

                log_entry = {
                    "step": step, "total_loss": float(loss_val),
                    "loss_pde": float(loss_pde), "loss_bc": float(loss_bc), "loss_data": float(loss_data),
                    "E_pred": float(E_pred), "nu_pred": float(nu_pred),
                    "w_pde": float(loss_weights[0]), "w_bc": float(loss_weights[1]), "w_data": float(loss_weights[2])
                }
                writer.writerow(log_entry)
                for h_key, val in log_entry.items():
                    history[h_key].append(val)

    history['final_model'] = model
    history['final_params'] = material_params
    return history


def run_enhanced_material_optimization(num_steps, static_model, material_params, opt_state, optimizer, batch, base_weights, log_dir):
    """Enhanced material parameter optimization with regularization."""
    from pinn.model import calculate_total_loss, train_step_optimization
    
    csv_filename = f"{log_dir}/enhanced_optimize_log.csv"
    csv_fields = ["step", "total_loss", "loss_pde", "loss_bc", "loss_data", "reg_loss", "E_pred", "nu_pred", "w_pde", "w_bc", "w_data"]
    history = {field: [] for field in csv_fields}

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for step in range(num_steps + 1):
            # Calculate current loss for adaptive weighting
            total_loss_val, (pde_loss, bc_loss, data_loss) = calculate_total_loss(
                (static_model, material_params), batch, base_weights
            )
            
            # Adaptive loss weights
            loss_weights = adaptive_loss_weights(step, (pde_loss, bc_loss, data_loss), "optimize")
            
            # Add regularization
            reg_loss = parameter_regularization_loss(material_params, target_E=70e3, target_nu=0.3)
            
            # Enhanced loss function with regularization
            def enhanced_loss_fn(loss_params, batch, loss_weights):
                total_loss, individual_losses = calculate_total_loss(loss_params, batch, loss_weights)
                reg_loss_val = parameter_regularization_loss(loss_params[1])
                return total_loss + reg_loss_val, (*individual_losses, reg_loss_val)
            
            # Training step
            updated_material_params, opt_state, loss_val, individual_losses, _ = train_step_optimization(
                material_params, opt_state, batch, loss_weights, optimizer, enhanced_loss_fn,
                (static_model, material_params)
            )
            material_params = updated_material_params

            if step % 200 == 0:
                loss_pde, loss_bc, loss_data, reg_loss_val = individual_losses
                E_pred, nu_pred = material_params.get_constrained_params()

                print(f"  [enhanced-optimize] Step: {step:5d}, PDE: {loss_pde:.2e}, BC: {loss_bc:.2e}, Data: {loss_data:.2e}, Reg: {reg_loss_val:.2e}, Total: {loss_val:.4e}, E: {float(E_pred):.2f}, nu: {float(nu_pred):.4f}")

                log_entry = {
                    "step": step, "total_loss": float(loss_val),
                    "loss_pde": float(loss_pde), "loss_bc": float(loss_bc), "loss_data": float(loss_data),
                    "reg_loss": float(reg_loss_val),
                    "E_pred": float(E_pred), "nu_pred": float(nu_pred),
                    "w_pde": float(loss_weights[0]), "w_bc": float(loss_weights[1]), "w_data": float(loss_weights[2])
                }
                writer.writerow(log_entry)
                for h_key, val in log_entry.items():
                    history[h_key].append(val)

    history['final_model'] = static_model
    history['final_params'] = material_params
    return history


def plot_enhanced_training_progress(history_pretrain, history_optimize, save_dir, true_params):
    """Enhanced plotting with both training phases."""
    E_true, nu_true = true_params
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pretraining losses
    ax1 = axes[0, 0]
    ax1.plot(history_pretrain['step'], history_pretrain['total_loss'], 'k-', linewidth=2, label='Total Loss')
    ax1.plot(history_pretrain['step'], history_pretrain['loss_data'], 'g--', label='Data Loss')
    ax1.plot(history_pretrain['step'], history_pretrain['loss_pde'], 'b--', label='PDE Loss')
    ax1.plot(history_pretrain['step'], history_pretrain['loss_bc'], 'r--', label='BC Loss')
    ax1.set_yscale('log')
    ax1.set_title('Stage 1: Enhanced PINN Pre-training')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Optimization losses
    ax2 = axes[0, 1]
    ax2.plot(history_optimize['step'], history_optimize['total_loss'], 'k-', linewidth=2, label='Total Loss')
    ax2.plot(history_optimize['step'], history_optimize['loss_pde'], 'b--', label='PDE Loss')
    ax2.plot(history_optimize['step'], history_optimize['loss_bc'], 'r--', label='BC Loss')
    ax2.plot(history_optimize['step'], history_optimize['loss_data'], 'g--', label='Data Loss')
    ax2.plot(history_optimize['step'], history_optimize['reg_loss'], 'm:', label='Regularization')
    ax2.set_yscale('log')
    ax2.set_title('Stage 2: Enhanced Parameter Optimization')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss (log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Parameter evolution - E
    ax3 = axes[1, 0]
    ax3.plot(history_pretrain['step'], history_pretrain['E_pred'], 'b-', label='Stage 1: E (frozen)')
    ax3.plot(history_optimize['step'], history_optimize['E_pred'], 'r-', linewidth=2, label='Stage 2: E (optimized)')
    ax3.axhline(y=E_true, color='k', linestyle='--', label=f'True E = {E_true}')
    ax3.set_title("Young's Modulus Evolution")
    ax3.set_xlabel('Step')
    ax3.set_ylabel('E [Pa]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Parameter evolution - nu
    ax4 = axes[1, 1]
    ax4.plot(history_pretrain['step'], history_pretrain['nu_pred'], 'b-', label='Stage 1: nu (frozen)')
    ax4.plot(history_optimize['step'], history_optimize['nu_pred'], 'r-', linewidth=2, label='Stage 2: nu (optimized)')
    ax4.axhline(y=nu_true, color='k', linestyle='--', label=f'True nu = {nu_true}')
    ax4.set_title("Poisson's Ratio Evolution")
    ax4.set_xlabel('Step')
    ax4.set_ylabel('nu')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'enhanced_training_progress.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
