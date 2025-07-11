# main_pinn.py
#
# A complete script to train a Physics-Informed Neural Network (PINN)
# for the inverse problem of identifying material parameters (Young's Modulus E and Poisson's Ratio v)
# from displacement data of a linear elastic beam.
#
# Based on the JAX-FEM example for a bending beam.

import jax
import jax.numpy as jnp
import jax.tree_util
import equinox as eqx
import optax
import time
import csv
from datetime import datetime

# For reproducibility
key = jax.random.PRNGKey(42)

# --------------------------------------------------------------------------------
# ## Step 1: Define Model Architecture
# --------------------------------------------------------------------------------
# The PINN model is a simple MLP that maps spatial coordinates (x, y, z)
# to a displacement vector (ux, uy, uz).

class PINN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        """Initializes the neural network."""
        # Using 4 hidden layers with 128 neurons each. Tanh is a good activation for PINNs.
        self.mlp = eqx.nn.MLP(in_size=3, out_size=3, width_size=128, depth=4, activation=jnp.tanh, key=key)

    def __call__(self, x, y, z):
        """Performs a forward pass."""
        in_vec = jnp.array([x, y, z])
        return self.mlp(in_vec)

# The material parameters E and nu are also defined as a trainable Equinox module.
# This allows JAX to compute gradients with respect to them automatically.
class MaterialParameters(eqx.Module):
    E: jnp.ndarray
    nu: jnp.ndarray

    def __init__(self, E_init, nu_init):
        """Initializes with starting guesses for E and nu."""
        # It's important to start from a "wrong" guess to let the network learn.
        self.E = jnp.array(E_init) 
        self.nu = jnp.array(nu_init)

# --------------------------------------------------------------------------------
# ## Step 2: Define Loss Functions
# --------------------------------------------------------------------------------
# The total loss is a combination of physics loss (PDE residual),
# boundary condition loss, and data mismatch loss.

def calculate_pde_residual(model, params, x, y, z):
    """Calculates the Navier-Cauchy equation residual for a single point."""
    
    # Defensive: ensure x, y, z are scalars (not arrays)
    assert jnp.shape(x) == (), f"x shape: {jnp.shape(x)}"
    assert jnp.shape(y) == (), f"y shape: {jnp.shape(y)}"
    assert jnp.shape(z) == (), f"z shape: {jnp.shape(z)}"

    # Lamé parameters are derived from the trainable E and nu
    mu = params.E / (2 * (1 + params.nu))
    lmbda = (params.E * params.nu) / ((1 + params.nu) * (1 - 2 * params.nu))

    # We need second derivatives of the displacement u w.r.t coordinates (x, y, z).
    # JAX's hessian is perfect for this. It computes d2f/(dxi dxj).
    # We apply it to each component of the displacement vector u = [u0, u1, u2].
    u = lambda x, y, z: model(x, y, z)

    hessian_u0 = jax.hessian(lambda x, y, z: u(x, y, z)[0])(x, y, z)
    hessian_u1 = jax.hessian(lambda x, y, z: u(x, y, z)[1])(x, y, z)
    hessian_u2 = jax.hessian(lambda x, y, z: u(x, y, z)[2])(x, y, z)

    # Defensive: ensure hessians are at least 2D
    hessian_u0 = jnp.atleast_2d(hessian_u0)
    hessian_u1 = jnp.atleast_2d(hessian_u1)
    hessian_u2 = jnp.atleast_2d(hessian_u2)

    # The Navier-Cauchy residual for zero body force is: (lmbda + mu) * grad(div(u)) + mu * laplacian(u)
    # laplacian(u_i) = trace(hessian(u_i))
    laplacian_u0 = jnp.trace(hessian_u0)
    laplacian_u1 = jnp.trace(hessian_u1)
    laplacian_u2 = jnp.trace(hessian_u2)

    # grad(div(u))_i = d/dx_i (du0/dx0 + du1/dx1 + du2/dx2)
    grad_div_u_0 = hessian_u0[0][0] + hessian_u1[0][1] + hessian_u2[0][2]
    grad_div_u_1 = hessian_u0[1][0] + hessian_u1[1][1] + hessian_u2[1][2]
    grad_div_u_2 = hessian_u0[2][0] + hessian_u1[2][1] + hessian_u2[2][2]

    res_0 = (lmbda + mu) * grad_div_u_0 + mu * laplacian_u0
    res_1 = (lmbda + mu) * grad_div_u_1 + mu * laplacian_u1
    res_2 = (lmbda + mu) * grad_div_u_2 + mu * laplacian_u2

    return jnp.array([res_0, res_1, res_2])


def calculate_traction(model, params, x, y, z):
    """Calculates the traction vector sigma . n"""
    jac_u = jax.jacfwd(lambda x, y, z: model(x, y, z))(x, y, z)
    jac_u = jnp.atleast_2d(jac_u)
    epsilon = 0.5 * (jac_u + jac_u.T)
    epsilon = jnp.atleast_2d(epsilon)

    mu = params.E / (2 * (1 + params.nu))
    lmbda = (params.E * params.nu) / ((1 + params.nu) * (1 - 2 * params.nu))

    sigma = lmbda * jnp.trace(epsilon) * jnp.eye(3) + 2 * mu * epsilon

    # Normal vector for the right face (x=Lx) is n = [1, 0, 0]
    n = jnp.array([1., 0., 0.])
    traction = sigma @ n
    return traction


@eqx.filter_jit
def calculate_total_loss(trainable_params, batch):
    """Calculates the weighted total loss."""
    model, material_params = trainable_params
    pde_points, dirichlet_points, neumann_points, data_points = batch
    
    # 1. PDE Loss
    # Vectorize the residual calculation over all collocation points
    v_pde_res = jax.vmap(calculate_pde_residual, in_axes=(None, None, 0, 0, 0))
    pde_residuals = v_pde_res(model, material_params, pde_points[:,0], pde_points[:,1], pde_points[:,2])
    loss_pde = jnp.mean(pde_residuals**2)
    
    # 2. Boundary Condition Loss
    # Vectorize model prediction for batches
    v_model = jax.vmap(model, in_axes=(0, 0, 0))
    
    # Dirichlet BC: u=0 on the left face
    dirichlet_preds = v_model(dirichlet_points[:,0], dirichlet_points[:,1], dirichlet_points[:,2])
    loss_dirichlet = jnp.mean(dirichlet_preds**2)
    
    # Neumann BC: Traction boundary condition on the right face
    v_traction = jax.vmap(calculate_traction, in_axes=(None, None, 0, 0, 0))
    traction_preds = v_traction(model, material_params, neumann_points[:,0], neumann_points[:,1], neumann_points[:,2])
    traction_target = jnp.array([0., 0., 100.]) # From example.py
    loss_neumann = jnp.mean((traction_preds - traction_target)**2)
    
    loss_bc = loss_dirichlet + loss_neumann
    
    # 3. Data Loss
    # Compare model prediction to the "ground truth" FEM data
    data_coords, data_displacements = data_points
    data_preds = v_model(data_coords[:,0], data_coords[:,1], data_coords[:,2])
    loss_data = jnp.mean((data_preds - data_displacements)**2)
    
    # 4. Total Loss with weights
    # These weights can be tuned to improve convergence.
    w_pde = 1.0
    w_bc = 10.0
    w_data = 100.0
    total_loss = w_data * loss_data + w_bc * loss_bc + w_pde * loss_pde
    
    return total_loss

# --------------------------------------------------------------------------------
# ## Step 3: Training Setup
# --------------------------------------------------------------------------------

@eqx.filter_jit
def train_step(trainable_params, opt_state, batch):
    """Performs a single training step."""
    loss_val, grads = eqx.filter_value_and_grad(calculate_total_loss)(trainable_params, batch)
    
    updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
    trainable_params = eqx.apply_updates(trainable_params, updates)
    
    return trainable_params, opt_state, loss_val, grads

def generate_ground_truth_data(key, num_points):
    """
    Placeholder for generating data with the FEM model.
    In a real scenario, you would run your jax-fem script here and extract points.
    For this example, we generate some plausible dummy data.
    """
    print("--- Generating placeholder 'ground truth' data... ---")
    E_true = 70e3
    nu_true = 0.3
    
    # Generate random points inside the beam
    data_key, key = jax.random.split(key)
    coords = jax.random.uniform(data_key, shape=(num_points, 3)) * jnp.array([Lx, Ly, Lz])
    
    # Create a simple analytical displacement field that vaguely resembles beam bending
    # u_z ~ C * x^2, u_x ~ -C * x*z
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    C = 0.01
    ux = -C * x * (z - Lz/2)
    uy = jnp.zeros_like(x)
    uz = C * x**2
    
    # Add some noise
    noise_key, key = jax.random.split(key)
    noise = 0.01 * jax.random.normal(noise_key, shape=(num_points, 3))
    
    displacements = jnp.stack([ux, uy, uz], axis=1) + noise
    print(f"--- Generated {num_points} data points with E_true={E_true}, nu_true={nu_true} ---")
    
    return coords, displacements


# --------------------------------------------------------------------------------
# ## Step 4: Main Execution
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Configuration ---
    # Beam dimensions from example.py
    Lx, Ly, Lz = 10., 2., 2.
    
    # Training settings
    learning_rate = 1e-4
    learning_rate_material=1e-1
    num_steps = 20000
    
    # Number of points to sample for each loss term per step
    N_pde = 2000 # Collocation points inside the domain
    N_bc = 250   # Boundary points on each relevant face
    N_data = 500 # Number of "experimental" data points
    
    # --- Initialization ---
    model_key, params_key, data_key, train_key = jax.random.split(key, 4)
    
    # Initialize the network
    model = PINN(model_key)
    
    # Initialize material params with a "wrong" guess
    material_params = MaterialParameters(E_init=40e3, nu_init=0.45)
    
    # Combine model and material params into a single PyTree for the optimizer
    trainable = (model, material_params)
    
    # Filter out non-trainable/static parts
    params, static = eqx.partition(trainable, eqx.is_array)
    
    # Label each parameter as 'material' or 'model' (compatible with all Equinox versions)
    def label_param(x):
        # If the parent is MaterialParameters, label as 'material', else 'model'
        # We check the type of the object containing the leaf
        # Since we only have two modules, this is safe
        if isinstance(x, MaterialParameters):
            # This is the module, not the leaf, so skip
            return None
        return 'material' if hasattr(x, 'shape') and x.shape == () and hasattr(params, 'E') and (x is params.E or x is params.nu) else 'model'

    # But the above is not robust for pytree leaves, so instead:
    # We know params is a tuple: (model_params, material_params)
    # We want to label all leaves in model_params as 'model', all in material_params as 'material'
    def label_tree(tree, label):
        return jax.tree_util.tree_map(lambda _: label, tree)
    param_labels = (label_tree(params[0], 'model'), label_tree(params[1], 'material'))

    optimizer = optax.multi_transform(
        {
            'model': optax.adam(learning_rate=learning_rate),
            'material': optax.adam(learning_rate=learning_rate_material)
        },
        param_labels=param_labels
    )
    opt_state = optimizer.init(params)

    # --- Data Generation ---
    data_coords, data_displacements = generate_ground_truth_data(data_key, N_data)
    data_points = (data_coords, data_displacements)

    # --- Training Loop ---
    print("\n--- Starting PINN Training ---")
    start_time = time.time()

    # Prepare CSV file
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"results/{now_str}-training-run.csv"
    csv_fields = ["step", "loss", "E_pred", "nu_pred", "grad_E", "grad_nu", "elapsed_time"]
    csv_file = open(csv_filename, mode="w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    for step in range(num_steps + 1):
        # Sample points for the current batch
        iter_key, train_key = jax.random.split(train_key)
        pde_key, dir_key, neu_key = jax.random.split(iter_key, 3)

        # PDE collocation points (randomly inside the beam)
        pde_points = jax.random.uniform(pde_key, shape=(N_pde, 3)) * jnp.array([Lx, Ly, Lz])

        # Dirichlet boundary points (left face, x=0)
        dirichlet_points = jax.random.uniform(dir_key, shape=(N_bc, 3)) * jnp.array([0., Ly, Lz])

        # Neumann boundary points (right face, x=Lx)
        neumann_points = jax.random.uniform(neu_key, shape=(N_bc, 3)) * jnp.array([Lx, Ly, Lz])
        # Force x to be exactly Lx
        neumann_points = neumann_points.at[:, 0].set(Lx)

        # Assemble batch
        batch = (pde_points, dirichlet_points, neumann_points, data_points)

        # Perform a training step
        trainable_params = eqx.combine(params, static)
        trainable_params, opt_state, loss, grads = train_step(trainable_params, opt_state, batch)

        params, static = eqx.partition(trainable_params, eqx.is_array)

        if step % 200 == 0:
            current_model, current_params = trainable_params
            E_pred = float(current_params.E)
            nu_pred = float(current_params.nu)

            model_grads, material_grads = grads
            grad_E = float(material_grads.E)
            grad_nu = float(material_grads.nu)

            elapsed_time = time.time() - start_time
            print(f"Grad E: {grad_E}, Grad nu:{grad_nu}\n")
            print(f"Step: {step:5d}, Loss: {loss:.4e}, E_pred: {E_pred:.2f}, nu_pred: {nu_pred:.4f}, Time: {elapsed_time:.2f}s")

            # Write to CSV
            csv_writer.writerow({
                "step": step,
                "loss": float(loss),
                "E_pred": E_pred,
                "nu_pred": nu_pred,
                "grad_E": grad_E,
                "grad_nu": grad_nu,
                "elapsed_time": elapsed_time
            })
            csv_file.flush()
            start_time = time.time()

    csv_file.close()

    # --- Final Result ---
    print("\n--- Training Complete ---")
    final_model, final_params = trainable_params
    print(f"Final Prediction -> E: {final_params.E:.2f} (True Value: 70000.00)")
    print(f"Final Prediction -> nu: {final_params.nu:.4f} (True Value: 0.3000)")
    print(f"Results written to {csv_filename}")