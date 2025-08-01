# A complete script to train a Physics-Informed Neural Network (PINN)
# for the inverse problem of identifying material parameters (Young's Modulus E and Poisson's Ratio v)
# from displacement data of a linear elastic beam.
#
# Based on the JAX-FEM example for a bending beam.

import jax
import jax.numpy as jnp
import equinox as eqx

# For reproducibility
key = jax.random.PRNGKey(42)

# --------------------------------------------------------------------------------
# ## Step 1: Define Model Architecture
# --------------------------------------------------------------------------------

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

class MaterialParameters(eqx.Module):
    E: jnp.ndarray
    nu: jnp.ndarray

    def __init__(self, E_init, nu_init):
        """Initializes with starting guesses for E and nu, or with string labels."""
        # This allows the class to be used for both training + for setting up the optimizer labels.
        if isinstance(E_init, str):
            self.E = E_init
            self.nu = nu_init
        else:
            self.E = jnp.array(E_init)
            self.nu = jnp.array(nu_init)

# --------------------------------------------------------------------------------
# ## Step 2: Define Loss Functions
# --------------------------------------------------------------------------------

def calculate_pde_residual(model, params, x, y, z):
    """Calculates the Navier-Cauchy equation residual for a single point."""
    
    # ensure x, y, z are scalars (not arrays)
    assert jnp.shape(x) == (), f"x shape: {jnp.shape(x)}"
    assert jnp.shape(y) == (), f"y shape: {jnp.shape(y)}"
    assert jnp.shape(z) == (), f"z shape: {jnp.shape(z)}"

    # Lamé parameters are derived from the trainable E and nu
    mu = params.E / (2 * (1 + params.nu))
    lmbda = (params.E * params.nu) / ((1 + params.nu) * (1 - 2 * params.nu))

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
def calculate_physics_loss(trainable_params, batch, loss_weights):
    """Calculates the weighted physics-based loss (PDE and BCs)."""
    model, material_params = trainable_params
    pde_points, dirichlet_points, neumann_points, _ = batch
    w_pde, w_bc, _ = loss_weights

    # 1. PDE Loss
    v_pde_res = jax.vmap(calculate_pde_residual, in_axes=(None, None, 0, 0, 0))
    pde_residuals = v_pde_res(model, material_params, pde_points[:,0], pde_points[:,1], pde_points[:,2])
    loss_pde = jnp.mean(pde_residuals**2)

    # 2. Boundary Condition Loss
    v_model = jax.vmap(model, in_axes=(0, 0, 0))
    
    # Dirichlet BC
    dirichlet_preds = v_model(dirichlet_points[:,0], dirichlet_points[:,1], dirichlet_points[:,2])
    loss_dirichlet = jnp.mean(dirichlet_preds**2)
    
    # Neumann BC
    v_traction = jax.vmap(calculate_traction, in_axes=(None, None, 0, 0, 0))
    traction_preds = v_traction(model, material_params, neumann_points[:,0], neumann_points[:,1], neumann_points[:,2])
    traction_target = jnp.array([0., 0., 100.])
    loss_neumann = jnp.mean((traction_preds - traction_target)**2)
    
    loss_bc = loss_dirichlet + loss_neumann
    
    total_loss = w_pde * loss_pde + w_bc * loss_bc
    
    return total_loss, (loss_pde, loss_bc, jnp.array(0.))


@eqx.filter_jit
def calculate_data_loss(trainable_params, batch, loss_weights):
    """Calculates the weighted data-based loss."""
    model, _ = trainable_params # Material parameters are not used here
    _, _, _, data_points = batch
    _, _, w_data = loss_weights

    # Data Loss
    v_model = jax.vmap(model, in_axes=(0, 0, 0))
    data_coords, data_displacements = data_points
    data_preds = v_model(data_coords[:,0], data_coords[:,1], data_coords[:,2])
    loss_data = jnp.mean((data_preds - data_displacements)**2)

    total_loss = w_data * loss_data
    
    return total_loss, (jnp.array(0.), jnp.array(0.), loss_data)


@eqx.filter_jit
def calculate_total_loss(trainable_params, batch, loss_weights):
    """Calculates the weighted total loss."""
    model, material_params = trainable_params
    pde_points, dirichlet_points, neumann_points, data_points = batch
    w_pde, w_bc, w_data = loss_weights
    
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
    total_loss = w_data * loss_data + w_bc * loss_bc + w_pde * loss_pde
    
    # Return individual losses for logging
    return total_loss, (loss_pde, loss_bc, loss_data)

# --------------------------------------------------------------------------------
# ## Step 3: Training Setup
# --------------------------------------------------------------------------------


# TODO: merge to one function
@eqx.filter_jit
def train_step_pretraining(model, opt_state, batch, loss_weights, optimizer, loss_fn, loss_params):
    """Performs a single training step for PINN pretraining (only model parameters updated)."""
    
    # Get loss value and gradients w.r.t. the loss parameters
    (loss_val, individual_losses), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(loss_params, batch, loss_weights)
    
    # For pretraining, we only need gradients w.r.t. the model
    # Since loss_params = (model, material_params), grads[0] are the model gradients
    model_grads = grads[0]
    
    updates, opt_state = optimizer.update(model_grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss_val, individual_losses, grads


@eqx.filter_jit  
def train_step_optimization(material_params, opt_state, batch, loss_weights, optimizer, loss_fn, loss_params):
    """Performs a single training step for material parameter optimization (only material params updated)."""
    
    # Get loss value and gradients w.r.t. the loss parameters
    (loss_val, individual_losses), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(loss_params, batch, loss_weights)
    
    # For optimization, we only need gradients w.r.t. the material parameters
    # Since loss_params = (static_model, material_params), grads[1] are the material param gradients
    param_grads = grads[1]
    
    updates, opt_state = optimizer.update(param_grads, opt_state, material_params)
    material_params = eqx.apply_updates(material_params, updates)
    
    return material_params, opt_state, loss_val, individual_losses, grads