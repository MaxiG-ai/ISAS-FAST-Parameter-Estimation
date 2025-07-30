import jax.numpy as jnp
import optimistix as optx
import jax


# 1. Calculate stress from strain and estimated parameters
# 2. Generate ground truth from Jax-FEM
# 3. Calculate residuals
# 4. Use Levenberg-Marquardt to optimize parameters

def lm_solver(init_params, epsilon, sigma_mes):
    """
    This function sets up and runs the Levenberg-Marquardt optimization solver using Optimistix.
    It optimizes the parameters of a linear elasticity problem based on the residuals between
    predicted and measured stress tensors.
    Returns:
        pred_params: The optimized parameters [E, nu] where E is Young's modulus and nu is Poisson's ratio.
    """
    
    # Define the solver using Optimistix's Levenberg-Marquardt
    solver = optx.LevenbergMarquardt(
        rtol = 1e-8,
        atol = 1e-8,
        norm = optx.max_norm,
        verbose=frozenset(["loss", "step", "accepted", "step_size", "y"])
    )

    # Perform optimization
    sol = optx.least_squares(residuals, solver, init_params, args=(epsilon, sigma_mes))

    pred_params = sol.value
    return pred_params


def residuals(params, epsilon__sigma_mes):
    """ Calculate residuals between predicted and measured stress.
    params: [E, nu]
    sigma_pred: predicted stress tensor
    sigma_mes: measured stress tensor
    Returns: residuals
    """
    # Unpack strain and measured stress
    epsilon, sigma_mes = epsilon__sigma_mes

    # Calculate residuals
    sigma_pred = stress_function(epsilon, params)
    res_sigma = sigma_pred - sigma_mes

    return res_sigma

def stress_function(epsilon, params):
    """
    Compute stress independently for each 3x3 strain matrix.
    epsilon: strain tensor, shape (N, M, 3, 3)
    params: [E, nu]
    Returns: stress tensor, shape (N, M, 3, 3)
    """

    # Unpack parameters
    E, nu = params
    mu = E / (2 * (1 + nu))
    lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))

    # Flatten to shape (N*M, 3, 3)
    epsilon_flat = epsilon.reshape(-1, 3, 3)

    # Compute trace for each 3x3 matrix
    trace = jnp.trace(epsilon_flat, axis1=1, axis2=2)  # shape (N*M,)
    trace_expanded = trace[:, None, None]              # shape (N*M, 1, 1)

    # Identity matrix (3x3) broadcasted to each sample
    identity = jnp.eye(3)[None, :, :]                  # shape (1, 3, 3)

    # Compute stress: σ = λ tr(ε) I + 2μ ε
    sigma_flat = lmbda * trace_expanded * identity + 2 * mu * epsilon_flat  # (N*M, 3, 3)

    # Reshape back to (N, M, 3, 3)
    sigma_pred = sigma_flat.reshape(epsilon.shape)

    return sigma_pred