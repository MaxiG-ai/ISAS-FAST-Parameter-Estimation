import jax.numpy as jnp
import optimistix as optx
import jax


# 1. Calculate stress from strain and estimated parameters
# 2. Generate ground truth from Jax-FEM
# 3. Calculate residuals
# 4. Use Levenberg-Marquardt to optimize parameters

def lm_solver(init_params, epsilon, sigma_mes, sigma_pred):
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
    sol = optx.least_squares(residuals, solver, init_params, args=(sigma_pred, sigma_mes), max_steps=512)

    pred_params = sol.value
    return pred_params


def residuals(params, sigma_pred__sigma_mes):
    """ Calculate residuals between predicted and measured stress.
    params: [E, nu]
    sigma_pred: predicted stress tensor
    sigma_mes: measured stress tensor
    Returns: residuals
    """
    # Unpack strain and measured stress
    sigma_pred, sigma_mes = sigma_pred__sigma_mes

    # # Unpack parameters
    # E = params[0]
    # nu = params[1]

    # # Calculate predicted stress
    # sigma_pred = calculate_stress(epsilon, [E, nu])

    # Calculate residuals
    res_sigma = sigma_pred - sigma_mes

    return res_sigma

def stress_function(epsilon, params):
    """ Calculate stress from strain and parameters.
    epsilon: strain tensor, shape (3, 3)
    params: [E, nu] where E is Young's modulus and nu is Poisson's ratio
    Returns: stress tensor, shape (3, 3)
    """

    # Unpack parameters
    E = params[0]  # Young's modulus
    nu = params[1]  # Poisson's ratio

    # Calculate Lamé parameters from E and nu
    mu = E / (2 * (1 + nu))
    lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))

    # Calculate stress tensor from strain tensor
    sigma_pred = lmbda * jnp.trace(epsilon) * jnp.eye(3) + 2 * mu * epsilon
    
    return sigma_pred