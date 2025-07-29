import jax.numpy as jnp
import optimistix as optx
import jax

# from nls.stress import calculate_stress

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
        norm = optx.two_norm,
        verbose=frozenset(["loss", "step"])
    )

    # Perform optimization
    sol = optx.least_squares(residuals, solver, init_params, args=(epsilon, sigma_mes), max_steps=512)

    pred_params = sol.value
    print(pred_params)
    return pred_params

# 1. Calculate stress from strain and estimated parameters
# 2. Generate ground truth from Jax-FEM
# 3. Calculate residuals
# 4. Use Levenberg-Marquardt to optimize parameters

def calculate_stress(epsilon, params):
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
    print(sigma_pred)
    return sigma_pred

def residuals(params, epsilon__sigma_mes):
    """ Calculate residuals between predicted and measured stress.
    params: [E, nu]
    epsilon: strain tensor
    sigma_mes: measured stress tensor
    Returns: residuals
    """
    # Unpack strain and measured stress
    epsilon, sigma_mes = epsilon__sigma_mes
    # Unpack parameters
    E = params[0]
    nu = params[1]
    # # Ensure E and nu are in valid ranges
    # # E should be positive, nu should be in [0, 0.5)
    # E  = jax.nn.softplus(E) + 1e-6
    # nu = 0.499 * jax.nn.sigmoid(nu)

    # Calculate predicted stress
    sigma_pred = calculate_stress(epsilon, [E, nu])

    res_sigma = sigma_pred - sigma_mes

    return res_sigma