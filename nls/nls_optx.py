import jax.numpy as jnp
import optimistix as optx
import tests_optx


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
    # sigma_pred = lmbda * epsilon + 2 * mu * epsilon
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
    # Calculate predicted stress
    sigma_pred = calculate_stress(epsilon, params)
    return sigma_pred - sigma_mes


# Define the solver using Optimistix's Levenberg-Marquardt
solver = optx.LevenbergMarquardt(
    rtol = 1e-8,
    atol = 1e-8,
    norm = optx.rms_norm
)

# Initial guess for parameters [E, nu]
initial_params = jnp.array([20000, 0.1])

# Generate synthetic data
epsilon, sigma_mes = tests_optx.generate_real_data()

# Perform optimization
sol = optx.least_squares(residuals, solver, initial_params, args=(epsilon, sigma_mes))

pred_params = sol.value

print("Estimated parameters:", pred_params)



# 1. Calculate stress from strain and estimated parameters
# 2. Generate ground truth from Jax-FEM
# 3. Calculate residuals
# 4. Use Levenberg-Marquardt to optimize parameters