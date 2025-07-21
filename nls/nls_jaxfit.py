import jax
import jax.numpy as jnp
from jaxfit import CurveFit



# def navier_cauchy_equation(u_fn, params):
#     """
#     Navier-Cauchy stress-strain relation for curve fitting.
#     u_fn: displacement function, shape (N, 3, 3) or (3, 3) for a single sample
#     params: [E, nu]
#     Returns: stress prediction, shape (N, 3, 3) or (3, 3)
#     """
#     # Define Lamé parameters from Young's modulus E and Poisson's ratio nu
#     E, nu = params
#     mu = E / (2 * (1 + nu))
#     lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    
#     u_T = jnp.transpose(u_fn, axes=(0, 2, 1))  # Transpose displacement for strain calculation
#     jac_u = jax.jacobian(u_fn)  # Calculate Jacobian for strain tensor
#     jac_u_T = jax.jacobian(u_T)  # Calculate Jacobian for transposed displacement
#     epsilon = 0.5 * (jac_u + jac_u_T)  # Calculate strain tensor
#     eye = jnp.eye(3)  # Identity matrix for stress calculation
#     sigma = lmbda * jnp.trace(epsilon) * eye + 2 * mu * epsilon  # Calculate stress tensor
#     return sigma.squeeze()  # Remove batch dim if input was single sample

def linear_function(u, params):
    """Linear function for curve fitting."""
    return params[0] * u + params[1] * u 

params_init = jnp.array([20000, 0.1])
bounds = ([0, 0], [None, 0.5])
jcf = CurveFit()
popt, pcov = jcf.curve_fit(linear_function, u, sigma_mes, params_init, bounds=bounds)
sigma_fit = linear_function(u, popt)