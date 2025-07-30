# Import some useful modules.
import jax.numpy as jnp

import matplotlib.pyplot as plt

from LinearElasticityEKF import LinearElasticityEKF

from util import run_and_solve, get_problem

# Import JAX-FEM specific modules.
from jax_fem import logger

import logging
logger.setLevel(logging.DEBUG)

# Material properties. Example inital values
def _init(E=10e3, nu=0.0):
    return E, nu

def _init_problem(E=70e3, nu=0.3):
    return E, nu

if __name__ == "__main__":
    problem = get_problem("linear_elasticity")
    
    # Set material parameters.
    E, nu = _init()
    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  
    x0 = jnp.array([E, nu])
    P0 = jnp.array([[1e10, 0], [0, 1e1]])
    # Process noise covariance
    Q = jnp.array([[1e10, 0], [0, 1e1]])

    # Measurement noise
    # The measurement noise is the uncertainty in the displacement field.
    # The displacement field is a vector of length 3703, so we create a diagonal
    # matrix with the same length.
    # Small covariance since we trust the simulation.
    disp_var = 1e-3
    stress_var = 1e3 
    R = jnp.diag(jnp.asarray([disp_var] * 3703))
    # Create an instance of the Extended Kalman Filter.
    ekf = LinearElasticityEKF(Q=Q, R=R, x0=x0, P0=P0)

    # Set the initial state and covariance.
    estimated_states = [_init()]
    measured_states = []
    covariance_states = [P0]
    
    ### Solve problem and improve model
    i = 0
    # Init and run problem
    problem_E, problem_nu = _init_problem()
    problem.set_material_parameters(problem_E, problem_nu)
    u, _, _, _ = run_and_solve(problem=problem, system_type=problem.to_string())
    for _ in range(10):

        # Run EKF
        ekf.predict()
        ekf.update(u)
        x1, P1 = ekf.get_state()
        estimated_states.append(x1)
        covariance_states.append(P1)
        i += 1
        print(f"ITER {i}")

    estimated_E = [a[0] for a in estimated_states]
    estimated_nu = [a[1] for a in estimated_states]
    covariance_E = [a[0] for a in covariance_states]
    covariance_nu = [a[1] for a in covariance_states]
    measured_E = [_init_problem()[0] for _ in estimated_E]
    measured_nu = [_init_problem()[1] for _ in estimated_nu]

    fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(10, 5))

    ax[0].plot(estimated_nu, color="red", label="EKF Estimate")
    ax[0].plot(measured_nu, color="blue", label="Simulation Measurement")
    ax[0].legend()
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Poisson's Ratio")

    ax[1].plot(estimated_E, color="red", label="EKF Estimate")
    ax[1].plot(measured_E, color="blue", label="Simulation Measurement")
    ax[1].legend()
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Young's Modulus [Pa]")

    fig.suptitle("EKF Estimates of Material Parameters")
    plt.savefig("plots/resultsEKF/EKF_estimate9.pdf")


# Calculate standard deviation and create uncertainty bands
std_E = [jnp.sqrt(P[0, 0]) for P in covariance_states]
std_nu = [jnp.sqrt(P[1, 1]) for P in covariance_states]

# Create upper and lower bounds for uncertainty bands
# Using the mean ± standard deviation
upper_E = [m + s for m, s in zip(estimated_E, std_E)]
lower_E = [m - s for m, s in zip(estimated_E, std_E)]

upper_nu = [m + s for m, s in zip(estimated_nu, std_nu)]
lower_nu = [m - s for m, s in zip(estimated_nu, std_nu)]

fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(10, 5))

ax[0].plot(estimated_nu, color="red", label="EKF Estimate")
ax[0].fill_between(range(len(estimated_nu)), lower_nu, upper_nu, color='red', alpha=0.2)
ax[0].plot(measured_nu, color="blue", label="Simulation Measurement")
ax[0].legend()
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Poisson's Ratio")

ax[1].plot(estimated_E, color="red", label="EKF Estimate")
ax[1].fill_between(range(len(estimated_E)), lower_E, upper_E, color='red', alpha=0.2)
ax[1].plot(measured_E, color="blue", label="Simulation Measurement")
ax[1].legend()
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Young's Modulus [Pa]")

fig.suptitle("EKF Estimates with Prediction Uncertainty ")
plt.savefig("plots/resultsEKF/EKF_estimate_uncertainty9.pdf")
