# Import some useful modules.
import jax.numpy as jnp
import jax

import os
import matplotlib.pyplot as plt

from LinearElasticity.problem import LinearElasticity
from nls.nls import lm_solver

from util import run_and_solve, _mesh_config

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.utils import save_sol
from jax_fem import logger

import logging
logger.setLevel(logging.DEBUG)

# Material properties. Example inital values
def _init(E=10e3, nu=0.0):
    return E, nu

def _init_problem(E=70e3, nu=0.3):
    return E, nu

if __name__ == "__main__":
    mesh, ele_type, dirichlet_bc_info, location_fns = _mesh_config()

    # Create an instance of the problem.
    problem = LinearElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns=location_fns)
    E, nu = _init()
    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  
    init_params = jnp.array([E, nu])
    

    estimated_states = [_init()]
    measured_states = []
    
    ### Solve problem and improve model
    i = 0
    # Init and run problem
    problem_E, problem_nu = _init_problem()
    problem.set_material_parameters(problem_E, problem_nu)
    _, _, sigma, epsilon = run_and_solve(problem=problem)
    for _ in range(5):
        # sigma_pred, epsilon_pred = calculate_stress(init_params)
        # Run NLS
        pred_params = lm_solver(init_params, epsilon, sigma)
        estimated_states.append(pred_params)
        i += 1
        # Set predicted materialparameters as initial guess for next iteration
        init_params = pred_params
        print(f"ITER {i}")

    estimated_E = [a[0] for a in estimated_states]
    estimated_nu = [a[1] for a in estimated_states]
    measured_E = [_init_problem()[0] for _ in estimated_E]
    measured_nu = [_init_problem()[1] for _ in estimated_nu]

    # print(f"Estimated nu: {estimated_nu} Measured nu: {measured_nu}")
    # print(f"Estimated E: {estimated_E} Measured E: {measured_E}")

    fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(10, 5))

    ax[0].plot(estimated_nu, color="red", label="NLS Estimate")
    ax[0].plot(measured_nu, color="blue", label="Simulation Measurement")
    ax[0].legend()
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Poisson's Ratio")

    ax[1].plot(estimated_E, color="red", label="NLS Estimate")
    ax[1].plot(measured_E, color="blue", label="Simulation Measurement")
    ax[1].legend()
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Young's Modulus [Pa]")

    fig.suptitle("NLS Estimates of Material Parameters")
    plt.savefig("plots/resultsNLS/NLS_estimate4.pdf")
