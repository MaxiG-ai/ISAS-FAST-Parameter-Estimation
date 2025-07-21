# Import some useful modules.
import jax.numpy as np

import numpy as onp
import os
import matplotlib.pyplot as plt

from LinearElasticity.problem import LinearElasticity
from LinearElasticity.LinearElasticityEKF import LinearElasticityEKF

from util import run_and_solve, _mesh_config

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.utils import save_sol
from jax_fem import logger

import logging
logger.setLevel(logging.DEBUG)

# Material properties. Example inital values
def _init(E=50e3, nu=0.1):
    return E, nu

def _init_problem(E=80e3, nu=0.45):
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
    x0 = np.array([E, nu])
    P0 = np.array([[1e8, 0], [0, 1e8]])
    Q = np.array([[1e2, 0], [0, 1e2]])
    
    disp_var = 10
    stress_var = 1e3
    R = np.diag(np.asarray([disp_var] * 3703))
    ekf = LinearElasticityEKF(Q=Q, R=R, x0=x0, P0=P0)

    estimated_states = [_init()]
    measured_states = []
    
    ### Solve problem and improve model
    i = 0
    for _ in range(30):
        # Init and run problem
        problem_E, problem_nu = _init_problem()
        problem.set_material_parameters(problem_E, problem_nu)
        u, _ = run_and_solve(problem=problem)

        # Run EKF
        ekf.predict()
        ekf.update(u)
        x1, _ = ekf.get_state()
        estimated_states.append(x1)
        i += 1
        print(f"ITER {i}")

    estimated_E = [a[0] for a in estimated_states]
    estimated_nu = [a[1] for a in estimated_states]
    measured_E = [_init_problem()[0] for _ in estimated_E]
    measured_nu = [_init_problem()[1] for _ in estimated_nu]

    print(estimated_nu)
    print(measured_nu)

    plt.plot(estimated_E, color="red", label="EKF estimate")
    plt.plot(measured_E, color="blue", label="Simulation Measurement")
    plt.legend()
    plt.savefig("plots/nu2.pdf")
