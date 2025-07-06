# Import some useful modules.
import jax.numpy as np

import numpy as onp
import os

from LinearElasticity.problem import LinearElasticity
from LinearElasticity.LinearElasticityEKF import LinearElasticityEKF

from util import run_and_solve, _mesh_config

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.utils import save_sol
from jax_fem import logger

import logging
logger.setLevel(logging.DEBUG)

# Material properties.
def _init(E=70e3, nu=0.3):
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
    problem.set_material_parameters(E, nu)
    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  

    x0 = np.array([E, nu])
    P0 = np.array([[1e8, 0], [0, 0.01]])
    Q = np.array([[1e2, 0], [0, 1e-4]])
    
    disp_var = 1e-12
    stress_var = 1e10
    R = np.diag(np.concatenate([np.asarray([disp_var] * 186), np.asarray([stress_var] * 38)]))
    ekf = LinearElasticityEKF(Q=Q, R=R, x0=x0, P0=P0)

    ### Solve problem and improve model
    for _ in range (5):
        u, vm_stress = run_and_solve(E, nu, problem=problem)
        z = np.concatenate([u, vm_stress])
        z = z + onp.random.normal(0, 1, size=z.shape)
        ekf.predict()
        ekf.update(z)
        x1, _ = ekf.get_state()
        E, nu = x1
        problem.set_material_parameters(E, nu)
        print(x1)
