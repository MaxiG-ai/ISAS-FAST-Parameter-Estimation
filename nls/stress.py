from LinearElasticity.problem import LinearElasticity

from util import run_and_solve, _mesh_config

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.utils import save_sol
from jax_fem import logger

import logging
logger.setLevel(logging.DEBUG)

def calculate_stress(params):
    
    problem_E, problem_nu = params

    mesh, ele_type, dirichlet_bc_info, location_fns = _mesh_config()

    # Create an instance of the problem.
    problem = LinearElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns=location_fns)

    problem.set_material_parameters(problem_E, problem_nu)
    _, _, sigma_pred, _ = run_and_solve(problem=problem)

    return sigma_pred