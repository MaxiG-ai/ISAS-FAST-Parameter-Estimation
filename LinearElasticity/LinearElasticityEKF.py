import numpy as np
import jax.numpy as jnp
from ekf.ekf import ExtendedKalmanFilter
from LinearElasticity.problem import LinearElasticity
from util import run_and_solve, _mesh_config
import jax
from jax_fem.solver import ad_wrapper


class LinearElasticityEKF(ExtendedKalmanFilter):
    def __init__(self, Q, R, x0, P0, f=None, h=None, F_jacobian=None, H_jacobian=None):

        def f(x, u=None):
            return x

        def F_jacobian(x, u=None):
            return jnp.eye(len(x))


        

        def h(x, epsilon=None):
            # Convert to float for external code
            E, nu = float(x[0]), float(x[1])
            mesh, ele_type, dirichlet_bc_info, location_fns = _mesh_config()
            problem = LinearElasticity(
                mesh,
                vec=3,
                dim=3,
                ele_type=ele_type,
                dirichlet_bc_info=dirichlet_bc_info,
                location_fns=location_fns
            )
            problem.set_material_parameters(E, nu)
            u, _, _, _ = run_and_solve(problem)
            return jnp.asarray(u)

        # Use adjoint-based Jacobian via ad_wrapper
        def H_jacobian(x, epsilon=1e-5):
            # Set up the problem as in h(x)
            E, nu = float(x[0]), float(x[1])
            mesh, ele_type, dirichlet_bc_info, location_fns = _mesh_config()
            problem = LinearElasticity(
                mesh,
                vec=3,
                dim=3,
                ele_type=ele_type,
                dirichlet_bc_info=dirichlet_bc_info,
                location_fns=location_fns
            )
            problem.set_material_parameters(E, nu)
            # Wrap the forward solve with ad_wrapper
            fwd = ad_wrapper(problem)
            # Compute Jacobian using JAX's jacobian on the adjoint-wrapped function
            jac = jax.jacobian(lambda params: jnp.asarray(fwd(params)[0]))(jnp.array([E, nu]))
            return jac

        self.f = f
        self.F_jacobian = F_jacobian
        self.h = h
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0