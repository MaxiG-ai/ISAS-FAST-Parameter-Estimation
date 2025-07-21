import numpy as np
from ekf.ekf import ExtendedKalmanFilter
from LinearElasticity.problem import LinearElasticity
from util import run_and_solve, _mesh_config


class LinearElasticityEKF(ExtendedKalmanFilter):
    def __init__(self, Q, R, x0, P0, f=None, h=None, F_jacobian=None, H_jacobian=None):
        def f(x, u=None):
            return x

        def F_jacobian(x, u=None):
            return np.eye(len(x))

        def h(x, epsilon=None):
            E, nu = x

            mesh, ele_type, dirichlet_bc_info, location_fns = _mesh_config()
            problem = LinearElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns=location_fns)
            problem.set_material_parameters(E, nu)
            #u, vm_stress = run_and_solve(problem)
            u, _ = run_and_solve(problem)

            #return np.concatenate([u, vm_stress])
            return u
            

        def H_jacobian(x, epsilon=1e-05):
            n_outputs = len(self.h(x))
            n_states = len(x)
            H = np.zeros((n_outputs, n_states))

            for i in range(n_states):
                dx = np.zeros_like(x)
                dx[i] = epsilon

                h_plus = self.h(x + dx)
                h_minus = self.h(x - dx)

                H[:, i] = (h_plus - h_minus) / (2 * epsilon)
            return H

        self.f = f
        self.F_jacobian = F_jacobian
        self.h = h
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0