import jax.numpy as np
from jax_fem.problem import Problem

# Weak forms.
class LinearElasticity(Problem):
    def set_material_parameters(self, E, nu):
        self.E = E
        self.nu = nu
        self.mu = E / (2. * (1. + nu))
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def get_material_parameters(self):
        return self.E, self.nu

    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    def get_tensor_map(self):
        def stress(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = self.lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * self.mu * epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0., 100.])
        return [surface_map]
