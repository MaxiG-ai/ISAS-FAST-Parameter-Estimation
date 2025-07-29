import jax.numpy as np
import os

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem import logger
import logging

# Reduce verbosity from jax-fem
logger.setLevel(logging.WARNING)

class LinearElasticitySimulation:
    """A wrapper for the JAX-FEM linear elasticity simulation."""
    def __init__(self, Lx=10., Ly=2., Lz=2., Nx=25, Ny=5, Nz=5, ele_type='TET10'):
        """Initializes the simulation setting up the mesh and boundary conditions."""
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.ele_type = ele_type
        self.cell_type = get_meshio_cell_type(self.ele_type)
        
        # Generate mesh
        data_dir = os.path.join(os.path.dirname(__file__), '../data')
        os.makedirs(data_dir, exist_ok=True)
        meshio_mesh = box_mesh_gmsh(Nx=self.Nx, Ny=self.Ny, Nz=self.Nz,
                                    Lx=self.Lx, Ly=self.Ly, Lz=self.Lz,
                                    data_dir=data_dir, ele_type=self.ele_type)
        self.mesh = Mesh(meshio_mesh.points, hio_mesh.cells_dict[self.cell_type])

        # Define boundary locations
        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)

        def right(point):
            return np.isclose(point[0], self.Lx, atol=1e-5)

        # Define Dirichlet boundary values
        def zero_dirichlet_val(point):
            return 0.
        
        self.dirichlet_bc_info = [[left] * 3, [0, 1, 2], [zero_dirichlet_val] * 3]
        self.location_fns = [right]

    def run(self, E, nu):
        """
        Runs the simulation with the given material parameters.

        Args:
            E (float): Young's Modulus.
            nu (float): Poisson's Ratio.

        Returns:
            tuple: A tuple containing:
                - points (np.ndarray): The coordinates of the mesh nodes.
                - displacements (np.ndarray): The displacement vector at each node.
        """
        # Define the problem with the specified material properties
        problem = self._create_problem(E, nu)
        
        # Solve the problem
        sol_list = solver(problem) # Use default solver
        
        # Return node coordinates and the corresponding displacements
        return self.mesh.points, sol_list[0]

    def _create_problem(self, E, nu):
        """Creates a JAX-FEM problem instance with the given material parameters."""
        
        # Define the LinearElasticity problem class within this scope
        # to capture the material parameters E and nu.
        class _LinearElasticity(Problem):
            def get_tensor_map(self):
                # Lamé parameters derived from E and nu
                mu = E / (2. * (1. + nu))
                lmbda = (E * nu) / ((1. + nu) * (1. - 2. * nu))
                
                def stress(u_grad):
                    epsilon = 0.5 * (u_grad + u_grad.T)
                    sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
                    return sigma
                return stress

            def get_surface_maps(self):
                def surface_map(u, x):
                    # Traction force on the right face
                    return np.array([0., 0., 100.])
                return [surface_map]

        return _LinearElasticity(self.mesh,
                                 vec=3,
                                 dim=3,
                                 ele_type=self.ele_type,
                                 dirichlet_bc_info=self.dirichlet_bc_info,
                                 location_fns=self.location_fns)
