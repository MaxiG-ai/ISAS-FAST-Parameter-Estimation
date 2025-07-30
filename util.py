import os

import jax.numpy as jnp
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem.solver import solver

from LinearElasticity.problem import LinearElasticity


def run_and_solve(problem, system_type):
    if system_type == "linear_elasticity":
        E, nu = problem.get_material_parameters()
        mu = E / (2. * (1. + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        sol_list = solver(problem, solver_options={'umfpack_solver': {}})
        u_grad = problem.fes[0].sol_to_grad(sol_list[0])
        epsilon = 0.5 * (u_grad + u_grad.transpose(0,1,3,2))
        sigma = lmbda * jnp.trace(epsilon, axis1=2, axis2=3)[:,:,None,None] * jnp.eye(problem.dim) + 2*mu*epsilon
        cells_JxW = problem.JxW[:,0,:]
        sigma_average = jnp.sum(sigma * cells_JxW[:,:,None,None], axis=1) / jnp.sum(cells_JxW, axis=1)[:,None,None]

        # Von Mises stress
        s_dev = (sigma_average - 1/problem.dim * jnp.trace(sigma_average, axis1=1, axis2=2)[:,None,None]
                                                * jnp.eye(problem.dim)[None,:,:])
        vm_stress = jnp.sqrt(3./2. * jnp.sum(s_dev*s_dev, axis=(1,2)))

        u = sol_list[0].flatten()
        return u[::5], vm_stress[::100], sigma, epsilon
    
    else:        
        return None

def get_problem(system_type):
    """
        Returns an instance of a problem to be used within a simulation.
        @param system_type: Specifies the exact problem. "linear_elasticity" is implemented first while other system settings can be added within the case statement

        returns: instance of the problem
    """
    if system_type == "linear_elasticity":
        mesh, ele_type, dirichlet_bc_info, location_fns = _mesh_config()

        # Create an instance of the problem.
        problem = LinearElasticity(mesh,
                                vec=3,
                                dim=3,
                                ele_type=ele_type,
                                dirichlet_bc_info=dirichlet_bc_info,
                                location_fns=location_fns)
    else:    
        problem = None
        
    return problem

def _mesh_config():# Specify mesh-related information (second-order tetrahedron element).
    ele_type = 'TET10'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    Lx, Ly, Lz = 10., 2., 2.
    Nx, Ny, Nz = 25, 5, 5
    meshio_mesh = box_mesh_gmsh(Nx=Nx,
                        Ny=Ny,
                        Nz=Nz,
                        Lx=Lx,
                        Ly=Ly,
                        Lz=Lz,
                        data_dir=data_dir,
                        ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


    # Define boundary locations.
    def left(point):
        return jnp.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return jnp.isclose(point[0], Lx, atol=1e-5)


    # Define Dirichlet boundary values.
    # This means on the 'left' side, we apply the function 'zero_dirichlet_val' 
    # to all components of the displacement variable u.
    def zero_dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[left] * 3, [0, 1, 2], [zero_dirichlet_val] * 3]

    # Define Neumann boundary locations.
    # This means on the 'right' side, we will perform the surface integral to get 
    # the tractions with the function 'get_surface_maps' defined in the class 'LinearElasticity'.
    location_fns = [right]

    return mesh, ele_type, dirichlet_bc_info, location_fns