# Physics-Informed Neural Network for Hyperelastic Beam Parameter Optimization
import jax
import jax.numpy as np
import optax
import os
from functools import partial

# Import JAX-FEM specific modules
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh


class HyperElasticityPINN(Problem):
    """
    Hyperelastic problem class with parameterized material properties.
    This allows the PINN to optimize the material parameters E and nu.
    """
    
    def __init__(self, mesh, E, nu, **kwargs):
        # Store material parameters as instance variables
        self.E_param = E
        self.nu_param = nu
        super().__init__(mesh, **kwargs)
    
    def get_tensor_map(self):
        """
        Define the constitutive relationship with parameterized material properties.
        The PINN will optimize E and nu to match target displacement patterns.
        """
        def psi(F, E, nu):
            # Convert material parameters to Lamé parameters
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            
            # Compute deformation invariants
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            
            # Neo-Hookean strain energy density
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        # Create gradients for first Piola-Kirchhoff stress
        P_fn = jax.grad(psi, argnums=0)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, self.E_param, self.nu_param)
            return P

        return first_PK_stress


def create_mesh_and_bc():
    """
    Create the mesh and boundary conditions for the hyperelastic beam problem.
    Returns mesh and boundary condition information.
    """
    # Mesh parameters (using smaller mesh for faster PINN training)
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    Lx, Ly, Lz = 1., 1., 1.
    
    # Generate mesh with reduced resolution for faster computation
    meshio_mesh = box_mesh_gmsh(Nx=10, Ny=10, Nz=10,
                               Lx=Lx, Ly=Ly, Lz=Lz,
                               data_dir=data_dir,
                               ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    
    # Define boundary locations
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
    
    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)
    
    # Define Dirichlet boundary values
    def zero_dirichlet_val(point):
        return 0.
    
    def dirichlet_val_x2(point):
        return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
                (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.
    
    def dirichlet_val_x3(point):
        return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
                (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.
    
    # Assemble boundary condition information
    dirichlet_bc_info = [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                         [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] +
                         [zero_dirichlet_val] * 3]
    
    return mesh, dirichlet_bc_info


def solve_fem_problem(params, mesh, dirichlet_bc_info):
    """
    Solve the FEM problem with given material parameters.
    
    Args:
        params: Dictionary containing E and nu values
        mesh: FEM mesh
        dirichlet_bc_info: Boundary condition information
    
    Returns:
        Displacement solution array
    """
    E, nu = params['E'], params['nu']
    
    # Create problem instance with current parameters
    problem = HyperElasticityPINN(mesh, E, nu,
                                  vec=3, dim=3, ele_type='HEX8',
                                  dirichlet_bc_info=dirichlet_bc_info)
    
    # Solve the problem
    sol_list = solver(problem, solver_options={'petsc_solver': {}})
    
    return sol_list[0]  # Return displacement solution


def compute_loss(params, mesh, dirichlet_bc_info, target_displacement):
    """
    Compute the physics-informed loss function.
    This function measures how well the current parameters reproduce target behavior.
    
    Args:
        params: Material parameters to optimize
        mesh: FEM mesh
        dirichlet_bc_info: Boundary conditions
        target_displacement: Target displacement field for comparison
    
    Returns:
        Loss value (scalar)
    """
    # Solve FEM with current parameters
    predicted_displacement = solve_fem_problem(params, mesh, dirichlet_bc_info)
    
    # Compute L2 norm difference between predicted and target displacements
    loss = np.mean((predicted_displacement - target_displacement)**2)
    
    # Add parameter regularization to keep E and nu in physically reasonable ranges
    E_penalty = np.maximum(0., -params['E'] + 1e-3)  # E should be positive
    nu_penalty = np.maximum(0., np.abs(params['nu']) - 0.49)  # |nu| < 0.5 for stability
    
    total_loss = loss + 1e-3 * (E_penalty + nu_penalty)
    
    return total_loss


def generate_target_data(mesh, dirichlet_bc_info):
    """
    Generate target displacement data using known material parameters.
    In practice, this would be experimental or reference simulation data.
    """
    print("Generating target data with reference parameters...")
    
    # Reference parameters (these are the "true" values we want to recover)
    true_params = {'E': 15.0, 'nu': 0.25}
    
    # Solve with reference parameters to get target displacement
    target_displacement = solve_fem_problem(true_params, mesh, dirichlet_bc_info)
    
    print(f"Target generated with E={true_params['E']}, nu={true_params['nu']}")
    return target_displacement, true_params


def pinn_optimization():
    """
    Main PINN optimization loop for material parameter identification.
    """
    print("Starting PINN optimization for hyperelastic beam parameters...")
    
    # Setup mesh and boundary conditions
    mesh, dirichlet_bc_info = create_mesh_and_bc()
    print(f"Mesh created with {len(mesh.points)} nodes")
    
    # Generate target data (normally this would be experimental data)
    target_displacement, true_params = generate_target_data(mesh, dirichlet_bc_info)
    
    # Initialize parameters to be optimized (starting guess)
    initial_params = {'E': 8.0, 'nu': 0.35}  # Initial guess different from true values
    print(f"Starting optimization from E={initial_params['E']}, nu={initial_params['nu']}")
    
    # Setup optimizer (Adam with learning rate scheduling)
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)
    
    # JIT compile the loss function for faster execution
    @jax.jit
    def loss_and_grad(params):
        loss_fn = lambda p: compute_loss(p, mesh, dirichlet_bc_info, target_displacement)
        loss, grad = jax.value_and_grad(loss_fn)(params)
        return loss, grad
    
    # Optimization loop
    params = initial_params.copy()
    num_epochs = 50  # Sensible number of iterations for demonstration
    
    print("\nStarting optimization...")
    print("Epoch | Loss      | E        | nu      | Error E  | Error nu")
    print("-" * 65)
    
    for epoch in range(num_epochs):
        # Compute loss and gradients
        loss, grads = loss_and_grad(params)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Compute errors relative to true parameters
        error_E = abs(params['E'] - true_params['E'])
        error_nu = abs(params['nu'] - true_params['nu'])
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"{epoch:5d} | {loss:.6f} | {params['E']:8.4f} | {params['nu']:7.4f} | "
                  f"{error_E:8.4f} | {error_nu:8.4f}")
    
    print("\nOptimization completed!")
    print(f"True parameters:      E = {true_params['E']:.4f}, nu = {true_params['nu']:.4f}")
    print(f"Optimized parameters: E = {params['E']:.4f}, nu = {params['nu']:.4f}")
    print(f"Final errors:         E = {error_E:.6f}, nu = {error_nu:.6f}")
    
    return params, true_params


if __name__ == "__main__":
    # Run the PINN optimization
    optimized_params, true_params = pinn_optimization()
