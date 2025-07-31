# Repo for ISAS - Schätzung von Materialparametern

Contains our framework for material parameter estimation.

## Running the PINN

1. Activate the conda environment using: `conda activate jax-fem-env`
2. open pinn directory (paths are static): `cd pinn`
3. start a training run with `python model.py`
4. parameters to be optimized are defined in lines 218ff. 

# Linear Elasticity Simulation Framework

This repository provides tools and models for simulating linear elasticity problems and performing parameter estimation using methods such as EKF, NLS, and PINNs.

---

## Project Structure

### `LinearElasticity/`
Contains resources for simulations of linear elasticity.  
In `LinearElasticity/problem.py`, we define a problem setup consistent with the equations governing linear elastic systems.

---

## Implemented Models

### EKF (Extended Kalman Filter)
- **Core implementation**: `ekf/ekf.py`
- **Linear elasticity-specific implementation**: `ekf/LinearElasticityEKF.py`
- **Tests**: `ekf/tests.py` — validates functionality with a toy example.
- **Simulation run**: `ekf/main_ekf.py` — performs material parameter estimation using the `LinearElasticityEKF`.

### NLS (Nonlinear Least Squares - Levenberg-Marquardt)
- **Implementation**: `nls/nls_optx.py`
- **Tests**: `nls/tests_optx.py`
- **Simulation run**: `nls/main_nls.py` — estimates material parameters using the `nls_optx` implementation.

### PINNs (Physics-Informed Neural Networks)
- **Model**: `pinn/model.py`
- **Training script**: `pinn/iterative_trainer.py`

---

## Supporting Files & Utilities

- **`plots/`**  
  Stores plots generated during experiments. These are referenced in the corresponding results chapter of your report or paper.

- **`Dockerfile`**  
  Container setup for managing dependencies, especially [`jax-fem`](https://github.com/deepmodeling/jax-fem).  
  After building and starting the container, run the following setup commands:

  ```bash
  git clone git@github.com:deepmodeling/jax-fem.git
  cd jax-fem/
  conda env create -f environment.yml -p ./.conda
  conda activate ./.conda
  pip install -U "jax[cuda12]"
  pip install -e .

 - ** `util.py`**
   Provides support-functions such as \texttt{run\_and\_solve} which runs a given simulation, \texttt{get\_problem}  which creates an instance of a specified problem
