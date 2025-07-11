# Repo for ISAS - Schätzung von Materialparametern

Contains our framework for material parameter estimation.

## Running the PINN

1. Activate the conda environment using: `conda activate jax-fem-env`
2. open pinn directory (paths are static): `cd pinn`
3. start a training run with `python model.py`
4. parameters to be optimized are defined in lines 218ff. 

### TODO: 

- how to get from this to online learning?
  - we need an interface to run the fem simulation
  - do we pretrain the model?
  - how many training steps between simulation runs?

> Answers may be in the [paper](https://www.sciencedirect.com/science/article/abs/pii/S004578252300693X)