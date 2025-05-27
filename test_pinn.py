#!/usr/bin/env python3
# Simple test script to verify PINN setup

import jax
import jax.numpy as jnp
import optax
import numpy as np

print("Testing PINN setup...")
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")

# Test basic JAX functionality
x = jnp.array([1.0, 2.0, 3.0])
print(f"JAX array: {x}")

# Test gradient computation
def simple_loss(param):
    return param**2

grad_fn = jax.grad(simple_loss)
gradient = grad_fn(5.0)
print(f"Gradient of x^2 at x=5: {gradient}")

# Test optax
optimizer = optax.adam(0.1)
print("Optax working ✓")

print("All tests passed! Ready for PINN.")
