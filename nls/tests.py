from nls import levenberg_marquardt
import numpy as np

# Define model function
def model(x, t):
    return x[0] * np.exp(x[1] * t)

# Define Jacobian
def jacobian(x, t):
    J = np.zeros((len(t), len(x)))
    J[:, 0] = np.exp(x[1] * t)
    J[:, 1] = x[0] * t * np.exp(x[1] * t)
    return J

# Generate synthetic data
np.random.seed(0)
t_data = np.linspace(0, 1, 10)
true_params = [2.5, -1.3]
y_data = model(true_params, t_data) + 0.05 * np.random.randn(len(t_data))

# Initial guess
x0 = np.array([1.0, 0.0])

# Fit using Levenberg-Marquardt
x_est, history = levenberg_marquardt(model, jacobian, x0, y_data, t_data)

print("Estimated parameters:", x_est)