import numpy as np

def levenberg_marquardt(f, jacobian, x0, y, t, max_iter=100, tol=1e-6, lambda_init=0.01):
    """
    Levenberg-Marquardt algorithm for non-linear least squares.

    Parameters:
        f : callable
            Model function f(x, t), where x is parameter vector, t is input data.
        jacobian : callable
            Jacobian function J(x, t), returns Jacobian matrix of f at x.
        x0 : ndarray
            Initial guess for parameters.
        y : ndarray
            Observed data.
        t : ndarray
            Independent variable.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        lambda_init : float
            Initial damping parameter.

    Returns:
        x : ndarray
            Estimated parameters.
        history : list
            History of parameter vectors.
    """
    x = x0.copy()
    λ = lambda_init
    history = [x.copy()]

    for iteration in range(max_iter):
        r = y - f(x, t)  # residuals
        J = jacobian(x, t)
        H = J.T @ J  # Approximate Hessian
        g = J.T @ r  # Gradient
        delta = np.linalg.solve(H + λ * np.eye(len(x)), g)

        x_new = x + delta
        r_new = y - f(x_new, t)

        if np.linalg.norm(r_new) < np.linalg.norm(r):  # Improvement
            x = x_new
            λ *= 0.7
        else:  # No improvement
            λ *= 2.0

        history.append(x.copy())

        if np.linalg.norm(delta) < tol:
            break

    return x, history