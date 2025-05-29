import numpy as np

"""
    Extended Calman Filter. Implements its predict and update function which are called recursively.
    
    @param f: Nonlinear state transition function
    @param h: Nonlinear measurement function
    @param F_jacobian: Jacobian of f wrt x
    @param H_jacobian: Jacobian of h wrt x
    @param Q: Process noise covariance
    @param R: Measurement noise covariance
    @param x0: Initial state estimate. Stored in x and updated recursively
    @param P0: Initial covariance estimate. Stored in P and updated recursively
"""
class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u=None):
        self.x = self.f(self.x, u)
        F = self.F_jacobian(self.x, u)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ H @ self.P

    def get_state(self):
        return self.x, self.P