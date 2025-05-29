import numpy as np
from ekf.ekf import ExtendedKalmanFilter
import matplotlib.pyplot as plt

def f(x, u):
    dt = 1.0
    pos, vel = x
    acc = u if u is not None else 0
    return np.array([pos + vel * dt + 0.5 * acc * dt**2,
                     vel + acc * dt])

# Measurement: z = h(x)
def h(x):
    return np.array([x[0]])  # we only measure position

# Jacobians
def F_jacobian(x, u):
    dt = 1.0
    return np.array([
        [1, dt],
        [0, 1]
    ])

def H_jacobian(x):
    return np.array([[1, 0]])

"""
    Initializes and runs an experiment set up as follows:
    We localize the movement of a 1D-robot. The system model is given by:
        A state: Position and Velocity (pos, vel)
        Control input: Acceleration (u)
        Measurement: Noisy position
    
    return: 
        true_states: The actual movement of the robot, which develops linearly
        estimated_states: The position of the robot on the y-axis dependent on each time step
        measurements: Noise, obtained by adding a random error on the true positional states
"""
def run_test():
    # Initial state and covariance
    x0 = np.array([0, 1])  # initial pos=0, vel=1
    P0 = np.eye(2)

    # Covariance matrices
    Q = np.array([[0.1, 0], [0, 0.1]])
    R = np.array([[1]])

    # Create EKF
    ekf = ExtendedKalmanFilter(f, h, F_jacobian, H_jacobian, Q, R, x0, P0)

    # Simulate some measurements
    true_states = []
    estimated_states = []
    measurements = []

    for t in range(20):
        # Simulate true motion
        true_pos = x0[0] + x0[1] * t
        z = np.array([true_pos + np.random.normal(0, 1)])  # measurement with noise
        
        ekf.predict()
        ekf.update(z)

        x_est, _ = ekf.get_state()
        
        true_states.append(true_pos)
        estimated_states.append(x_est[0])
        measurements.append(z[0])
    return true_states, estimated_states, measurements

def plot_results(true_states, estimated_states, measurements):
    plt.plot(true_states, label="True Position")
    plt.plot(estimated_states, label="EKF Estimated Position")
    plt.plot(measurements, label="Measurements", linestyle='dotted')
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.title("EKF Position Estimation")
    plt.show()

if __name__ == "__main__":
    true_states, estimated_states, measurements = run_test()
    plot_results(true_states, estimated_states, measurements)
