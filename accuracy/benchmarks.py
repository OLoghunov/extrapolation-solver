import numpy as np

def linear_ode(t, y):
    """Linear problem: y' = -y, y(0) = 1."""
    return -y

def exact_linear(t):
    """Exact solution for linear_ode."""
    return np.exp(-t)

def nonlinear_ode(t, y):
    """Non linear problem: y' = y^2, y(0) = 0.5."""
    return y**2

def exact_nonlinear(t):
    """Exact solution for nonlinear_ode."""
    return 1 / (2 - t)