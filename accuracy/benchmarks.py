import numpy as np
from core.runge_kutta import RungeKutta4

def linear_ode(t, y):
    return -y

def exact_linear(t):
    return np.exp(-t)

def nonlinear_ode(t, y):
    return y * (1 - y)

def exact_nonlinear(t):
    return 1 / (1 + np.exp(-t))

def lorenz_system(t, y, sigma=10, rho=28, beta=8/3):
    dydt = np.zeros(3)
    dydt[0] = sigma * (y[1] - y[0])
    dydt[1] = y[0] * (rho - y[2]) - y[1]
    dydt[2] = y[0] * y[1] - beta * y[2]
    return dydt

# Config for Lorenz system
t_span = (0.0, 5.0)
y0 = np.array([1.0, 1.0, 1.0])

# "Pseudo-accurate" solution of the Lorentz system obtained by RK4 with a very small step
def exact_lorenz(t_array):
    h_ref = 1e-5
    t_eval = np.arange(t_span[0], t_span[1] + h_ref, h_ref)
    solver = RungeKutta4(lorenz_system, t_eval, y0, h=h_ref)
    t_ref, y_ref = solver.solve()

    # Time interpolation
    return np.array([np.interp(t_array, t_ref, y_ref[:, i]) for i in range(3)]).T

benchmarks = {
    "linear": {
        "ode": linear_ode,
        "exact": exact_linear,
        "y0": 1.0,
        "t_span": [0.0, 1.5]
    },
    "nonlinear": {
        "ode": nonlinear_ode,
        "exact": exact_nonlinear,
        "y0": 0.5,
        "t_span": [0.0, 5.0]
    },
    "lorenz": {
        "ode": lorenz_system,
        "exact": exact_lorenz,
        "y0": y0,
        "t_span": t_span
    }
}
