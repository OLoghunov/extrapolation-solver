import numpy as np
from core.runge_kutta import RungeKutta4
from core.esimm_solver import ESIMMSolver
from accuracy.error_estimator import compute_error, estimate_order
from accuracy.benchmarks import benchmarks
from visualization import plot_errors

# Config
h_list = [0.4, 0.2, 0.1, 0.05]
extrapolation_seq = [1, 2, 4]
problem = benchmarks["lorenz"]

ode = problem["ode"]
exact = problem["exact"]
y0 = problem["y0"]
t_span = problem["t_span"]

# Error estimation RK4
rk4_errors = []
for h in h_list:
    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    solver = RungeKutta4(ode, t_eval, y0, h=h)
    t, y = solver.solve()
    y_true = exact(t)
    rk4_errors.append(compute_error(y, y_true))

# Error estimation ESIMM
esimm_errors = []
for h in h_list:
    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    solver = ESIMMSolver(ode, t_eval, y0, extrapolation_seq, h=h)
    t, y = solver.solve()
    y_true = exact(t)
    esimm_errors.append(compute_error(y, y_true))

# ESIMM order estimation
esimm_order = estimate_order(
    ode, t_span, y0, exact, 
    ESIMMSolver, h_list,
    extrapolation_seq=extrapolation_seq
)

print(f"ESIMM estimated order: {esimm_order:.2f}")

# Plot
plot_errors(h_list, rk4_errors, esimm_errors)
