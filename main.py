import numpy as np
from core.runge_kutta import RungeKutta4
from core.esimm_solver import ESIMMSolver
from accuracy.benchmarks import linear_ode, exact_linear
from accuracy.error_estimator import estimate_order
import matplotlib.pyplot as plt

t_span = [0, 1]
y0 = 1
h = 0.1

solvers = {
    "RK4": RungeKutta4(linear_ode, t_span, y0, h),
    "ESIMM": ESIMMSolver(
        linear_ode, t_span, y0, extrapolation_seq=[1, 2], boot_method="rk4", h=h
    ),
}

for name, solver in solvers.items():
    t, y = solver.solve()
    y_exact = exact_linear(t)
    error = np.max(np.abs(y - y_exact))
    print(f"{name}: error = {error:.2e}")

h_list = [0.1, 0.05, 0.025]

esimm_params = {
    'extrapolation_seq': [1, 2],
    'boot_method': 'rk4',
    'multistep_method': 'ab2'
}

order_esimm = estimate_order(
    linear_ode,
    t_span,
    y0,
    exact_linear,
    solver=ESIMMSolver,
    h_list=h_list,
    **esimm_params,
)

print(f"ESIMM order of accuracy: {order_esimm:.2f}")

plt.loglog(t, np.abs(y - y_exact), 'o-', label='ESIMM')
plt.xlabel('Step (h)')
plt.ylabel('Error')
plt.legend()
plt.show()