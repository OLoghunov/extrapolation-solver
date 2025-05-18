import numpy as np

from core.runge_kutta import RungeKutta4
from core.esimm_solver import ESIMMSolver
from accuracy.error_estimator import compute_error
from accuracy.benchmarks import benchmarks


def compute_rk4_error(problem_name, h):
    problem = benchmarks[problem_name]
    ode = problem["ode"]
    exact = problem["exact"]
    y0 = problem["y0"]
    t_span = problem["t_span"]

    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    solver = RungeKutta4(ode, t_eval, y0, h=h)
    t, y = solver.solve()
    y_true = exact(t)
    return compute_error(y, y_true)


def compute_esimm_error(problem_name, extrapolation_seq, h):
    problem = benchmarks[problem_name]
    ode = problem["ode"]
    exact = problem["exact"]
    y0 = problem["y0"]
    t_span = problem["t_span"]

    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    solver = ESIMMSolver(ode, t_eval, y0, extrapolation_seq=extrapolation_seq, h=h)
    t, y = solver.solve()
    y_true = exact(t)
    return compute_error(y, y_true)
