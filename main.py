from multiprocessing import Pool, set_start_method
from experiments.calcs import compute_rk4_error, compute_esimm_error
from core.esimm_solver import ESIMMSolver
from accuracy.error_estimator import estimate_order
from accuracy.benchmarks import benchmarks
from visualization import plot_errors

# Configuration
h_list = [1.6, 0.8, 0.4, 0.2, 0.1, 0.05]
extrapolation_seq = [1, 2, 6]
problem_name = "lorenz"

problem = benchmarks[problem_name]
ode = problem["ode"]
exact = problem["exact"]
y0 = problem["y0"]
t_span = problem["t_span"]


def rk4_wrapper(h):
    return compute_rk4_error(problem_name, h)


def esimm_wrapper(h):
    return compute_esimm_error(problem_name, extrapolation_seq, h)


if __name__ == "__main__":
    set_start_method("spawn", force=True)

    with Pool() as pool:
        rk4_errors = pool.map(rk4_wrapper, h_list)
        esimm_errors = pool.map(esimm_wrapper, h_list)

    esimm_order = estimate_order(
        ode, t_span, y0, exact, ESIMMSolver, h_list, extrapolation_seq=extrapolation_seq
    )

    print(f"ESIMM estimated order: {esimm_order:.2f}")

    plot_errors(h_list, rk4_errors, esimm_errors)
