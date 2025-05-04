import numpy as np

def compute_error(y_num, y_exact, per_component=False):
    """Calculates maximum absolute error."""
    if per_component:
        return np.max(np.abs(y_num - y_exact), axis=0)
    return np.max(np.abs(y_num - y_exact))

def estimate_order(f, t_span, y0, exact_solution, solver, h_list, **solver_kwargs):
    """
    Evaluates the accuracy order of the method.
    
    :param solver_kwargs: Additional parameters for the solver (extrapolation_seq)
    """
    errors = []
    for h in h_list:
        t_eval = np.arange(t_span[0], t_span[1] + h, h)
        
        solver_obj = solver(f, t_eval, y0, h=h, **solver_kwargs)
        t, y = solver_obj.solve()
        
        y_exact = exact_solution(t)
        errors.append(np.max(np.abs(y - y_exact)))

    orders = [np.log(errors[i]/errors[i+1]) / np.log(h_list[i]/h_list[i+1]) 
              for i in range(len(errors)-1)]
    return np.mean(orders)