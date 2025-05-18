import numpy as np
import matplotlib.pyplot as plt

def plot_errors(h_list, rk4_errors, esimm_errors):
    plt.figure(figsize=(8, 6))

    plt.loglog(h_list, rk4_errors, 'o-', label='RK4', markersize=8)
    plt.loglog(h_list, esimm_errors, 's-', label='ESIMM', markersize=8)

    for errors, label, color in zip(
        [rk4_errors, esimm_errors], 
        ['RK4', 'ESIMM'], 
        ['blue', 'orange']
    ):
        log_h = np.log(h_list)
        log_err = np.log(errors)
        slope, intercept = np.polyfit(log_h, log_err, 1)
        fit_errors = np.exp(intercept) * np.array(h_list) ** slope
        plt.loglog(h_list, fit_errors, '--', label=f'{label} trend (p ≈ {slope:.2f})', color=color)

    plt.xlabel('Step size h')
    plt.ylabel('Max error')
    plt.title('Error vs Step size')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def plot_error_vs_time(t, errors, label):
    plt.plot(t, errors, label=label)
    plt.xlabel('t')
    plt.ylabel('Absolute error')
    plt.title('Error vs Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
