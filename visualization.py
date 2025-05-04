import matplotlib.pyplot as plt

def plot_errors(h_list, rk4_errors, esimm_errors):
    plt.figure(figsize=(8, 6))
    plt.loglog(h_list, rk4_errors, 'o-', label='RK4')
    plt.loglog(h_list, esimm_errors, 's-', label='ESIMM')
    plt.xlabel('Step size h')
    plt.ylabel('Max error')
    plt.title('Error vs Step size')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()
