import numpy as np
from scipy.interpolate import interp1d

from .base_solver import BaseODESolver
from .runge_kutta import RungeKutta2, RungeKutta4
from .multistep import AdamsBashforth2, AdamsBashforth4

class ESIMMSolver(BaseODESolver):
    def __init__(self, f, t_span, y0, extrapolation_seq, boot_method='rk4',
                 multistep_method='ab2', h=None):
        super().__init__(f, t_span, y0, h)
        self.extrapolation_seq = extrapolation_seq
        self.boot_method = boot_method
        self.multistep_method = multistep_method

    def _init_bootstrapper(self, method, h, t_eval):
        match method.lower():
            case 'rk2':
                return RungeKutta2(self.f, t_eval, self.y0, h)
            case 'rk4':
                return RungeKutta4(self.f, t_eval, self.y0, h)
            case _:
                raise ValueError(f"Unknown boot method: {method}")

    def _init_multistepper(self, method, h, t_eval, y_boot):
        match method.lower():
            case 'ab2':
                return AdamsBashforth2(self.f, t_eval, y_boot)
            case 'ab4':
                return AdamsBashforth4(self.f, t_eval, y_boot)
            case _:
                raise ValueError(f"Unknown multistep method: {method}")


    def _get_method_order(self):
        match self.multistep_method.lower():
            case 'ab2':
                return 2
            case 'ab4':
                return 4
            case _:
                raise ValueError(f"Unknown multistep method: {self.multistep_method}")

    def _richardson_extrapolation(self, solutions, extrap_seq, order):
        """Apply Richardson extrapolation across multiple approximations."""
        k = len(extrap_seq)
        table = [[None] * k for _ in range(k)]
        for i in range(k):
            table[i][0] = solutions[i]
        for j in range(1, k):
            for i in range(k - j):
                n_j = extrap_seq[i + j]
                n_0 = extrap_seq[i]
                factor = (n_j / n_0) ** order
                table[i][j] = (factor * table[i + 1][j - 1] - table[i][j - 1]) / (factor - 1)
        return table[0][k - 1]

    def solve(self):
        """Solve ODE using extrapolated multistep method."""
        solutions = []
        t_common = None

        for n in self.extrapolation_seq:
            h_eff = self.h / n
            t_eval = np.arange(self.t_span[0], self.t_span[1] + h_eff, h_eff)

            # Runge-Kutta for bootstrap
            boot = self._init_bootstrapper(self.boot_method, h_eff, t_eval)
            _, y_boot = boot.solve()

            # Multistep method
            multistep = self._init_multistepper(self.multistep_method, h_eff, t_eval, y_boot)
            t_full, y_full = multistep.solve()

            if t_common is None:
                t_common = np.linspace(self.t_span[0], self.t_span[1], len(t_full))  # фиксированное количество точек

            # Interpolation on t_common
            interp_func = interp1d(t_full, y_full, axis=0, kind='linear', fill_value="extrapolate")
            y_interp = interp_func(t_common)
            solutions.append(y_interp)

        if len(solutions) == 1:
            return t_common, solutions[0]

        method_order = self._get_method_order()
        y_extrapolated = self._richardson_extrapolation(solutions, self.extrapolation_seq, method_order)

        return t_common, y_extrapolated
