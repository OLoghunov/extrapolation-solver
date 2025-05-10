import numpy as np
from .base_solver import BaseODESolver
from .runge_kutta import RungeKutta2, RungeKutta4

class AdamsBashforth2(BaseODESolver):
    def __init__(self, f, t_eval, y_boot):
        super().__init__(f, [t_eval[0], t_eval[-1]], y_boot[0], t_eval[1] - t_eval[0])
        self.t_eval = t_eval
        self.y_boot = y_boot

    def solve(self):
        t = self.t_eval
        y = np.zeros((len(t), len(self.y0)))
        y[:2] = self.y_boot[:2]

        for i in range(1, len(t) - 1):
            h = t[i+1] - t[i]
            y[i+1] = y[i] + h * (
                3/2 * self.f(t[i], y[i]) - 1/2 * self.f(t[i-1], y[i-1])
            )

        return t, y

class AdamsBashforth4(BaseODESolver):
    def __init__(self, f, t_eval, y_boot):
        super().__init__(f, [t_eval[0], t_eval[-1]], y_boot[0], t_eval[1] - t_eval[0])
        self.t_eval = t_eval
        self.y_boot = y_boot

    def solve(self):
        t = self.t_eval
        y = np.zeros((len(t), len(self.y0)))
        y[:4] = self.y_boot[:4]

        for i in range(3, len(t) - 1):
            h = t[i+1] - t[i]
            y[i+1] = y[i] + h * (
                55/24 * self.f(t[i], y[i])
                - 59/24 * self.f(t[i-1], y[i-1])
                + 37/24 * self.f(t[i-2], y[i-2])
                - 9/24 * self.f(t[i-3], y[i-3])
            )

        return t, y
