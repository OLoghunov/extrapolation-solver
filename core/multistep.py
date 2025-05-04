import numpy as np
from .base_solver import BaseODESolver
from .runge_kutta import RungeKutta2, RungeKutta4

class AdamsBashforth2(BaseODESolver):
    """Adams-Bashforth second order method with RK2 bootstrap."""
    
    def solve(self):
        t = np.arange(self.t0, self.tf + self.h, self.h)
        y = np.zeros((len(t), len(self.y0)))
        y[0] = self.y0
        
        rk_solver = RungeKutta2(self.f, [t[0], t[1]], y[0], h=self.h)
        _, y_rk = rk_solver.solve()
        y[1] = y_rk[-1]
        
        # Main cycle
        for i in range(1, len(t) - 1):
            h = t[i+1] - t[i]
            y[i+1] = y[i] + h * (3/2 * self.f(t[i], y[i]) - 1/2 * self.f(t[i-1], y[i-1]))
        
        return t, y

class AdamsBashforth4(BaseODESolver):
    """Adams-Bashforth forth order method with RK4 bootstrap."""
    
    def solve(self):
        t = np.arange(self.t0, self.tf + self.h, self.h)
        y = np.zeros((len(t), len(self.y0)))
        y[0] = self.y0
        
        rk_solver = RungeKutta4(self.f, t[:4], y[0], h=self.h)
        _, y_rk = rk_solver.solve()
        y[1:4] = y_rk[1:4]
        
        # Main cycle
        for i in range(3, len(t) - 1):
            h = t[i+1] - t[i]
            y[i+1] = y[i] + h * (
                55/24 * self.f(t[i], y[i]) 
                - 59/24 * self.f(t[i-1], y[i-1]) 
                + 37/24 * self.f(t[i-2], y[i-2]) 
                - 9/24 * self.f(t[i-3], y[i-3])
            )
        
        return t, y