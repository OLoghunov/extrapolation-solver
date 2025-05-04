import numpy as np
from .base_solver import BaseODESolver

class RungeKutta4(BaseODESolver):
    """RK4 method."""
    
    def solve(self):
        t = np.arange(self.t0, self.tf + self.h, self.h)
        y = np.zeros((len(t), len(self.y0)))
        y[0] = self.y0
        
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            k1 = h * self.f(t[i], y[i])
            k2 = h * self.f(t[i] + h/2, y[i] + k1/2)
            k3 = h * self.f(t[i] + h/2, y[i] + k2/2)
            k4 = h * self.f(t[i] + h, y[i] + k3)
            y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return t, y

class RungeKutta2(BaseODESolver):
    """RK2 method."""
    
    def solve(self):
        t = np.arange(self.t0, self.tf + self.h, self.h)
        y = np.zeros((len(t), len(self.y0)))
        y[0] = self.y0
        
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            k1 = h * self.f(t[i], y[i])
            k2 = h * self.f(t[i] + h, y[i] + k1)
            y[i+1] = y[i] + (k1 + k2) / 2
        
        return t, y