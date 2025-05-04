import numpy as np
from .base_solver import BaseODESolver
from .runge_kutta import RungeKutta2, RungeKutta4
from .multistep import AdamsBashforth2, AdamsBashforth4

class ESIMMSolver(BaseODESolver):
    """An extrapolation multistep solver.
    
    Parameters:
        f: ODE function dy/dt = f(t, y)
        t_span: Interval [t0, tf]
        y0: Initial conditions
        extrapolation_seq: Extrapolation sequence (example: [1, 2, 4])
        boot_method: Boot method ('rk2', 'rk4')
        multistep_method: Multistep method ('ab2', 'ab4')
        h: Step
    """
    def __init__(self, f, t_span, y0, extrapolation_seq, boot_method='rk4', 
                 multistep_method='ab2', h=None):
        super().__init__(f, t_span, y0, h)
        self.extrapolation_seq = extrapolation_seq
        
        self.bootstrapper = self._init_bootstrapper(boot_method)
        
        self.multistepper = self._init_multistepper(multistep_method)

    def _init_bootstrapper(self, method):
        """Return initialized bootstrapper."""
        match method.lower():
            case 'rk2':
                return RungeKutta2(self.f, self.t_span, self.y0, self.h)
            case 'rk4':
                return RungeKutta4(self.f, self.t_span, self.y0, self.h)
            case _:
                raise ValueError(f"Unknown boot method: {method}")

    def _init_multistepper(self, method):
        """Return initialized multistep solver."""
        match method.lower():
            case 'ab2':
                return AdamsBashforth2(self.f, self.t_span, self.y0, self.h)
            case 'ab4':
                return AdamsBashforth4(self.f, self.t_span, self.y0, self.h)
            case _:
                raise ValueError(f"Unknown multistep method: {method}")

    def solve(self):
        """Performs calculations with extrapolation."""
        
        t_boot, y_boot = self.bootstrapper.solve()
        
        t_full, y_full = self.multistepper.solve()
        
        if len(self.extrapolation_seq) > 1:
            # Temporary - averaging
            y_extrapolated = self._apply_extrapolation(t_full, y_full, t_boot, y_boot)
            return t_full, y_extrapolated
        
        return t_full, y_full

    def _apply_extrapolation(self, t_full, y_full, t_boot, y_boot):
        """Applying extrapolation."""
        # Temporary - averaging
        y_extrapolated = np.zeros_like(y_full)
        for i in range(len(y_full)):
            weight = 0.7 if i < len(y_boot) else 1.0
            y_extrapolated[i] = weight * y_full[i] + (1 - weight) * y_boot[min(i, len(y_boot)-1)]
        return y_extrapolated