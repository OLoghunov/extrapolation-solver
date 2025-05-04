from abc import ABC, abstractmethod

class BaseODESolver(ABC):
    """The base class for all numerical ODE solvers."""
    
    def __init__(self, f, t_span, y0, h=None):
        """
        :param f: ODE function dy/dt = f(t, y)
        :param t_span: Interval [t0, tf]
        :param y0: Initial conditions y(t0) = y0
        :param h: Step (optional)
        """
        self.f = f
        self.t_span = t_span
        self.y0 = y0
        self.t0, self.tf = t_span[0], t_span[-1]
        self.h = h if h is not None else (t_span[1] - t_span[0])
        
    @abstractmethod
    def solve(self):
        """Solves the ODE and returns an array of time and solutions."""
        pass