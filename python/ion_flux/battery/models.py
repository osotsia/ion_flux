from ion_flux.dsl.core import PDE, State, Parameter, Terminal
from ion_flux.dsl.operators import dt

class SPM(PDE):
    """Single Particle Model."""
    V = State()
    i_app = State()
    terminal = Terminal(current=i_app, voltage=V)
    
    def math(self):
        return {
            dt(self.V): -0.01 * self.i_app,
            self.V.t0: 4.2,
            self.i_app.t0: 1.0
        }

class DFN(PDE):
    """
    Doyle-Fuller-Newman Model placeholder for performance benchmarking.
    Inherits from PDE so it can be compiled by the Engine.
    Uses Terminal abstraction for dynamic protocol execution.
    """
    V = State()
    i_app = State()
    terminal = Terminal(current=i_app, voltage=V)
    
    def __init__(self, thermal: str = "isothermal", dimensions: int = 1, options: dict = None):
        super().__init__()
        self.thermal = thermal
        self.dimensions = dimensions
        self.options = options or {}

    def math(self):
        # Provide a structurally valid ODE to ensure a non-singular Jacobian during default tests.
        return {
            dt(self.V): -0.01 * self.i_app,
            self.V.t0: 4.2,
            self.i_app.t0: 1.0
        }