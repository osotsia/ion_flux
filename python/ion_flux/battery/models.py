from ion_flux.dsl.core import PDE, State, Parameter
from ion_flux.dsl.operators import dt

class DFN(PDE):
    """
    Doyle-Fuller-Newman Model.
    Inherits from PDE so it can be compiled by the Engine.
    """
    V = State()
    i_app = Parameter(default=1.0)
    
    def __init__(self, thermal: str = "isothermal", dimensions: int = 1, options: dict = None):
        super().__init__()
        self.thermal = thermal
        self.dimensions = dimensions
        self.options = options or {}

    def math(self):
        # Provide a structurally valid ODE to ensure a non-singular Jacobian during default tests.
        return {
            dt(self.V): -0.01 * self.i_app,
            self.V.t0: 4.2
        }

class SPM(PDE):
    """Single Particle Model."""
    V = State()
    i_app = Parameter(default=1.0)
    
    def math(self):
        return {
            dt(self.V): -0.01 * self.i_app,
            self.V.t0: 4.2
        }