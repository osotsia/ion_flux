from ion_flux.dsl.core import PDE, State

class DFN(PDE):
    """
    Doyle-Fuller-Newman Model.
    Inherits from PDE so it can be compiled by the Engine.
    """
    V = State()
    
    def __init__(self, thermal: str = "isothermal", dimensions: int = 1, options: dict = None):
        super().__init__()
        self.thermal = thermal
        self.dimensions = dimensions
        self.options = options or {}

    def math(self):
        # Stub math for tests. In reality, this contains the full DFN equations.
        return {
            self.V: self.V - 2.5
        }

class SPM(PDE):
    """Single Particle Model."""
    V = State()
    
    def math(self):
        return {
            self.V: self.V - 3.0
        }