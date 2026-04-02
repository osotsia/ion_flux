import pytest
import ion_flux as fx

class TransientHeatPDE(fx.PDE):
    rod = fx.Domain(bounds=(0.0, 2.0), resolution=30)
    T = fx.State(domain=rod)
    k = fx.Parameter(default=0.75)
    
    def math(self):
        x = self.rod.coords
        heat_flux = -self.k * fx.grad(self.T)
        source = 1 - fx.abs(x - 1)
        
        return {
            "equations": {
                self.T: fx.dt(self.T) == -fx.div(heat_flux) + source
            },
            "boundaries": {
                self.T: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.T: 2 * x - x**2
            }
        }

class CoupledDAE(fx.PDE):
    bulk = fx.Domain(bounds=(0.0, 1.0), resolution=10)
    c = fx.State(domain=bulk)
    V = fx.State(domain=None)
    p_fail = fx.Parameter(default=1.0)
    
    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == fx.grad(self.c),
                self.V: self.V == self.c.right / self.p_fail
            },
            "boundaries": {
                self.c: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c: 1.0,
                self.V: 0.0
            }
        }

@pytest.fixture
def heat_model():
    return TransientHeatPDE()

@pytest.fixture
def dae_model():
    return CoupledDAE()