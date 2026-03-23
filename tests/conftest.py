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
            "regions": {
                self.rod: [ fx.dt(self.T) == -fx.div(heat_flux) + source ]
            },
            "boundaries": [
                self.T.left == 0.0, 
                self.T.right == 0.0
            ],
            "global": [
                self.T.t0 == 2 * x - x**2
            ]
        }

class CoupledDAE(fx.PDE):
    bulk = fx.Domain(bounds=(0.0, 1.0), resolution=10)
    c = fx.State(domain=bulk)
    V = fx.State(domain=None)
    p_fail = fx.Parameter(default=1.0)
    
    def math(self):
        return {
            "regions": {
                self.bulk: [ fx.dt(self.c) == fx.grad(self.c) ]
            },
            "boundaries": [
                self.c.left == 0.0,
                self.c.right == 0.0
            ],
            "global": [
                self.c.t0 == 1.0,
                self.V.t0 == 0.0,
                # Pure algebraic relationship (no phi == phi - hack)
                self.V == self.c.right / self.p_fail
            ]
        }

@pytest.fixture
def heat_model():
    return TransientHeatPDE()

@pytest.fixture
def dae_model():
    return CoupledDAE()