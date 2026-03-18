import pytest
import ion_flux as fx

class TransientHeatPDE(fx.PDE):
    """Simple 1D PDE for testing spatial stencils and compilation."""
    rod = fx.Domain(bounds=(0.0, 2.0), resolution=30)
    T = fx.State(domain=rod)
    k = fx.Parameter(default=0.75)

    def math(self):
        x = self.rod.coords
        heat_flux = -self.k * fx.grad(self.T)
        source = 1 - fx.abs(x - 1)
        return {
            fx.dt(self.T): -fx.div(heat_flux) + source,
            self.T.left: 0.0,
            self.T.right: 0.0,
            self.T.t0: 2 * x - x**2
        }

class CoupledDAE(fx.PDE):
    """System containing both a PDE and an Algebraic constraint (DAE)."""
    bulk = fx.Domain(bounds=(0.0, 1.0), resolution=10)
    c = fx.State(domain=bulk)     # PDE
    V = fx.State(domain=None)     # ODE/Algebraic (Scalar)

    def math(self):
        return {
            fx.dt(self.c): fx.grad(self.c),
            self.c.t0: 1.0,
            self.c.left: 0.0,
            self.c.right: 0.0,
            # No fx.dt(V) -> evaluated as 0 = V - c.right
            self.V: self.V - self.c.right, 
            self.V.t0: 0.0
        }

@pytest.fixture
def heat_model():
    return TransientHeatPDE()

@pytest.fixture
def dae_model():
    return CoupledDAE()