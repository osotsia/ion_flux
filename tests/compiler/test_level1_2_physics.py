import pytest
import ion_flux as fx
from ion_flux.compiler.memory import MemoryLayout
from ion_flux.compiler.codegen import generate_cpp

class Level1_Spherical(fx.PDE):
    r = fx.Domain(bounds=(0, 1), resolution=5, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    def math(self):
        return { fx.dt(self.c): fx.div(fx.grad(self.c, axis=self.r), axis=self.r) }

class Level1_FluxBC(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    def math(self):
        flux = -fx.grad(self.c)
        return { fx.dt(self.c): -fx.div(flux), flux.right(): 1.0 }

class Level2_ALE(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    L = fx.State(name="L")
    def math(self):
        return { fx.dt(self.L): 1.0, self.x.right: self.L, fx.dt(self.c): fx.grad(self.c) }

class Level2_MacroMicro(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=3, name="x")
    r = fx.Domain(bounds=(0, 1), resolution=4, name="r")
    macro_micro = x * r
    c = fx.State(domain=macro_micro, name="c")
    def math(self):
        flux_r = fx.grad(self.c, axis=self.r)
        flux_x = fx.grad(self.c, axis=self.x)
        
        return { 
            fx.dt(self.c): fx.div(flux_r, axis=self.r) + fx.div(flux_x, axis=self.x), 
            flux_r.right(domain=self.r): 0.5 
        }

def test_spherical_divergence_emission():
    model, states = Level1_Spherical(), [Level1_Spherical().c]
    cpp = generate_cpp(model.ast(), MemoryLayout(states, []), states)
    assert "(((i) + 1) * dx_r)" in cpp 
    assert "std::max(1e-12" in cpp 

def test_flux_boundary_neumann():
    model, states = Level1_FluxBC(), [Level1_FluxBC().c]
    cpp = generate_cpp(model.ast(), MemoryLayout(states, []), states)
    # The right boundary maps to index 4 with the correct spatial operator dx.
    assert "res[0 + 4] = ((-((y[0 + CLAMP((4) + 1, 5)]) - (y[0 + CLAMP((4) - 1, 5)])) / (2.0 * dx_x))) - (1.0);" in cpp

def test_ale_moving_boundary_injection():
    model, states = Level2_ALE(), [Level2_ALE().c, Level2_ALE().L]
    layout = MemoryLayout(states, [])
    cpp = generate_cpp(model.ast(), layout, states)
    
    # Verify the bounds extraction natively links dx to the changing state L
    assert "double dx_x = y[0] / 4.0;" in cpp
    # Verifies standard ALE grid velocity v_mesh = (x_coord / L)*L_dot injected against grad(c)
    assert "((((i) * dx_x) / std::max(1e-12, y[0])) * ydot[0]" in cpp

def test_macro_micro_nested_loops():
    model, states = Level2_MacroMicro(), [Level2_MacroMicro().c]
    cpp = generate_cpp(model.ast(), MemoryLayout(states, []), states)
    
    # Validation of hierarchical loop unrolling
    assert "for (int i_mac = 0; i_mac < 3; ++i_mac) {" in cpp
    assert "int b_idx = i_mac * 4 + 3;" in cpp
    
    # Validation of cross-product spatial stride calculations
    # Micro axis (r) stride must be 1
    assert "((i) + 1)" in cpp
    assert "((i) - 1)" in cpp
    
    # Macro axis (x) stride must bridge across the inner micro resolution (4)
    assert "((i) + 4)" in cpp
    assert "((i) - 4)" in cpp