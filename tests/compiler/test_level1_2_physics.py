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
    
    # Verify L'Hopital's rule is applied at the origin (r=0) to prevent 0/0 singularities
    assert "((i) == 0 ? (3.0 *" in cpp 
    
    # Verify the expanded divergence formula (grad(c) + 2/r * c) is used for r > 0
    assert "(2.0 / (std::max(1e-12, (double)(i) * dx_r)))" in cpp

def test_flux_boundary_neumann():
    model, states = Level1_FluxBC(), [Level1_FluxBC().c]
    cpp = generate_cpp(model.ast(), MemoryLayout(states, []), states)
    # Verify the ternary operator successfully injects the Neumann BC into the flux evaluation
    assert "((i) == 5 - 1 ? (1.0)" in cpp
    # Verify the bulk gradient evaluation is preserved for the 'else' branch
    assert "(2.0 * dx_x)" in cpp

def test_ale_moving_boundary_injection():
    model, states = Level2_ALE(), [Level2_ALE().c, Level2_ALE().L]
    layout = MemoryLayout(states, [])
    cpp = generate_cpp(model.ast(), layout, states)
    # Verify the bounds extraction natively links dx to the changing state L
    assert "double dx_x = y[0] / 4.0;" in cpp
    # Verify ALE advection velocity evaluates full state expressions, not hardcoded offsets
    assert "dx_x) / std::max(1e-12, (double)(y[" in cpp
    assert "* (ydot[" in cpp

def test_macro_micro_nested_loops():
    model, states = Level2_MacroMicro(), [Level2_MacroMicro().c]
    cpp = generate_cpp(model.ast(), MemoryLayout(states, []), states)
    
    # Validation of hierarchical loop unrolling
    assert "for (int i_mac = 0; i_mac < 3; ++i_mac) {" in cpp
    assert "int i = i_mac * 4 + i_mic;" in cpp
    # Validation of micro Neumann boundary injection mapping with modulo
    assert "((i % 4)) == 4 - 1 ? (0.5)" in cpp