# --- File: tests/02_middle_end_codegen/test_spatial_topologies.py ---
"""
Middle-End Codegen: Spatial Topologies

Tests the generation of correct C++ memory strides, coordinate systems, and
hierarchical loop unrolling (e.g., 1D spherical, macro-micro cross-products).
"""

import pytest
import ion_flux as fx
from ion_flux.compiler.memory import MemoryLayout
from ion_flux.compiler.codegen import generate_cpp

class Level2_ALE(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    L = fx.State(name="L")
    def math(self):
        return {
            "regions": {
                self.x: [ fx.dt(self.c) == fx.grad(self.c) ]
            },
            "global": [
                fx.dt(self.L) == 1.0
            ],
            "boundaries": [
                self.x.right == self.L
            ]
        }

class Level2_MacroMicro(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=3, name="x")
    r = fx.Domain(bounds=(0, 1), resolution=4, name="r")
    macro_micro = x * r
    c = fx.State(domain=macro_micro, name="c")
    def math(self):
        flux_r = fx.grad(self.c, axis=self.r)
        flux_x = fx.grad(self.c, axis=self.x)
        
        return { 
            "regions": {
                self.macro_micro: [
                    fx.dt(self.c) == fx.div(flux_r, axis=self.r) + fx.div(flux_x, axis=self.x)
                ]
            },
            "boundaries": [
                flux_r.right(domain=self.r) == 0.5 
            ]
        }

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