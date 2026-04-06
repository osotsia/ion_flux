"""
test_piecewise_micro_routing_oracle.py

Compiler Bug Oracle: Global Piecewise to Micro-Domain Index Routing

This suite proves that when a micro-domain surface boundary (c_surf) is evaluated 
inside a global Piecewise block (e.g. for the electrolyte c_e), the loop variable `idx` 
is a global topological index (e.g. 81-143). The compiler incorrectly routes this 
directly into the micro-domain's local flat array, resulting in severe data corruption.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

class PiecewiseMicroRoutingOracle(fx.PDE):
    # Mimics O'Regan LG M50 topology
    cell = fx.Domain(bounds=(0, 100), resolution=144)
    x_n = cell.region(bounds=(0, 40), resolution=71, name="x_n")
    x_s = cell.region(bounds=(40, 50), resolution=10, name="x_s")
    x_p = cell.region(bounds=(50, 100), resolution=63, name="x_p")
    
    r_p = fx.Domain(bounds=(0, 5), resolution=10, name="r_p")
    
    # State bound to the sub-mesh composite domain (Size: 63 * 10 = 630)
    c_s_p = fx.State(domain=x_p * r_p, name="c_s_p")
    
    # State bound to the global cell (Size: 144)
    c_e = fx.State(domain=cell, name="c_e")
    
    def math(self):
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p)
        
        return {
            "equations": {
                # Lock original state
                self.c_s_p: fx.dt(self.c_s_p) == 0.0,
                
                # Expose the evaluation of c_surf_p inside a global Piecewise loop
                self.c_e: fx.Piecewise({
                    self.x_n: fx.dt(self.c_e) == 0.0,
                    self.x_s: fx.dt(self.c_e) == 0.0,
                    
                    # Inside the x_p loop, the C++ 'idx' goes from 81 to 143.
                    # We store c_surf_p directly into c_e's derivative to extract it.
                    self.x_p: fx.dt(self.c_e) == c_surf_p
                })
            },
            "boundaries": {},
            "initial_conditions": {
                self.c_s_p: 0.0,
                self.c_e: 0.0
            }
        }

@REQUIRES_COMPILER
def test_piecewise_micro_domain_routing():
    engine = Engine(model=PiecewiseMicroRoutingOracle(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y0, ydot0, _, _, _ = engine._extract_metadata()
    y0 = np.array(y0)
    
    # Create a distinct value for EVERY micro node so we know exactly which one is read
    off_csp, size_csp = engine.layout.state_offsets["c_s_p"]
    y0[off_csp : off_csp + size_csp] = np.arange(size_csp)
    
    # The true physical surface values are at local indices 9, 19, 29, ..., 629
    true_surface_vals = np.arange(9, 630, 10)
    
    res = engine.evaluate_residual(y0.tolist(), ydot0, parameters={})
    
    # Extract the values evaluated during the Piecewise loop
    off_ce, size_ce = engine.layout.state_offsets["c_e"]
    
    # The residual equation is F = ydot - rhs = 0.0 - c_surf_p -> rhs = -F
    extracted_surfaces = -np.array(res[off_ce + 81 : off_ce + 144])
    
    # If the compiler is correct, the piecewise loop fetched the exact same 
    # array of surface nodes (9, 19, ..., 629).
    # If the bug is active, it fetched indices like 89, 89, 99, 99, ..., 149.
    is_corrupted = not np.allclose(extracted_surfaces, true_surface_vals)
    
    assert is_corrupted, \
        "The bug was NOT detected. The compiler successfully routed the indices."
        
    if is_corrupted:
        pytest.fail(
            f"\nBUG CONFIRMED: The AST compiler incorrectly routed the macro-to-micro "
            f"surface extraction inside a Piecewise loop!\n"
            f"Expected: {true_surface_vals[:5]}...\n"
            f"Actual:   {extracted_surfaces[:5]}...\n"
            f"This destroys the electrolyte concentration gradient inside the cathode."
        )

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])