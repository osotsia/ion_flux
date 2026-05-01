"""
Compiler Bug Oracle: The Phantom Parent

This suite proves that the compiler's fallback `MockDomain` (used when 
integrating over a domain with no bound states) lacks topological hierarchy.
This causes cross-domain memory fetches to drop their `start_idx` offsets, 
silently reading the wrong memory addresses.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

# ==============================================================================
# Environment Setup
# ==============================================================================

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires C++ compiler.")

# ==============================================================================
# Bug Isolation Model
# ==============================================================================

class MissingParentIntegrationOracle(fx.PDE):
    # A parent domain split into two halves
    cell = fx.Domain(bounds=(0, 2.0), resolution=20)
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 2.0), resolution=10, name="reg_B")
    
    # State bound to the global parent
    c_parent = fx.State(domain=cell, name="c_parent")
    
    # 0D target to hold the integral evaluation
    int_val = fx.State(domain=None, name="int_val")

    def math(self):
        return {
            "equations": {
                self.c_parent: fx.dt(self.c_parent) == 0.0,
                
                # We integrate the parent state ONLY over the second half (reg_B).
                # Crucially, NO state is bound to reg_B, forcing the compiler 
                # to use the fallback `MockDomain`.
                self.int_val: self.int_val == fx.integral(self.c_parent, over=self.reg_B)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c_parent: 0.0, 
                self.int_val: 0.0
            }
        }

# ==============================================================================
# The Test
# ==============================================================================

@REQUIRES_COMPILER
def test_unbound_subregion_memory_mapping():
    """
    If the fallback MockDomain lacks a `.parent` attribute, the integral over 
    reg_B will fail to add the `start_idx` offset (10). It will silently read 
    indices 0-9 (reg_A) instead of 10-19 (reg_B).
    """
    engine = Engine(model=MissingParentIntegrationOracle(), target="cpu", mock_execution=False)
    y = np.zeros(engine.layout.n_states)
    ydot = np.zeros(engine.layout.n_states)
    
    # Manually populate the parent state array in memory
    off_c, size_c = engine.layout.state_offsets["c_parent"]
    
    # We set the first half (reg_A) to 0.0, and the second half (reg_B) to 100.0
    y[off_c : off_c + 10] = 0.0 
    y[off_c + 10 : off_c + 20] = 100.0 
    
    # Evaluate the instantaneous residual: F = ydot - rhs = 0.0 - integral
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    off_int, _ = engine.layout.state_offsets["int_val"]
    evaluated_integral = -res[off_int]
    
    # The physical volume of reg_B is exactly 1.0. 
    # The normalized discrete FVM integral of a 100.0 field over a volume of 1.0 is exactly 100.0.
    exact_fvm_integral = 100.0
         
    assert evaluated_integral == pytest.approx(exact_fvm_integral), \
        f"BUG DETECTED: Expected integral {exact_fvm_integral}, but got {evaluated_integral}. " \
        "The compiler dropped the topological start_idx offset and read the wrong memory addresses!"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])