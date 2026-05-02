"""
Steady-State Discontinuity Oracle

This suite proves that arithmetic averaging of fluxes at a discontinuous material 
interface fundamentally corrupts mass conservation and steady-state thermodynamics.
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

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)

class DiscontinuousSteadyStateOracle(fx.PDE):
    """
    Simulates a 1D diffusion problem across a stark material discontinuity.
    Domain length = 2.0, split into two regions. Resolution = 4 nodes total.
    Node 0: reg_L (Left Boundary, Dirichlet=110.0)
    Node 1: reg_L (Bulk)
    Node 2: reg_R (Bulk, adjacent to interface)
    Node 3: reg_R (Right Boundary, Dirichlet=0.0)
    
    Left Cell: D_L = 100.0
    Right Cell: D_R = 0.01
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=4, name="cell")
    reg_L = cell.region(bounds=(0, 1.0), resolution=2, name="reg_L")
    reg_R = cell.region(bounds=(1.0, 2.0), resolution=2, name="reg_R")
    
    c = fx.State(domain=cell, name="c")
    
    def math(self):
        D_L = 100.0
        D_R = 0.01
        
        flux_L = -D_L * fx.grad(self.c)
        flux_R = -D_R * fx.grad(self.c)
        
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_L),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_R)
                })
            },
            "boundaries": {
                self.c: {"left": fx.Dirichlet(110.0), "right": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {
                self.c: 0.0
            }
        }

@REQUIRES_RUNTIME
def test_steady_state_discontinuous_flux_averaging():
    engine = Engine(model=DiscontinuousSteadyStateOracle(), target="cpu", mock_execution=False)
    
    # Integrate to a massive time to guarantee thermodynamic steady-state
    # D_R = 0.01, so diffusion time scale is L^2/D = 1^2 / 0.01 = 100s. 100,000s is fully steady.
    res = engine.solve(t_span=(0, 100000.0), t_eval=np.array([0.0, 100000.0]))
    
    c_final = res["c"].data[-1]
    
    # We evaluate Node 2 (index 2), the first node in the highly resistive right region.
    simulated_c_2 = c_final[2]
    
    # =========================================================================
    # EXACT ANALYTICAL FVM SOLUTION (Harmonic Mean Interpolation):
    # =========================================================================
    # At steady state, fluxes between cell centers are identical.
    # J = D_eff * (C_i - C_i+1) / dx.  Let J' = J * dx.
    # Resistance = 1 / D_eff.
    # 
    # R01 (Node 0 to 1) = 1 / 100.0 = 0.01
    # R12 (Interface)   = 1 / D_harmonic
    # R23 (Node 2 to 3) = 1 / 0.01  = 100.0
    #
    # D_harmonic = 2 * (100 * 0.01) / (100 + 0.01) = 0.019998
    # R12 = 1 / 0.019998 = 50.005
    # 
    # Total Resistance R_tot = 0.01 + 50.005 + 100.0 = 150.015
    # J' = (110.0 - 0.0) / 150.015 = 0.73326
    # 
    # c_2 = c_3 + J' * R23 = 0.0 + 0.73326 * 100.0 = 73.326
    exact_c_2 = 73.326
    
    # =========================================================================
    # ERRONEOUS ARITHMETIC MEAN SOLUTION:
    # =========================================================================
    # D_arithmetic = (100.0 + 0.01) / 2 = 50.005
    # R12 = 1 / 50.005 = 0.019998
    # 
    # Total Resistance R_tot = 0.01 + 0.019998 + 100.0 = 100.029998
    # J' = (110.0 - 0.0) / 100.029998 = 1.09967
    # 
    # c_2 = c_3 + J' * R23 = 0.0 + 1.09967 * 100.0 = 109.967
    
    error_msg = (
        f"\nBUG DETECTED: Discontinuous Flux Averaging Failure.\n"
        f"Expected steady-state concentration in the right cell: {exact_c_2:.3f}\n"
        f"Simulated concentration: {simulated_c_2:.3f}\n\n"
        f"Explanation:\n"
        f"The compiler applies an arithmetic average to disjointed fluxes at the interface: "
        f"0.5 * (flux_L + flux_R). Because D_L=100.0 and D_R=0.01, this forces an effective "
        f"interface conductivity of 50.005, which completely destroys the interface resistance. "
        f"Mass conservation across material boundaries STRICTLY requires the Harmonic Mean "
        f"(D_eff = 0.01999), otherwise physical gradients deviate by ~50%."
    )
    
    np.testing.assert_allclose(simulated_c_2, exact_c_2, rtol=1e-3, err_msg=error_msg)

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])