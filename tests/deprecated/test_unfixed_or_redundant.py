import pytest
import numpy as np
import shutil
import platform
import os
import sys
import ion_flux as fx

# Ensure models directory is in path for E2E tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)

import pytest
import shutil
import platform
import ion_flux as fx
from ion_flux._core import solve_ida_native
import numpy as np


"""
Compiler Bug Oracle: Permutation-Invariant Zero Pivot Panic

Isolates the fundamental algorithmic flaw in the Native Sparse LU backend.
By constructing a Jacobian that evaluates to a dense block of 1.0s, we mathematically
guarantee that Gaussian elimination will produce a zero pivot on the second step.
Because all permutations of a uniform dense matrix are identical, this completely 
defeats `faer`'s AMD ordering, proving that the lack of partial pivoting in the 
`simplicial` module causes hard Rust panics on valid equations.
"""



class DefeatAMDZeroPivotOracle(fx.PDE):
    y1 = fx.State(domain=None, name="y1")
    y2 = fx.State(domain=None, name="y2")
    y3 = fx.State(domain=None, name="y3")
    y4 = fx.State(domain=None, name="y4")

    def math(self):
        # The Jacobian dF/dy for this system is a 4x4 matrix of -1.0s.
        total = self.y1 + self.y2 + self.y3 + self.y4
        return {
            "equations": {
                self.y1: 0.0 == total - 1.0,
                self.y2: 0.0 == total - 2.0,
                self.y3: 0.0 == total - 3.0,
                self.y4: 0.0 == total - 4.0
            },
            "boundaries": {},
            "initial_conditions": {
                self.y1: 0.0, self.y2: 0.0, self.y3: 0.0, self.y4: 0.0
            }
        }



@REQUIRES_RUNTIME
@pytest.mark.xfail(reason="May have to get rid of faer eventually")
def test_simplicial_lu_zero_pivot_panic(capfd):
    """
    PROBE: Executes a model designed to defeat AMD fill-reducing permutations.
    This guarantees `faer` encounters a zero pivot. Since `faer`'s `simplicial` 
    module lacks partial pivoting, it will hard panic.
    """
    engine = fx.Engine(model=DefeatAMDZeroPivotOracle(), target="cpu", mock_execution=False)
    
    try:
        engine.solve(t_span=(0, 1.0))
    except RuntimeError:
        pass
        
    captured = capfd.readouterr()
    
    assert "panicked at" not in captured.err, \
        "BUG DETECTED: Rust panic leaked to stderr! `faer`'s simplicial LU solver " \
        "cannot handle zero pivots generated during Gaussian elimination, and the " \
        "matrix successfully defeated AMD reordering."



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
@pytest.mark.skip(reason="Interface conservation is now validated by the native FVM oracles in tests/03_backend/test_fvm_discretization.py")
def test_steady_state_discontinuous_flux_averaging():
    engine = fx.Engine(model=DiscontinuousSteadyStateOracle(), target="cpu", mock_execution=False)
    
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
