"""
End-to-End Industry Oracles: The FVM Refactor Crucible

This suite provides exact Method of Manufactured Solutions (MMS) to confirm 
the robustness of the normalized FVM geometry, integral context propagation, 
and topological graph connectivity.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

# ==============================================================================
# Environment Configuration
# ==============================================================================

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

# ==============================================================================
# ORACLE 1: Integral Context Propagation (The Structural Singularity Bug)
# ==============================================================================

class IntegralJacobianSingularityOracle(fx.PDE):
    """
    Exposes the bug where `fx.integral()` failed to pass `self.current_axis` 
    to its child AST nodes. This caused nested `fx.grad()` operators to lose 
    their spatial dimension and fall back to evaluating `0.0`.
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=20, name="cell")
    reg = cell.region(bounds=(0, 2.0), resolution=20, name="reg")
    
    phi = fx.State(domain=cell, name="phi")
    T_var = fx.State(domain=None, name="T_var")
    
    def math(self):
        return {
            "equations": {
                # Lock phi to the exact manufactured profile
                self.phi: fx.dt(self.phi) == 0.0,
                
                # dt(T) = integral(grad(phi)) - T
                # If context is lost, grad(phi) -> 0.0, and T decays to 0.0.
                self.T_var: fx.dt(self.T_var) == fx.integral(fx.grad(self.phi), over=self.reg) - self.T_var
            },
            "boundaries": {
                self.phi: {"left": fx.Dirichlet(0.0), "right": fx.Dirichlet(2.0)}
            },
            "initial_conditions": {
                self.phi: (self.cell.coords ** 2) / 2.0, # Manufactured: phi = x^2 / 2
                self.T_var: 0.0
            }
        }

@REQUIRES_RUNTIME
def test_oracle_integral_jacobian_singularity():
    """
    PROBE: Manufactured Truth: 
    phi = x^2 / 2 -> grad(phi) = x
    integral(grad(phi), over=cell) = integral(x) from 0 to 2 = 2.0.
    Thus, dt(T) = 2.0 - T. T should asymptote exactly to 2.0.
    """
    engine = Engine(model=IntegralJacobianSingularityOracle(), target="cpu", mock_execution=False)
    
    # 1. Assert the Jacobian is strictly full-rank and fully coupled
    y0, ydot0, _, _, _ = engine._extract_metadata()
    J = np.array(engine.evaluate_jacobian(y0, ydot0, c_j=1.0, parameters={}))
    
    # Ensure no rows/cols are sheared to 0.0 identically (except fixed boundaries)
    assert np.linalg.matrix_rank(J) >= engine.layout.n_states - 2, \
        "Structural Singularity Detected! Enzyme failed to emit cross-coupled sensitivities " \
        "because `fx.integral()` wiped the spatial context of `fx.grad()`."

    # 2. Assert Dynamic Trajectory Truth
    res = engine.solve(t_span=(0, 15.0), t_eval=np.array([0.0, 15.0]))
    
    T_final = res["T_var"].data[-1]
    
    # Relaxed rtol to 5% to account for the discrete FVM truncation error on a coarse N=20 mesh. 
    # The true discrete value is exactly 37/19 (~1.947). The bug previously evaluated to exactly 0.0.
    np.testing.assert_allclose(T_final, 2.0, rtol=0.05,
        err_msg=f"Integral Context Loss Detected! Expected T to reach ~2.0, but got {T_final:.3f}. "
                "The nested spatial gradient evaluated to a static 0.0.")


# ==============================================================================
# ORACLE 2: AST Fragmentation & Geometric `KeyError` Aliasing
# ==============================================================================

class SubregionGeometricScalingOracle(fx.PDE):
    """
    Exposes the deepcopy AST fragmentation where sub-regions lose their parent 
    links, causing missing volume geometry arrays and raising a `KeyError`.
    """
    cell = fx.Domain(bounds=(0, 3.0), resolution=30, name="cell")
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 3.0), resolution=20, name="reg_B")
    
    c = fx.State(domain=cell, name="c")
    mass_A = fx.State(domain=None, name="mass_A")
    mass_B = fx.State(domain=None, name="mass_B")
    
    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                # Pure algebraic integrals
                self.mass_A: self.mass_A == fx.integral(self.c, over=self.reg_A),
                self.mass_B: self.mass_B == fx.integral(self.c, over=self.reg_B)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: self.cell.coords, # Manufactured: c(x) = x
                self.mass_A: 0.0,
                self.mass_B: 0.0
            }
        }

@REQUIRES_RUNTIME
def test_oracle_subregion_geometric_scaling():
    """
    PROBE: Manufactured Truth:
    mass_A = integral(x) from 0 to 1 = 0.5
    mass_B = integral(x) from 1 to 3 = (3^2 / 2) - (1^2 / 2) = 4.5 - 0.5 = 4.0
    
    If the TopologyAnalyzer fails to link subregions, `Engine()` instantiation 
    will crash violently with a KeyError during code-generation.
    """
    # The instantiation itself is part of the test (verifies the missing KeyError fix)
    engine = Engine(model=SubregionGeometricScalingOracle(), target="cpu", mock_execution=False)
    
    # Take a single short step to trigger the algebraic evaluation
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    mass_A_sim = res["mass_A"].data[-1]
    mass_B_sim = res["mass_B"].data[-1]
    
    # Relaxed rtol to 1% to account for FVM midpoint-rule quadrature truncation error on a coarse mesh.
    # The primary success metric is that the Engine instantiated without a KeyError crash.
    np.testing.assert_allclose(mass_A_sim, 0.5, rtol=0.01, err_msg="Volume Scaling mapped to incorrect L_phys for reg_A.")
    np.testing.assert_allclose(mass_B_sim, 4.0, rtol=0.01, err_msg="Volume Scaling mapped to incorrect L_phys for reg_B.")


# ==============================================================================
# ORACLE 3: Piecewise Harmonic Mean Conservation
# ==============================================================================

class HarmonicMeanDiscontinuityOracle(fx.PDE):
    """
    Exposes implicit instability and flux non-conservation at sharp material 
    interfaces (Piecewise domains). 
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=40, name="cell")
    reg_L = cell.region(bounds=(0, 1.0), resolution=20, name="reg_L")
    reg_R = cell.region(bounds=(1.0, 2.0), resolution=20, name="reg_R")
    
    c = fx.State(domain=cell, name="c")
    
    def math(self):
        # Extreme jump in material property (e.g., solid vs electrolyte conductivity)
        flux_L = -10.0 * fx.grad(self.c)
        flux_R = -0.1 * fx.grad(self.c)
        
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_L),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_R)
                })
            },
            "boundaries": {
                # Force a steady-state profile across the cell
                self.c: {"left": fx.Dirichlet(100.0), "right": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {
                self.c: 50.0
            }
        }

@REQUIRES_RUNTIME
def test_oracle_harmonic_mean_discontinuity():
    """
    PROBE: In steady-state, flux must be perfectly continuous at x=1.0.
    -10.0 * grad_L = -0.1 * grad_R
    Analytical Truth at Interface (c_int):
    10.0 * (100 - c_int) / 1.0 = 0.1 * (c_int - 0) / 1.0
    1000 - 10 c_int = 0.1 c_int  ->  10.1 c_int = 1000  -> c_int = 99.0099
    """
    engine = Engine(model=HarmonicMeanDiscontinuityOracle(), target="cpu", mock_execution=False)
    
    # Integrate to extreme steady state
    res = engine.solve(t_span=(0, 500.0), t_eval=np.array([0.0, 500.0]))
    c_final = res["c"].data[-1]
    
    # Interface nodes (indices 19 and 20 for a 40-node mesh sliced down the middle)
    c_int_L = c_final[19]
    c_int_R = c_final[20]
    
    # Because of the 100x discrepancy in diffusivity, the arithmetic mean is physically invalid.
    # We reconstruct the exact interface value by weighting nodes by their diffusivities
    # to explicitly invert the flux conservation: D_L * (c_L - c_int) = D_R * (c_int - c_R)
    D_L, D_R = 10.0, 0.1
    c_int_exact = (D_L * c_int_L + D_R * c_int_R) / (D_L + D_R)
    
    np.testing.assert_allclose(
        c_int_exact, 99.0099, rtol=1e-2,
        err_msg=f"Harmonic Mean Interface Failure! Flux is leaking across the Piecewise boundary. "
                f"Expected interface concentration ~99.01, got {c_int_exact:.2f}."
    )

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])