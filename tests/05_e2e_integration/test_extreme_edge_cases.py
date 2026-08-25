"""
Extreme Edge Case Oracles

This suite implements mathematically exact Method of Manufactured Solutions (MMS) 
to probe the limits of the `ion_flux` compiler and native solvers. 

It explicitly targets:
1. Internal Dirichlet Anchors at Piecewise Interfaces (Manifold Severing).
2. Integro-Differential Equations (IDE) (Dense Spatial Jacobian Coupling).
3. Transcendent Circular Algebraic Loops (DAE Initialization bounds).
4. Extreme Multi-Scale Asymmetry (Floating-Point Ill-Conditioning).

These tests will fail (NaNs, rank-deficiency, or divergence) if the 
engine's algorithms rely on simplified assumptions regarding spatial bandwidth, 
matrix scaling, or topological continuity.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx

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
# ORACLE 1: Internal Dirichlet Anchors at Piecewise Interfaces
# ==============================================================================

class InternalDirichletOracle(fx.PDE):
    """
    Proves whether the compiler can successfully sever a continuous spatial mesh 
    into two mathematically isolated sub-domains by pinning a Dirichlet constraint 
    exactly at the shared interface.
    
    If the compiler fails to mask the interface correctly, the solver will attempt 
    to simultaneously satisfy the fixed value and the adjacent Neumann flux, 
    resulting in a structurally singular Jacobian.
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=20, name="cell")
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 2.0), resolution=10, name="reg_B")
    
    c = fx.State(domain=cell, name="c")

    def math(self):
        flux = -fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_A: fx.dt(self.c) == -fx.div(flux),
                    self.reg_B: fx.dt(self.c) == -fx.div(flux)
                })
            },
            "boundaries": {
                # Outer boundaries
                self.c: {"left": fx.Dirichlet(0.0), "right": fx.Dirichlet(2.0)},
                
                # INTERNAL DIRICHLET ANCHORS
                # Forces the shared node(s) between A and B to remain exactly at 1.0.
                self.c.surface(domain=self.reg_A, side="right"): fx.Dirichlet(1.0),
                self.c.surface(domain=self.reg_B, side="left"): fx.Dirichlet(1.0)
            },
            "initial_conditions": {
                self.c: self.cell.coords
            }
        }

@REQUIRES_RUNTIME
def test_oracle_internal_dirichlet_manifold_severing():
    engine = fx.Engine(model=InternalDirichletOracle(), target="cpu", mock_execution=False)
    
    try:
        # A steady-state gradient of c(x) = x perfectly satisfies the diffusion 
        # equation and all boundaries.
        res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    except RuntimeError as e:
        pytest.fail(f"Engine failed to handle internal Dirichlet boundary. "
                    f"Likely caused a singular Jacobian or FVM flux collision: {e}")
        
    c_final = res["c"].data[-1]
    x_coords = np.linspace(0, 2.0, 20)
    
    np.testing.assert_allclose(c_final, x_coords, atol=1e-5,
        err_msg="Internal Dirichlet boundary failed to properly clamp the interface.")


# ==============================================================================
# ORACLE 2: Integro-Differential Spatial Coupling (IDE)
# ==============================================================================

class IntegroDifferentialOracle(fx.PDE):
    """
    Proves that evaluating an integral inside a spatial PDE correctly triggers 
    dense row/column intersections in the Jacobian.
    
    Manufactured Solution: c(x, t) = t + x^2
    - dt(c) = 1.0
    - grad(c) = 2x
    - div(grad(c)) = 2.0
    - integral(c) = t + 1/3
    
    PDE: dt(c) = div(grad(c)) + integral(c) - t - 4/3
    => 1.0 = 2.0 + (t + 1/3) - t - 4/3
    => 1.0 = 1.0
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=30, name="x")
    c = fx.State(domain=x, name="c")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        flux = fx.grad(self.c)
        integral_c = fx.integral(self.c, over=self.x)
        
        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.c: fx.dt(self.c) == fx.div(flux) + integral_c - self.t_var - (4.0 / 3.0)
            },
            "boundaries": {
                flux: {"left": 0.0, "right": 2.0}
            },
            "initial_conditions": {
                self.t_var: 0.0,
                self.c: self.x.coords ** 2
            }
        }

@REQUIRES_RUNTIME
def test_oracle_integro_differential_dense_coupling():
    engine = fx.Engine(model=IntegroDifferentialOracle(), target="cpu", mock_execution=False)
    
    # 1. Assert the Graph Colorer correctly identified the Dense Spatial Coupling
    # By forcing resolution > dense_threshold (30 > 20), the integral's dependency 
    # cross-section will flag the entire state block as dense.
    _, _, _, _, c_dense = engine._cpr_cache
    
    off_c, size_c = engine.layout.state_offsets["c"]
    assert len(c_dense) >= size_c, \
        "HybridGraphColorer failed! It assumed the PDE was sparsely banded and " \
        "ignored the dense $O(N^2)$ coupling introduced by the spatial integral."

    # 2. Assert Dynamic Trajectory Truth
    res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))
    
    x_coords = np.linspace(0, 1.0, 30)
    c_exact = 2.0 + x_coords ** 2
    
    np.testing.assert_allclose(res["c"].data[-1], c_exact, rtol=1e-3, atol=1e-4)


# ==============================================================================
# ORACLE 3: Transcendent Circular Algebraic Loops
# ==============================================================================

class CircularTranscendentDAEOracle(fx.PDE):
    """
    Stresses the algebraic root-finder (`calc_algebraic_roots`) with highly 
    non-linear, interdependent transcendental constraints.
    
    Manufactured Solution: u(t) = t, v(t) = t^2
    Eq 1: u = exp(-v) + t - exp(-t^2)
    Eq 2: v = sin(u) + t^2 - sin(t)
    """
    u = fx.State(domain=None, name="u")
    v = fx.State(domain=None, name="v")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.u: self.u == fx.exp(-self.v) + self.t_var - fx.exp(-(self.t_var ** 2)),
                self.v: self.v == fx.sin(self.u) + (self.t_var ** 2) - fx.sin(self.t_var)
            },
            "boundaries": {},
            "initial_conditions": {
                self.t_var: 0.0,
                self.u: 0.0,
                self.v: 0.0
            }
        }

@REQUIRES_RUNTIME
def test_oracle_transcendent_circular_dae_initialization():
    engine = fx.Engine(model=CircularTranscendentDAEOracle(), target="cpu", mock_execution=False)
    
    try:
        res = engine.solve(t_span=(0, 3.0), t_eval=np.array([0.0, 3.0]))
    except RuntimeError as e:
        pytest.fail(f"Algebraic Root Finder failed on transcendental cycle: {e}")
        
    u_final = res["u"].data[-1]
    v_final = res["v"].data[-1]
    
    np.testing.assert_allclose(u_final, 3.0, rtol=1e-4)
    np.testing.assert_allclose(v_final, 9.0, rtol=1e-4)


# ==============================================================================
# ORACLE 4: Extreme Multi-Scale Asymmetry (Ill-Conditioning)
# ==============================================================================

class IllConditionedAsymmetryOracle(fx.PDE):
    """
    Evaluates linear solver stability when spatial steps and capacities differ 
    by 10+ orders of magnitude within the same coupled Jacobian matrix.
    
    Macro domain: x in [0, 10^3] (1 km). D_mac = 10^6
    Micro domain: r in [0, 10^-9] (1 nm). D_mic = 10^-18
    
    Manufactured Solution:
    c_mac(x, t) = t + x
    c_mic(r, t) = t + r^2 * 10^18
    """
    macro = fx.Domain(bounds=(0, 1e3), resolution=10, name="macro")
    micro = fx.Domain(bounds=(0, 1e-9), resolution=10, coord_sys="spherical", name="micro")
    
    c_mac = fx.State(domain=macro, name="c_mac")
    c_mic = fx.State(domain=macro * micro, name="c_mic")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        D_mac = 1e6
        D_mic = 1e-18
        
        flux_mac = -D_mac * fx.grad(self.c_mac)
        flux_mic = -D_mic * fx.grad(self.c_mic, axis=self.micro)
        
        # c_mic at surface = t + (10^-9)^2 * 10^18 = t + 1.0
        c_surf = self.c_mic.boundary("right", domain=self.micro)
        
        # Coupling mechanism linking km scale to nm scale
        j = c_surf - self.c_mac  # (t + 1.0) - (t + x) = 1.0 - x
        
        eq_mac = -fx.div(flux_mac) + j - (1.0 - self.macro.coords) + 1.0
        eq_mic = -fx.div(flux_mic, axis=self.micro) - 5.0
        
        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.c_mac: fx.dt(self.c_mac) == eq_mac,
                self.c_mic: fx.dt(self.c_mic) == eq_mic
            },
            "boundaries": {
                flux_mac: {"left": -1e6, "right": -1e6},
                flux_mic: {"left": 0.0, "right": -2e-9} # grad(r^2 * 1e18) * -1e-18 = -2r
            },
            "initial_conditions": {
                self.t_var: 0.0,
                self.c_mac: self.macro.coords,
                self.c_mic: (self.micro.coords ** 2) * 1e18
            }
        }

@REQUIRES_RUNTIME
def test_oracle_ill_conditioned_multiscale_asymmetry():
    engine = fx.Engine(model=IllConditionedAsymmetryOracle(), target="cpu", mock_execution=False)
    
    try:
        res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))
    except RuntimeError as e:
        pytest.fail(f"Solver diverged due to extreme matrix ill-conditioning. "
                    f"The LU factorization lacks equilibration or partial pivoting necessary "
                    f"to bridge 10+ orders of magnitude: {e}")
                    
    x_coords = np.linspace(0, 1e3, 10)
    c_mac_exact = 2.0 + x_coords
    
    np.testing.assert_allclose(res["c_mac"].data[-1], c_mac_exact, rtol=1e-3,
        err_msg="Ill-conditioned matrix inversion injected massive floating-point truncation error.")

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])