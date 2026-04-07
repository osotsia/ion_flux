"""
Runtime Execution: ALE Kinematics and Moving Boundaries

This suite validates Arbitrary Lagrangian-Eulerian (ALE) grid kinematics.
It proves mass conservation across varying coordinate geometries, validates
non-linear coupled moving boundaries (Stefan problems), ensures stability near 
compression singularities, and verifies namespace isolation for multiple moving grids.
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
# Models
# ==============================================================================

class ALECoordinateScaling(fx.PDE):
    """Zero-flux contraction. Identical math, configurable coordinate system."""
    def __init__(self, coord_sys="cartesian"):
        super().__init__()
        self.r = fx.Domain(bounds=(0, 1.0), resolution=15, coord_sys=coord_sys, name="r")
        self.c = fx.State(domain=self.r, name="c")
        self.L = fx.State(domain=None, name="L")
        
        self.v_shrink = fx.Parameter(default=-0.1, name="v_shrink")
        self.D = fx.Parameter(default=10.0, name="D")

    def math(self):
        flux = -self.D * fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r),
                self.L: fx.dt(self.L) == self.v_shrink
            },
            "boundaries": {
                self.r: {"right": self.L},
                flux: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c: 1.0, self.L: 1.0
            }
        }


class ALEStefanProblem(fx.PDE):
    """
    Classic Stefan problem. Boundary velocity is coupled directly to the 
    spatial flux at the moving boundary, creating a highly stiff, non-linear feedback loop.
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=20, name="x")
    c = fx.State(domain=x, name="c")
    L = fx.State(domain=None, name="L")
    
    D = fx.Parameter(default=0.5, name="D")
    k_growth = fx.Parameter(default=0.2, name="k_growth")

    def math(self):
        flux = -self.D * fx.grad(self.c, axis=self.x)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.x),
                # Growth rate is proportional to flux arriving at the boundary
                self.L: fx.dt(self.L) == self.k_growth * flux.right
            },
            "boundaries": {
                self.x: {"right": self.L},
                self.c: {"left": fx.Dirichlet(5.0), "right": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {
                # Initialize with a steady-state linear gradient so the flux is 
                # maximum at t=0 and strictly decreases as the domain expands.
                self.c: 5.0 - 5.0 * self.x.coords, self.L: 1.0
            }
        }

class MultiDomainALE(fx.PDE):
    """Validates codegen namespace isolation for multiple independent moving meshes."""
    dom_A = fx.Domain(bounds=(0, 1.0), resolution=10, name="dom_A")
    dom_B = fx.Domain(bounds=(0, 1.0), resolution=10, name="dom_B")
    
    c_A = fx.State(domain=dom_A, name="c_A")
    c_B = fx.State(domain=dom_B, name="c_B")
    L_A = fx.State(domain=None, name="L_A")
    L_B = fx.State(domain=None, name="L_B")

    def math(self):
        flux_A = -1.0 * fx.grad(self.c_A)
        flux_B = -1.0 * fx.grad(self.c_B)
        return {
            "equations": {
                self.c_A: fx.dt(self.c_A) == -fx.div(flux_A),
                self.c_B: fx.dt(self.c_B) == -fx.div(flux_B),
                self.L_A: fx.dt(self.L_A) == -0.1,  # Shrinking
                self.L_B: fx.dt(self.L_B) == 0.2    # Expanding
            },
            "boundaries": {
                self.dom_A: {"right": self.L_A},
                self.dom_B: {"right": self.L_B},
                flux_A: {"left": 0.0, "right": 0.0},
                flux_B: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c_A: 1.0, self.c_B: 1.0, self.L_A: 1.0, self.L_B: 1.0
            }
        }

# ==============================================================================
# Tests
# ==============================================================================

@REQUIRES_RUNTIME
def test_ale_coordinate_dilution_scaling():
    """
    Category 1: Geometric Dilution.
    Proves that the ALE compiler perfectly applies the correct geometric multipliers 
    to the advection/dilution terms based on the topology of the moving mesh.
    """
    # 1D Cartesian (Linear scaling)
    model_1d = ALECoordinateScaling(coord_sys="cartesian")
    eng_1d = Engine(model=model_1d, target="cpu", mock_execution=False)
    
    # 3D Spherical (Cubic scaling)
    model_3d = ALECoordinateScaling(coord_sys="spherical")
    eng_3d = Engine(model=model_3d, target="cpu", mock_execution=False)
    
    t_eval = np.linspace(0, 5.0, 50)
    res_1d = eng_1d.solve(t_eval=t_eval, parameters={"v_shrink": -0.1})
    res_3d = eng_3d.solve(t_eval=t_eval, parameters={"v_shrink": -0.1})
    
    # L goes from 1.0 -> 0.5. 
    L_t = 1.0 - 0.1 * t_eval
    
    c_1d = np.mean(res_1d["c"].data, axis=1)
    c_3d = np.mean(res_3d["c"].data, axis=1)
    
    # In Cartesian, Vol ~ L. Mass = C * L. Constant Mass -> C = 1 / L
    exact_1d = 1.0 * (1.0 / L_t)
    # In Spherical, Vol ~ L^3. Constant Mass -> C = 1 / L^3
    exact_3d = 1.0 * (1.0 / L_t)**3
    
    np.testing.assert_allclose(c_1d, exact_1d, rtol=1e-2, err_msg="Cartesian ALE dilution failed.")
    np.testing.assert_allclose(c_3d, exact_3d, rtol=5e-2, err_msg="Spherical ALE dilution failed.")


@REQUIRES_RUNTIME
def test_ale_coupled_stefan_problem():
    """
    Category 2: Coupled Feedback Loops.
    Proves that Engine correctly drops to a dense Jacobian (bandwidth=0) to prevent 
    divergence when ALE velocities are dynamically coupled to boundary fluxes.
    """
    model = ALEStefanProblem()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    assert engine.jacobian_bandwidth == 0, "Engine failed to assign Dense Jacobian for coupled ALE state."
    
    t_eval = np.linspace(0, 2.0, 50)
    res = engine.solve(t_eval=t_eval)
    
    assert res.status == "completed"
    
    L_data = res["L"].data
    # As the boundary expands, the concentration gradient flattens, meaning 
    # the growth rate should continuously decelerate.
    growth_rates = np.diff(L_data)
    
    # Assert L is strictly increasing, but at a decelerating rate
    assert np.all(growth_rates > 0.0), "Boundary failed to expand."
    assert np.all(np.diff(growth_rates) < 0.0), "Growth rate failed to naturally decelerate."


@REQUIRES_RUNTIME
def test_ale_extreme_compression_singularity():
    """
    Category 3: Singularity Handling.
    Proves that compressing a grid to microscopic limits does not crash the solver 
    with divide-by-zero errors in the mesh divergence terms.
    """
    model = ALECoordinateScaling(coord_sys="cartesian")
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    # Shrink to 0.001 length
    t_eval = np.linspace(0, 9.99, 100)
    res = engine.solve(t_eval=t_eval, parameters={"v_shrink": -0.1})
    
    assert res.status == "completed"
    
    L_final = res["L"].data[-1]
    assert L_final == pytest.approx(0.001, rel=1e-4)
    
    # Concentration should scale roughly 1000x without hitting NaNs
    c_final = np.mean(res["c"].data[-1])
    assert np.isfinite(c_final)
    assert c_final > 900.0


@REQUIRES_RUNTIME
def test_multi_domain_ale_namespace_isolation():
    """
    Category 4: Namespace Isolation.
    Proves that multiple independent meshes can shrink/expand simultaneously 
    without their codegen C++ states (`dx`, dilution limits) colliding.
    """
    model = MultiDomainALE()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    # Check C++ source emission for correct variable isolation
    assert "dx_dom_A" in engine.cpp_source
    assert "dx_dom_B" in engine.cpp_source
    
    t_eval = np.linspace(0, 2.0, 20)
    res = engine.solve(t_eval=t_eval)
    
    assert res.status == "completed"
    
    L_A_final = res["L_A"].data[-1]
    L_B_final = res["L_B"].data[-1]
    
    # Dom A shrinks, Dom B expands
    assert L_A_final == pytest.approx(0.8) # 1.0 - 0.1 * 2.0
    assert L_B_final == pytest.approx(1.4) # 1.0 + 0.2 * 2.0
    
    # Because of zero flux and mass conservation, the concentrations should 
    # perfectly match the volumetric dilution/concentration analytical values.
    # We use a 5% tolerance as moving grid temporal integration introduces minor drift.
    np.testing.assert_allclose(res["c_A"].data[-1], 1.25, rtol=5e-2)
    np.testing.assert_allclose(res["c_B"].data[-1], 1.0 / 1.4, rtol=5e-2)