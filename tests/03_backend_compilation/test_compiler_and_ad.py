"""
Backend Compilation: Clang Invocation & Enzyme Automatic Differentiation

Comprehensive validation of the LLVM backend. 
Tests the subprocess invocation of Clang++, generation of `.so` binaries, 
FFI loading, Curtis-Powell-Reid (CPR) banded graph coloring, and Enzyme's 
ability to generate exact analytical Jacobians through smooth math, 
non-differentiable kinks (piecewise), and discontinuous boolean triggers.
"""

import pytest
import shutil
import platform
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine
from ion_flux.compiler.invocation import NativeCompiler

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

def _has_enzyme() -> bool:
    return bool(NativeCompiler().enzyme_plugin)

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")
REQUIRES_ENZYME = pytest.mark.skipif(not _has_compiler() or not _has_enzyme(), reason="Requires Enzyme LLVM plugin.")

# ==============================================================================
# Helper Oracle
# ==============================================================================

def approx_jacobian(engine: Engine, y: list, ydot: list, p: dict, c_j: float, eps: float = 1e-8) -> np.ndarray:
    """
    Computes the numerical Jacobian using central finite differences.
    For an implicit solver, the residual mapping is F(y, ydot).
    J = dF/dy + c_j * dF/dydot
    """
    N = len(y)
    J = np.zeros((N, N))
    
    for col in range(N):
        y_fwd, ydot_fwd = list(y), list(ydot)
        y_bwd, ydot_bwd = list(y), list(ydot)
        
        y_fwd[col] += eps
        ydot_fwd[col] += eps * c_j
        y_bwd[col] -= eps
        ydot_bwd[col] -= eps * c_j
        
        res_fwd = engine.evaluate_residual(y_fwd, ydot_fwd, parameters=p)
        res_bwd = engine.evaluate_residual(y_bwd, ydot_bwd, parameters=p)
        
        for row in range(N):
            J[row, col] = (res_fwd[row] - res_bwd[row]) / (2 * eps)
            
    return J

# ==============================================================================
# Heavyweight Models
# ==============================================================================

class MathGauntletPDE(fx.PDE):
    """Combines smooth math, piecewise functions, and step logic into one AD test."""
    y_smooth = fx.State(domain=None, name="y_smooth")
    y_piece = fx.State(domain=None, name="y_piece")
    y_step = fx.State(domain=None, name="y_step")
    
    p_scale = fx.Parameter(default=2.0)
    p_limit = fx.Parameter(default=1.0)
    p_thresh = fx.Parameter(default=2.5)
    
    def math(self):
        trigger = self.y_step > self.p_thresh
        return {
            "equations": {
                # Smooth: sin, exp, cos, log
                self.y_smooth: fx.dt(self.y_smooth) == fx.sin(self.y_smooth) * fx.exp(self.y_piece) - fx.cos(self.y_smooth * self.p_scale),
                
                # Piecewise: abs, max, min (Generates subgradients at kinks)
                self.y_piece: self.y_piece == fx.abs(self.y_piece) + fx.max(self.y_smooth, self.p_limit) + fx.min(self.y_piece, 0.0),
                
                # Step Logic: Relational operator acting as a Heaviside trigger
                self.y_step: fx.dt(self.y_step) == trigger * self.y_step
            },
            "boundaries": {},
            "initial_conditions": {
                self.y_smooth: 2.0, self.y_piece: -1.0, self.y_step: 3.0
            }
        }

class BandedCouplingPDE(fx.PDE):
    """1D model to validate Curtis-Powell-Reid (CPR) graph coloring."""
    x = fx.Domain(bounds=(0, 1), resolution=5)
    c = fx.State(domain=x, name="c")
    D = fx.Parameter(default=1.5)
    
    def math(self):
        flux = -self.D * fx.grad(self.c)
        return {
            "equations": { self.c: fx.dt(self.c) == -fx.div(flux) },
            "boundaries": { self.c: {"left": 1.0, "right": 0.0} },
            "initial_conditions": { self.c: 0.5 }
        }

# ==============================================================================
# Tests
# ==============================================================================

@REQUIRES_COMPILER
def test_clang_so_emission_and_ffi_loading():
    """
    Validates Clang properly compiles the emitted C++ into a portable shared object
    and safely loads it into Python memory via ctypes FFI.
    """
    engine = Engine(model=MathGauntletPDE(), target="cpu", mock_execution=False)
    
    # Prove the Engine didn't silently fall back to mock execution
    assert getattr(engine, "mock_execution", False) is False
    assert engine.runtime is not None
    assert engine.runtime.lib_path.endswith(".so") or engine.runtime.lib_path.endswith(".dylib")
    
    # Retrieve dynamic offsets
    off_s, _ = engine.layout.state_offsets["y_smooth"]
    off_p, _ = engine.layout.state_offsets["y_piece"]
    off_step, _ = engine.layout.state_offsets["y_step"]
    
    N = engine.layout.n_states
    y = np.zeros(N)
    y[off_s], y[off_p], y[off_step] = 2.0, -1.0, 3.0
    ydot = np.zeros(N)
    ydot[off_s], ydot[off_p], ydot[off_step] = 0.1, 0.0, 1.0
    
    params = {"p_scale": 2.0, "p_limit": 1.0, "p_thresh": 2.5}
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters=params)
    
    # Oracle Validation
    # y_step eq: ydot_step - (trigger * y_step) => 1.0 - (1.0 * 3.0) = -2.0
    assert res[off_step] == pytest.approx(-2.0, rel=1e-5)


@REQUIRES_ENZYME
def test_enzyme_analytical_dense_jacobian_smooth_math():
    """
    Proves Enzyme LLVM Reverse/Forward AD correctly differentiates smooth math 
    (sin, cos, exp) perfectly matching a rigorous Finite Difference oracle.
    """
    engine = Engine(model=MathGauntletPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    off_s, _ = engine.layout.state_offsets["y_smooth"]
    off_p, _ = engine.layout.state_offsets["y_piece"]
    off_step, _ = engine.layout.state_offsets["y_step"]
    
    y = np.zeros(N)
    y[off_s], y[off_p], y[off_step] = 2.0, -0.5, 3.0 # Evaluate away from kinks for FD safety
    ydot = np.zeros(N)
    ydot[off_s], ydot[off_p], ydot[off_step] = 0.1, 0.0, 1.0
    
    p = {"p_scale": 1.5, "p_limit": 1.0, "p_thresh": 2.5}
    c_j = 10.0
    
    jac_analytical = np.array(engine.evaluate_jacobian(y.tolist(), ydot.tolist(), c_j, parameters=p))
    jac_numerical = approx_jacobian(engine, y.tolist(), ydot.tolist(), p, c_j)
    
    np.testing.assert_allclose(jac_analytical, jac_numerical, rtol=1e-5, atol=1e-6)


@REQUIRES_ENZYME
def test_enzyme_subgradients_and_heaviside_triggers():
    """
    Validates Enzyme behavior on non-differentiable boundaries.
    Numerical FD fails at exactly x=0 for abs(x). Enzyme AD returns valid subgradients 
    (preventing NaN/Segfaults) and properly routes Boolean Heavyside step gradients.
    """
    engine = Engine(model=MathGauntletPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    off_s, _ = engine.layout.state_offsets["y_smooth"]
    off_p, _ = engine.layout.state_offsets["y_piece"]
    off_step, _ = engine.layout.state_offsets["y_step"]
    
    y = np.zeros(N)
    # y_piece = 0.0 creates a non-differentiable kink for abs() and min()
    # y_smooth = 1.0 creates a kink for max(y_smooth, 1.0)
    y[off_s], y[off_p], y[off_step] = 1.0, 0.0, 3.0 
    ydot = np.zeros(N)
    
    p = {"p_scale": 1.5, "p_limit": 1.0, "p_thresh": 2.5}
    c_j = 10.0
    
    jac_ana_kink = np.array(engine.evaluate_jacobian(y.tolist(), ydot.tolist(), c_j, parameters=p))
    
    # 1. Kink Subgradients
    assert not np.isnan(jac_ana_kink).any(), "Enzyme produced NaN at a mathematical kink."
    assert np.isfinite(jac_ana_kink).all()
    
    # 2. Boolean Heaviside Gradient Pass-Through
    # Active (y_step = 3.0 > 2.5) -> Eq: ydot_step = 1.0 * y_step
    # J(step, step) = c_j * d(ydot)/dydot + d(-y_step)/dy_step = c_j - 1.0
    assert jac_ana_kink[off_step, off_step] == pytest.approx(c_j - 1.0, rel=1e-5)
    
    # 3. Boolean Heaviside Gradient Blocking
    # Inactive (y_step = 1.0 < 2.5) -> Eq: ydot_step = 0.0 * y_step
    # J(step, step) = c_j * 1.0 + 0.0 = c_j
    y[off_step] = 1.0
    jac_inactive = np.array(engine.evaluate_jacobian(y.tolist(), ydot.tolist(), c_j, parameters=p))
    assert jac_inactive[off_step, off_step] == pytest.approx(c_j, rel=1e-5)


@REQUIRES_ENZYME
def test_cpr_graph_coloring_banded_jacobian():
    """
    Validates that the Curtis-Powell-Reid (CPR) algorithm correctly computes 
    compressed Banded Jacobians using minimal forward Enzyme AD sweeps.
    """
    # Force Tridiagonal bandwidth (bw=1)
    engine_banded = Engine(model=BandedCouplingPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=1)
    # Force Dense bandwidth (bw=0)
    engine_dense = Engine(model=BandedCouplingPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine_banded.layout.n_states
    y = np.random.uniform(0.1, 1.0, size=N).tolist()
    ydot = np.zeros(N).tolist()
    
    jac_banded = np.array(engine_banded.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    jac_dense = np.array(engine_dense.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    
    # Ensure CPR coloring correctly captured all coupling elements without truncation
    np.testing.assert_allclose(
        jac_banded, 
        jac_dense, 
        atol=1e-10, 
        err_msg="CPR Banded Graph Coloring mismatched the exact Dense Jacobian baseline."
    )