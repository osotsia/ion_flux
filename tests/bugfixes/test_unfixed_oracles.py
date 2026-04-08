"""
Unfixed Bug Oracles

This suite provides distinct, minimal-reproduction test cases for two critical engine bugs:
1. BDF History Corruption during `Session.restore()` (Affects event-locator bisection).
2. Missing FVM `V_nodes` array mapping in unstructured `Domain.from_mesh` compilations.

These tests will FAIL against the current compiler/runtime, and will PASS once the
fixes to `session.rs`, `memory.py`, and `spatial.py` are applied.
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
# BUG 1: Session Restore VSVO History Corruption
# ==============================================================================

class ExponentialGrowth(fx.PDE):
    """
    dy/dt = y
    Analytical solution: y(t) = exp(t)
    """
    y = fx.State(domain=None, name="y")
    def math(self):
        return {
            "equations": {self.y: fx.dt(self.y) == self.y},
            "boundaries": {},
            "initial_conditions": {self.y: 1.0}
        }

@REQUIRES_RUNTIME
def test_session_restore_bdf_history_corruption():
    """
    PROBE: Proves that `Session.restore()` fails to clear the BDF polynomial history.
    If unfixed, the massive `ydot` from the `dt=5.0` step will bleed into the 
    prediction for the subsequent `dt=0.1` step, causing a massive overshoot.
    """
    engine = Engine(model=ExponentialGrowth(), target="cpu", mock_execution=False)
    session = engine.start_session()
    
    # 1. Step to t=1.0. Exact analytical y(1.0) = exp(1.0)
    session.step(1.0)
    assert session.get("y") == pytest.approx(np.exp(1.0), rel=1e-4)
    
    # 2. Save state safely at t=1.0
    session.checkpoint()
    
    # 3. Corrupt the VSVO history polynomials with a massive forward jump
    session.step(5.0) 
    
    # 4. Restore back to t=1.0
    session.restore()
    
    # 5. Take a small step to t=1.1
    session.step(0.1)
    
    # Check against the exact analytical solution y(1.1) = exp(1.1)
    y_sim = session.get("y")
    y_exact = np.exp(1.1)
    
    # If the bug is present, y_sim will be severely inflated due to the poisoned history.
    np.testing.assert_allclose(
        y_sim, y_exact, rtol=1e-3, 
        err_msg=f"History Corruption Bug: Expected {y_exact:.4f}, got {y_sim:.4f}. "
                "The BDF history polynomials (ydot, phi) were not flushed during restore!"
    )


# ==============================================================================
# BUG 2: Unstructured FVM Volume Mapping
# ==============================================================================

# A simple unit tetrahedron (Vertices at origin, and 1.0 along x, y, z axes)
# Geometric Volume = (1/3) * Base_Area * Height = (1/3) * (1/2 * 1 * 1) * 1 = 1/6
tetrahedron_mesh = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "elements": [[0, 1, 2, 3]]
}

class UnstructuredVolumeOracle(fx.PDE):
    """
    Integrates a constant field of 1.0 over a spatial mesh.
    Mathematically: ∫ 1.0 dV = Total Mesh Volume
    """
    mesh = fx.Domain.from_mesh(tetrahedron_mesh, name="mesh")
    c = fx.State(domain=mesh, name="c")
    
    # 0D tracker for total volume evaluation
    mesh_volume = fx.State(domain=None, name="mesh_volume")
    
    def math(self):
        return {
            "equations": {
                # Lock spatial field to exactly 1.0
                self.c: fx.dt(self.c) == 0.0,
                # Integral of a constant 1.0 field is exactly its geometric volume
                self.mesh_volume: self.mesh_volume == fx.integral(self.c, over=self.mesh)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 1.0, 
                self.mesh_volume: 0.0
            }
        }

@REQUIRES_RUNTIME
def test_unstructured_integral_volume_mapping():
    """
    PROBE: Proves the compiler discards `V_nodes` for unstructured domains.
    If unfixed, `fx.integral` over an unstructured mesh defaults to a 1D Cartesian 
    loop with `dx=1.0`, resulting in a volume of ~3.0 to 4.0 instead of 1/6.
    """
    engine = Engine(model=UnstructuredVolumeOracle(), target="cpu", mock_execution=False)
    
    # Take one arbitrary step so the solver evaluates the algebraic integral equation
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    exact_volume = 1.0 / 6.0
    simulated_volume = res["mesh_volume"].data[-1]
    
    assert simulated_volume == pytest.approx(exact_volume, rel=1e-5), \
        f"Unstructured Volume Bug: Expected exact volume {exact_volume:.5f}, but got {simulated_volume:.5f}. " \
        "The compiler failed to inject the exact `V_nodes` array into the unstructured integration loop."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])