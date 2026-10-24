import pytest
import numpy as np
import shutil
import platform
import os
import sys
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, CV, Rest

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


# ==============================================================================
# Test Models
# ==============================================================================

class BatteryProtocolPDE(fx.PDE):
    """Validates CCCV Hot-Swapping, DAE Constraints, and Native vs Sundials accuracy."""
    soc = fx.State(domain=None, name="soc")
    V = fx.State(domain=None, name="V")
    i_app = fx.State(domain=None, name="i_app")
    
    terminal = fx.Terminal(current=i_app, voltage=V)
    R = fx.Parameter(default=0.05, name="R")
    
    def math(self):
        return {
            "equations": {
                self.soc: fx.dt(self.soc) == -self.i_app / 3600.0,
                self.V: self.V == 4.0 + self.soc - self.i_app * self.R
            },
            "boundaries": {},
            "initial_conditions": {
                self.soc: 1.0, self.V: 4.5, self.i_app: 10.0
            }
        }


class SpatialVerificationPDE(fx.PDE):
    """Verifies that spatial fields reject scalar get() operations."""
    x = fx.Domain(bounds=(0, 1.0), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    v = fx.State(domain=None, name="v")

    def math(self):
        flux = -fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux),
                self.v: self.v == 1.0
            },
            "boundaries": {
                flux: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c: 500.0,
                self.v: 1.0
            }
        }


# ==============================================================================
# Tests
# ==============================================================================

@REQUIRES_RUNTIME
def test_native_vs_sundials_cccv_hot_swapping():
    """
    Validates both Native and Sundials IDAS solvers can perfectly hot-swap 
    Algebraic constraints mid-solve (CC to CV) utilizing Python root-finding logic.
    """
    model = BatteryProtocolPDE()
    engine_native = fx.Engine(model=model, target="cpu", solver_backend="native")
    engine_sundials = fx.Engine(model=model, target="cpu", solver_backend="sundials")
    
    protocol = Sequence([
        CC(rate=10.0, until=model.V <= 3.2),
        CV(voltage=3.2, time=5.0)
    ])
    
    res_native = engine_native.solve(protocol=protocol)
    res_sundials = engine_sundials.solve(protocol=protocol)
    
    assert res_native.status == "completed"
    assert res_sundials.status == "completed"
    
    V_n = res_native["V"].data
    assert np.max(V_n) <= 5.0 + 1e-3
    assert V_n[-1] == pytest.approx(3.2, rel=1e-3)
    
    np.testing.assert_allclose(res_native["V"].data[-1], res_sundials["V"].data[-1], rtol=1e-3)
    np.testing.assert_allclose(res_native["i_app"].data[-1], res_sundials["i_app"].data[-1], rtol=2e-2)


@REQUIRES_RUNTIME
def test_stateful_session_hil_control():
    """Validates the SolverHandle maintains memory seamlessly for continuous micro-stepping (BMS HIL)."""
    engine = fx.Engine(model=BatteryProtocolPDE(), target="cpu", mock_execution=False)
    session = engine.start_session()
    
    session.terminal.rest()
    assert session.time == 0.0
    assert session.get_scalar("V") == pytest.approx(5.0, abs=1e-5)
    
    session.terminal.set_current(1.0)
    session.step(dt=1800.0) 
    
    assert session.time == 1800.0
    assert session.get_scalar("soc") == pytest.approx(0.5, abs=1e-4)
    assert session.get_scalar("V") == pytest.approx(4.45, abs=1e-4)


@REQUIRES_RUNTIME
def test_session_spatial_get_rejection():
    """Validates that get() and get_scalar() reject non-scalar states, enforcing get_array()."""
    engine = fx.Engine(model=SpatialVerificationPDE(), target="cpu", mock_execution=False)
    session = engine.start_session()
    
    # 0D scalar works cleanly
    assert session.get_scalar("v") == pytest.approx(1.0)
    assert session.get("v") == pytest.approx(1.0)
    
    # Spatial field must fail immediately
    with pytest.raises(ValueError, match="is a spatial field with size 5"):
        session.get_scalar("c")
        
    with pytest.raises(ValueError, match="is a spatial field with size 5"):
        session.get("c")

    # Explicit get_array succeeds
    c_arr = session.get_array("c")
    assert isinstance(c_arr, np.ndarray)
    assert c_arr.shape == (5,)
    assert np.all(c_arr == 500.0)


@REQUIRES_RUNTIME
def test_dae_gating_on_continuous_updates():
    """Validates that continuous current updates in CC mode do not invoke algebraic reinitialization."""
    engine = fx.Engine(model=BatteryProtocolPDE(), target="cpu", mock_execution=False)
    session = engine.start_session()
    
    session.terminal.set_current(1.0)
    
    # Monkeypatch the Python facade method to track invocations
    calc_calls = 0
    orig_reinit = session.reinitialize_algebraic
    def counting_reinit():
        nonlocal calc_calls
        calc_calls += 1
        return orig_reinit()
    session.reinitialize_algebraic = counting_reinit
    
    # Perturb current continuously 10 times within the same operating mode
    for i in range(10):
        session.terminal.set_current(1.0 + i * 0.1)
        session.step(dt=0.01)
        
    # Mode was unchanged (1.0 -> 1.0), so reinitialize_algebraic must not have been called
    assert calc_calls == 0, f"Redundant algebraic solve triggered {calc_calls} times during continuous updates."
    
    # Transitioning to CV mode must trigger algebraic solve exactly once
    session.terminal.set_voltage(4.2)
    assert calc_calls == 1
    

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])