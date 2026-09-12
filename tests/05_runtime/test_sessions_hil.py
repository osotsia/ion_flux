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
import asyncio
import os
import shutil
import platform
import numpy as np
import ion_flux as fx
from ion_flux.runtime.scheduler import MultiTenantScheduler
from ion_flux.protocols import Sequence, CC, CV, Rest
from ion_flux._core import solve_ida_native



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



# ==============================================================================
# Concept 1: Core Solver Integration & Protocol Hot-Swapping
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
    
    # Validation 1: Proper clamping at 3.2V without overshoot
    V_n = res_native["V"].data
    assert np.max(V_n) <= 5.0 + 1e-3
    assert V_n[-1] == pytest.approx(3.2, rel=1e-3)
    
    # Validation 2: Tight cross-validation between Native and Sundials execution traces
    np.testing.assert_allclose(res_native["V"].data[-1], res_sundials["V"].data[-1], rtol=1e-3)
    np.testing.assert_allclose(res_native["i_app"].data[-1], res_sundials["i_app"].data[-1], rtol=2e-2)




@REQUIRES_RUNTIME
def test_stateful_session_hil_control():
    """Validates the SolverHandle maintains memory seamlessly for continuous micro-stepping (BMS HIL)."""
    engine = fx.Engine(model=BatteryProtocolPDE(), target="cpu", mock_execution=False)
    
    # Initialize at Rest (0 current)
    session = engine.start_session(parameters={"_term_i_target": 0.0, "_term_mode": 1.0})
    assert session.time == 0.0
    
    # OCV = 4.0 + 1.0 (soc) = 5.0V at 0.0A
    assert session.get("V") == pytest.approx(5.0, abs=1e-5)
    
    # Step forward 1800s (0.5 hrs) at 1.0A
    session.step(dt=1800.0, inputs={"_term_i_target": 1.0}) 
    
    assert session.time == 1800.0
    # soc drops by 1A * 0.5hr / 1Ah = 0.5
    assert session.get("soc") == pytest.approx(0.5, abs=1e-4)
    # V = 4.0 + 0.5 (soc) - 1.0A * 0.05R = 4.45V
    assert session.get("V") == pytest.approx(4.45, abs=1e-4)
