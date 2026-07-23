"""
Runtime Execution: Robustness and Torture Tests

This suite subjects the implicit native solver to algorithmic extremes:
1. Massive t=0 algebraic discontinuities.
2. High-frequency protocol chatter (stressing history Nordsieck rebuilds).
3. Extreme non-linear stiffness and floating-point underflow.
4. Aggressive trigger hysteresis (stressing bisection root-finding).
5. Unsolvable DAEs (stressing graceful degradation and segfault guards).
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine
from ion_flux.protocols import Sequence, CC, Rest

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
# Torture Models
# ==============================================================================

class AlgebraicSnapCrucible(fx.PDE):
    x = fx.State(domain=None, name="x")
    V = fx.State(domain=None, name="V")
    i_app = fx.State(domain=None, name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=V)
    
    def math(self):
        return {
            "equations": {
                self.x: fx.dt(self.x) == 0.0,
                self.V: self.V == self.x - self.i_app * 100.0 
            },
            "boundaries": {},
            "initial_conditions": {self.x: 4.2, self.V: 4.2, self.i_app: 0.0}
        }

class PathologicalStiffness(fx.PDE):
    r = fx.Domain(bounds=(0, 1.0), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    D = fx.Parameter(default=1.0, name="D")
    
    def math(self):
        flux = -self.D * (self.c ** 2) * fx.grad(self.c, axis=self.r)
        return {
            "equations": {self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r)},
            "boundaries": {flux: {"left": 0.0, "right": -1.0}},
            "initial_conditions": {self.c: 100.0}
        }

class BisectionHysteresis(fx.PDE):
    y = fx.State(domain=None, name="y")
    i_app = fx.State(domain=None, name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=y)
    
    def math(self):
        return {
            "equations": {self.y: fx.dt(self.y) == self.i_app},
            "boundaries": {},
            "initial_conditions": {self.y: 0.0, self.i_app: 0.0}
        }

class UnsolvableParadox(fx.PDE):
    x = fx.State(domain=None, name="x")
    def math(self):
        return {
            "equations": {self.x: self.x == self.x + 1.0},
            "boundaries": {},
            "initial_conditions": {self.x: 0.0}
        }

class SimpleTimingIntegrator(fx.PDE):
    y = fx.State(domain=None, name="y")
    i_app = fx.State(domain=None, name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=y)
    def math(self):
        return {
            "equations": {self.y: fx.dt(self.y) == self.i_app},
            "boundaries": {},
            "initial_conditions": {self.y: 0.0, self.i_app: 0.0}
        }

# ==============================================================================
# Torture Tests
# ==============================================================================

@REQUIRES_RUNTIME
def test_torture_t0_algebraic_snap():
    engine = Engine(model=AlgebraicSnapCrucible(), target="cpu", mock_execution=False)
    protocol = Sequence([CC(rate=100.0, time=1.0)])
    res = engine.solve(protocol=protocol)
    
    assert res.status == "completed"
    assert res["V"].data[0] == pytest.approx(-9995.8, rel=1e-5)

@REQUIRES_RUNTIME
def test_torture_high_frequency_chatter():
    engine = Engine(model=AlgebraicSnapCrucible(), target="cpu", mock_execution=False)
    steps = []
    for _ in range(25):
        steps.append(CC(rate=10.0, time=1.0))
        steps.append(Rest(time=1.0))
        
    try:
        res = engine.solve(protocol=Sequence(steps), show_progress=False)
    except RuntimeError as e:
        pytest.fail(f"High-frequency protocol chatter crashed the solver: {e}")
        
    assert res.status == "completed"
    assert res["Time [s]"].data[-1] == pytest.approx(50.0)

@REQUIRES_RUNTIME
def test_torture_pathological_stiffness():
    """
    TORTURE 3: Extreme Non-Linear Stiffness.
    Drives a highly non-linear diffusion field with D = 1e-24. Validates that the 
    C++ backend either miraculously solves it or degrades gracefully into a Python Exception.
    It MUST NOT segfault the host process.
    """
    engine = Engine(model=PathologicalStiffness(), target="cpu", mock_execution=False)
    
    try:
        res = engine.solve(t_span=(0, 1.0), parameters={"D": 1e-24})
        assert res.status == "completed"
    except RuntimeError as e:
        err_str = str(e)
        assert "NATIVE SOLVER CRASH" in err_str
        assert "Tolerance Starvation" in err_str or "Nonlinear Divergence" in err_str

@REQUIRES_RUNTIME
def test_torture_bisection_asymptote_tracking():
    """
    TORTURE 4: Bisection Root-Finding Hysteresis.
    Drives an ODE to a sharp trigger, then immediately reverses the direction 
    of the ODE physics. The bisection algorithm must perfectly locate the boundaries 
    without thrashing or overstepping into the opposite regime.
    """
    engine = Engine(model=BisectionHysteresis(), target="cpu", mock_execution=False)
    
    protocol = Sequence([
        CC(rate=100.0, until=engine.model.y >= 55.5),
        CC(rate=-100.0, until=engine.model.y <= 11.1)
    ])
    
    res = engine.solve(protocol=protocol, show_progress=False)
    
    t_final = res["Time [s]"].data[-1]
    y_final = res["y"].data[-1]
    
    # 15 bisection iterations on a 1.0s step yields a time precision of ~3.05e-5s.
    assert t_final == pytest.approx(0.999, abs=5e-5), \
        "Bisection root-finder failed to safely isolate and chain tight hysteresis bounds."
        
    # Because dy/dt = 100, the state precision is 100 * 3.05e-5 = 3.05e-3.
    # We set tolerance to 5e-3 to safely encompass the engine's theoretical limit.
    assert y_final == pytest.approx(11.1, abs=5e-3)

@REQUIRES_RUNTIME
def test_torture_graceful_degradation_segfault_guard():
    engine = Engine(model=UnsolvableParadox(), target="cpu", mock_execution=False)
    
    with pytest.raises(RuntimeError) as excinfo:
        engine.solve(t_span=(0, 1.0))
        
    err_str = str(excinfo.value)
    assert "NATIVE SOLVER CRASH" in err_str
    assert "Top Offenders" in err_str, "Crash report failed to isolate the unsolvable state."


@REQUIRES_RUNTIME
def test_protocol_orchestrator_subsecond_timing_bug():
    """
    PROBE: The `dt_step` Clamping Bug.
    Validates that `_orchestrate_sequence` strictly respects protocol step limits.
    """
    engine = Engine(model=SimpleTimingIntegrator(), target="cpu", mock_execution=False)
    
    steps = [CC(rate=1.0, time=0.01) for _ in range(5)]
    protocol = Sequence(steps)
    res = engine.solve(protocol=protocol, show_progress=False)
    t_final = res["Time [s]"].data[-1]
    
    assert t_final == pytest.approx(0.05, abs=1e-5), \
        f"BUG DETECTED: The Protocol Orchestrator ignored sub-second timing limits! " \
        f"Expected exactly 0.05s of simulation, but got {t_final:.2f}s. " \
        f"The orchestrator loop is missing `dt_step = min(1.0, t_max - t_elapsed)` clamping."


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])