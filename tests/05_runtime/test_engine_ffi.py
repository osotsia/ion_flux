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
    from ion_flux._core import solve_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)
import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime import session, _3_dispatch
from ion_flux import metrics
from ion_flux.protocols import Sequence, CC



def test_native_ffi_loaded():
    """
    Ensures that the Rust FFI extension module (_core.so) successfully loaded.
    Catches silent dynamic linking failures (e.g., missing .dylib/.so dependencies).
    """
    assert session.RUST_FFI_AVAILABLE is True, \
        "Rust FFI failed to load in session.py! Check for silent ImportErrors (e.g., missing dynamic libraries)."
    assert _3_dispatch.RUST_FFI_AVAILABLE is True, \
        "Rust FFI failed to load in _3_dispatch.py!"
    assert metrics.RUST_FFI_AVAILABLE is True, \
        "Rust FFI failed to load in metrics.py!"



def test_engine_rejects_mock_execution():
    """
    Ensures the Engine does not silently degrade to mock execution 
    yielding flat/dummy arrays when native execution is expected.
    """
    class Minimal(fx.PDE):
        V_cell = fx.State(name="V_cell")
        i_app = fx.State(name="i_app")
        terminal = fx.Terminal(current=i_app, voltage=V_cell)
        
        def math(self):
            return {
                "equations": {self.V_cell: self.V_cell == 4.182 - self.i_app},
                "boundaries": {},
                "initial_conditions": {self.V_cell: 4.182, self.i_app: 0.0}
            }

    engine = fx.Engine(model=Minimal(), target="cpu:serial")
    
    # 1. Verify the engine isn't explicitly flagged for mock execution
    assert engine.mock_execution is False

    # 2. Execute a brief protocol
    res = engine.solve(protocol=Sequence([CC(rate=1.0, time=10)]), show_progress=False)
    
    # 3. In mock execution, V_cell stays exactly at 4.182 and i_app stays at 0.0.
    # In a real native solve, V_cell should drop to 3.182 (4.182 - 1.0) and i_app should be 1.0.
    i_app_data = res["i_app"].data
    
    # If the array is entirely 0.0, the native solver was bypassed
    assert np.any(i_app_data > 0.5), \
        "Engine silently degraded to mock execution! The trajectory data remained frozen at initial conditions."


@REQUIRES_RUNTIME
def test_runtime_diagnostics_exposure():
    """
    Native solver diagnostics are surfaced to Python on 
    SimulationResult (single and batch solves) and stateful Session.
    """
    class SimpleODE(fx.PDE):
        y = fx.State(domain=None, name="y")
        def math(self):
            return {
                "equations": {self.y: fx.dt(self.y) == -self.y},
                "boundaries": {},
                "initial_conditions": {self.y: 1.0}
            }

    engine = fx.Engine(model=SimpleODE(), target="cpu:serial")

    # 1. Single solve diagnostics
    res = engine.solve(t_span=(0, 1.0), show_progress=False)
    assert hasattr(res, "diagnostics")
    diag = res.diagnostics
    assert isinstance(diag, dict)
    assert diag["total_steps"] > 0
    assert diag["accepted_steps"] > 0
    assert diag["total_steps"] == diag["accepted_steps"] + diag["rejected_steps"]
    assert diag["newton_iterations"] >= diag["total_steps"]
    assert diag["jacobian_assembly_time_us"] >= 0
    assert diag["linear_solve_time_us"] >= 0
    assert diag["residual_time_us"] >= 0

    # 2. Batch solve diagnostics
    batch_res = engine.solve_batch(parameters=[{}, {}], t_span=(0, 1.0), max_workers=2, show_progress=False)
    for r in batch_res:
        assert isinstance(r.diagnostics, dict)
        assert r.diagnostics["total_steps"] > 0
        assert r.diagnostics["accepted_steps"] > 0

    # 3. Stateful session diagnostics
    session = engine.start_session()
    assert hasattr(session, "diagnostics")
    initial_diag = session.diagnostics
    assert isinstance(initial_diag, dict)

    session.step(0.1)
    stepped_diag = session.diagnostics
    assert stepped_diag["total_steps"] > 0
    assert stepped_diag["accepted_steps"] > 0


def test_static_telemetry_profiling():
    """
    TelemetryReport produces verified topological properties, exact 
    non-zero counts, CPR schedule counts, and memory footprint estimates.
    """
    class MultiStateDAE(fx.PDE):
        x = fx.Domain(bounds=(0, 1.0), resolution=5, name="x")
        c = fx.State(domain=x, name="c")
        v = fx.State(domain=None, name="v")
        i_app = fx.State(domain=None, name="i_app")
        terminal = fx.Terminal(current=i_app, voltage=v)

        def math(self):
            flux = -fx.grad(self.c)
            return {
                "equations": {
                    self.c: fx.dt(self.c) == -fx.div(flux),
                    self.v: self.v == self.c.right - self.i_app * 0.1
                },
                "boundaries": {
                    flux: {"left": 0.0, "right": self.i_app}
                },
                "initial_conditions": {
                    self.c: 1.0, self.v: 1.0, self.i_app: 0.0
                }
            }

    engine = fx.Engine(model=MultiStateDAE(), target="cpu:serial")
    report = engine.telemetry

    assert report is not None
    assert report.n_states == 7  # 5 spatial + 1 voltage + 1 current
    assert report.model_len == 7
    assert report.n_params >= 3  # _term_mode, _term_i_target, _term_v_target
    assert report.nnz > 0
    assert 0.0 <= report.sparsity < 1.0
    assert report.cpr_colors > 0
    assert report.memory_footprint_bytes > 0
    assert "Sparse LU" in report.solver_type

    summary = report.summary()
    assert summary["n_states"] == 7
    assert summary["nnz"] == report.nnz
    assert summary["memory_footprint_bytes"] == report.memory_footprint_bytes


@REQUIRES_RUNTIME
def test_hardware_and_memory_telemetry(capsys):
    """
    Validates Phase 3: Physical memory accounting (peak_rss_bytes) and
    optional hardware PMU metrics are reported and print_summary() renders cleanly.
    """
    class SimpleODE(fx.PDE):
        y = fx.State(domain=None, name="y")
        def math(self):
            return {
                "equations": {self.y: fx.dt(self.y) == -self.y},
                "boundaries": {},
                "initial_conditions": {self.y: 1.0}
            }

    engine = fx.Engine(model=SimpleODE(), target="cpu:serial")
    res = engine.solve(t_span=(0, 1.0), show_progress=False)

    diag = res.diagnostics
    assert "peak_rss_bytes" in diag
    assert diag["peak_rss_bytes"] > 0  # Process must consume memory on any host OS
    assert "cpu_cycles" in diag
    assert "instructions" in diag

    # Verify dashboard rendering
    res.print_summary()
    captured = capsys.readouterr().out
    assert "ION_FLUX EXECUTION & DIAGNOSTIC SUMMARY" in captured
    assert "Solver Dynamics:" in captured
    assert "Peak Resident Memory:" in captured
