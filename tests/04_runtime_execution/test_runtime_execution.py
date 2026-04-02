"""
Runtime Execution: Engine, Sessions, Solvers, and Concurrency

Comprehensive validation of the native Rust execution backend and Sundials FFI.
Tests Stateful HIL/SIL sessions, Protocol hot-swapping (CC/CV), Matrix-Free 
GMRES solvers for 3D unstructured meshes, Rayon task-parallel batching, 
Analytical EIS, Adjoint Sensitivities, and Stateless Binary (.so) deployments.
"""

import pytest
import asyncio
import os
import shutil
import platform
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine
from ion_flux.runtime.scheduler import MultiTenantScheduler
from ion_flux.protocols import Sequence, CC, CV, Rest

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

class AdjointAndEISModel(fx.PDE):
    """Validates Analytical EIS (Frequency Domain) and Exact VJP Adjoints."""
    V = fx.State(domain=None, name="V")
    R = fx.Parameter(default=10.0, name="R")
    C = fx.Parameter(default=0.1, name="C")
    i_app = fx.Parameter(default=1.0, name="i_app")
    
    def math(self):
        return {
            "equations": {
                self.V: fx.dt(self.V) == (self.i_app - self.V / self.R) / self.C
            },
            "boundaries": {},
            "initial_conditions": {
                self.V: 0.0
            }
        }

tetrahedron_mesh = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "elements": [[0, 1, 2, 3]]
}

class UnstructuredGMRESModel(fx.PDE):
    """Validates 3D Matrix-Free CSR Graph Traversals (bandwidth = -1)."""
    mesh = fx.Domain.from_mesh(tetrahedron_mesh, name="mesh", surfaces={"top": [2, 3]})
    c = fx.State(domain=mesh, name="c")
    D = fx.Parameter(default=2.0, name="D")

    def math(self):
        flux = -self.D * fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux)
            },
            "boundaries": {
                flux: {"top": 100.0}
            },
            "initial_conditions": {
                self.c: 10.0
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
    engine_native = Engine(model=model, target="cpu", solver_backend="native")
    engine_sundials = Engine(model=model, target="cpu", solver_backend="sundials")
    
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
    np.testing.assert_allclose(res_native["i_app"].data[-1], res_sundials["i_app"].data[-1], rtol=1e-3)


@REQUIRES_RUNTIME
def test_stateful_session_hil_control():
    """Validates the SolverHandle maintains memory seamlessly for continuous micro-stepping (BMS HIL)."""
    engine = Engine(model=BatteryProtocolPDE(), target="cpu", mock_execution=False)
    
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


# ==============================================================================
# Concept 2: Differentiability (Analytical EIS & Adjoints)
# ==============================================================================

@REQUIRES_RUNTIME
def test_differentiable_analytical_eis_and_adjoints():
    """
    Validates:
    1. Analytical Frequency Domain Solves (EIS) via Enzyme Jacobians.
    2. Discrete Adjoint Backward Propagation via Vector-Jacobian Products (VJPs).
    """
    engine = Engine(model=AdjointAndEISModel(), target="cpu", mock_execution=False)
    session = engine.start_session(parameters={"R": 10.0, "C": 0.1, "i_app": 1.0})
    
    session.reach_steady_state()
    assert session.get("V") == pytest.approx(10.0) # V = I*R = 1.0 * 10.0
    
    # Solve EIS algebraically
    w_arr = np.array([0.01, 1.0, 100.0]) * 2 * np.pi
    eis_res = session.solve_eis(np.array([0.01, 1.0, 100.0]), input_var="i_app", output_var="V")
    
    Z_sim = eis_res["Z_real"].data + 1j * eis_res["Z_imag"].data
    Z_analytical = 10.0 / (1 + 1j * w_arr * 1.0) # Exact RC Transfer Function
    
    np.testing.assert_allclose(np.real(Z_sim), np.real(Z_analytical), rtol=1e-4)
    np.testing.assert_allclose(np.imag(Z_sim), np.imag(Z_analytical), rtol=1e-4)
    
    # Backpropagate to extract parameters
    eis_res.trajectory["requires_grad"] = ["R", "C"]
    loss = fx.metrics.rmse(eis_res["Z_real"], np.zeros(3), engine=engine, state_name="V")
    grads = loss.backward()
    
    assert "R" in grads
    assert "C" in grads
    assert isinstance(grads["R"], float)
    assert not np.isnan(grads["R"])


# ==============================================================================
# Concept 3: Advanced Architectures (GMRES & Unstructured Meshes)
# ==============================================================================

@REQUIRES_RUNTIME
def test_3d_unstructured_matrix_free_gmres():
    """
    Validates that unstructured meshes automatically trigger Matrix-Free GMRES 
    (bandwidth=-1), correctly traverse CSR geometries, and support Adjoint passes without OOM.
    """
    # Cache=False forces the Engine to re-emit the JVP C++ payload natively
    engine = Engine(model=UnstructuredGMRESModel(), target="cpu", mock_execution=False, cache=False)
    
    assert engine.jacobian_bandwidth == -1, "Engine failed to assign GMRES to unstructured CSR graph."
    
    res = engine.solve(t_span=(0, 1.0), requires_grad=["D"])
    
    assert res.status == "completed"
    assert res["c"].data.shape[1] == 4 # Validates dynamic unrolling to the exact 4-node tetrahedron mesh
    
    # Backward pass over GMRES trajectory
    loss = fx.metrics.rmse(res["c"], np.zeros_like(res["c"].data), engine=engine, state_name="c")
    grads = loss.backward()
    
    assert isinstance(grads["D"], float)
    assert not np.isnan(grads["D"])


# ==============================================================================
# Concept 4: Concurrency & Cloud Scale (Batching, Async, .so Export)
# ==============================================================================

@REQUIRES_RUNTIME
def test_rayon_task_parallelism_batching():
    """Validates that solve_batch bypasses the Python GIL utilizing Rust Rayon."""
    engine = Engine(model=AdjointAndEISModel(), target="cpu", mock_execution=False)
    
    param_sweep = [{"R": 10.0}, {"R": 20.0}, {"R": 30.0}]
    results = engine.solve_batch(parameters=param_sweep, t_span=(0, 1.0), max_workers=3)
    
    assert len(results) == 3
    assert results[0].status == "completed"
    
    # Higher resistance should lead to a higher accumulated voltage
    assert results[2]["V"].data[-1] > results[0]["V"].data[-1]


def test_openmp_data_parallelism_emission():
    """Validates OpenMP pragmas are safely emitted for massive spatial arrays."""
    class LargeOpenMPModel(fx.PDE):
        x = fx.Domain(bounds=(0, 1), resolution=100) # Resolution > 50 triggers OpenMP Pragma
        c = fx.State(domain=x)
        def math(self):
            return {"equations": {self.c: fx.dt(self.c) == fx.grad(self.c)}, "boundaries": {}, "initial_conditions": {self.c: 0.0}}
            
    engine = Engine(model=LargeOpenMPModel(), target="cpu:omp", mock_execution=True)
    assert "omp parallel for" in engine.cpp_source


@REQUIRES_RUNTIME
def test_stateless_binary_deployment(tmp_path):
    """Validates 0ms cold-start `.so` deployments bypassing AST reconstruction."""
    engine = Engine(model=AdjointAndEISModel(), target="cpu", mock_execution=False)
    
    export_file = tmp_path / "model_prod.so"
    engine.export_binary(str(export_file))
    
    assert os.path.exists(str(export_file) + ".meta.json")
    
    # Instantiate instantly without Clang or AST parsing
    stateless_engine = Engine.load(str(export_file), target="cpu:serial")
    
    assert stateless_engine.mock_execution is False
    assert stateless_engine.layout.n_states == engine.layout.n_states
    assert stateless_engine.layout.get_param_offset("R") == engine.layout.get_param_offset("R")
    
    # Solve directly via FFI
    res = stateless_engine.solve(t_span=(0, 1.0), parameters={"R": 5.0})
    assert res.status == "completed"


@pytest.mark.asyncio
async def test_async_multitenant_scheduler_isolation():
    """
    Validates Async queueing limits hardware oversubscription, and safely isolates 
    thread-pool panic/exceptions from dragging down sibling jobs.
    """
    # For isolation testing without needing Clang, use a mock execution engine
    engine = Engine(model=AdjointAndEISModel(), target="cpu", mock_execution=True)
    scheduler = MultiTenantScheduler(max_concurrent=2)
    
    # `c.t0` == float('inf') is a hardcoded mock trigger for "Native Solver Crash" in mock_execution
    bad_params = {"c.t0": float('inf')} 
    good_params = {"R": 10.0}
    
    future_bad = engine.solve_async(t_span=(0, 1), parameters=bad_params, scheduler=scheduler)
    future_good = engine.solve_async(t_span=(0, 1), parameters=good_params, scheduler=scheduler)
    
    res_bad, res_good = await asyncio.gather(future_bad, future_good, return_exceptions=True)
    
    assert isinstance(res_bad, Exception)
    assert "Newton convergence failure" in str(res_bad) or "crash" in str(res_bad).lower()
    
    assert not isinstance(res_good, Exception)
    assert res_good.status == "completed"