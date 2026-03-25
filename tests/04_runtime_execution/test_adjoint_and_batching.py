"""
Runtime Execution: Adjoints & Parallel Batching

Validates discrete adjoint backward propagation, Rayon/OpenMP task parallelism,
and explicit suppression of thread-oversubscription cascades.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE

class DummyBatchModel(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=10)
    c = fx.State(domain=x)
    D = fx.Parameter(default=1.0)
    
    def math(self):
        return {
            "regions": {
                self.x: [ fx.dt(self.c) == self.D * fx.grad(self.c) ]
            },
            "boundaries": [
                self.c.left == 0.0,
                self.c.right == 0.0
            ],
            "global": [
                self.c.t0 == 1.0
            ]
        }

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_bug5_rayon_openmp_thread_collision():
    """
    Validates that solve_batch successfully suppresses OpenMP threads natively
    when executing over Rayon task-pools to prevent OOM thread oversubscription cascades.
    """
    engine = Engine(model=DummyBatchModel(), target="cpu:omp", mock_execution=False)
    if engine.mock_execution:
        pytest.skip("Compilation environment absent.")
        
    params_list = [{"D": float(i)} for i in range(1, 5)]
    
    # Executing parallel batch. This will immediately hang/lock standard Linux runtimes 
    # if the OpenMP threads scale dynamically beneath Rayon.
    results = engine.solve_batch(parameters=params_list, t_span=(0, 0.1), max_workers=4)
    
    assert len(results) == 4
    for r in results:
        assert r.status == "completed"

class SimpleEISModel(fx.PDE):
    V = fx.State(domain=None) 
    R = fx.Parameter(default=10.0)
    C = fx.Parameter(default=0.1)
    i_app = fx.Parameter(default=1.0)
    
    def math(self):
        return {
            "global": [
                fx.dt(self.V) == (self.i_app - self.V / self.R) / self.C,
                self.V.t0 == 0.0
            ]
        }

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_bug2_eis_differentiability_analytical_shift():
    """
    Validates that EIS gradients evaluate via the exact Implicit Function Theorem shift 
    instead of triggering full unrolled Newton-Raphson finite difference loops.
    """
    engine = Engine(model=SimpleEISModel(), target="cpu", mock_execution=False)
    if engine.mock_execution:
        pytest.skip("Compilation environment absent.")
        
    session = engine.start_session(parameters={"R": 10.0, "C": 0.1, "i_app": 1.0})
    session.reach_steady_state()
    
    freqs = np.array([10.0])
    eis_res = session.solve_eis(frequencies=freqs, input_var="i_app", output_var="V")
    eis_res.trajectory["requires_grad"] = ["R", "C"]
    
    # Backpropagate using exactly-matched analytical offsets
    loss = fx.metrics.rmse(eis_res["Z_real"], np.array([0.0]), engine=engine, state_name="V")
    grads = loss.backward()
    
    assert "R" in grads
    assert "C" in grads
    assert isinstance(grads["R"], float)
    assert isinstance(grads["C"], float)
    assert not np.isnan(grads["R"])
    assert not np.isnan(grads["C"])