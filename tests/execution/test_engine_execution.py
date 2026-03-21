import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE

class ExponentialDecay(fx.PDE):
    """
    y_dot = -k * y
    y(0) = 5.0
    """
    y = fx.State()
    k = fx.Parameter(default=2.0)
    
    def math(self):
        return {
            fx.dt(self.y): -self.k * self.y,
            self.y.t0: 5.0
        }

def test_engine_emits_enzyme_cpp(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    cpp = engine.cpp_source
    assert "extern void __enzyme_fwddiff" in cpp
    assert "void evaluate_residual" in cpp

def test_stateful_session_execution():
    engine = Engine(model=ExponentialDecay(), target="cpu", mock_execution=True)
    session = engine.start_session(parameters={"k": 1.0})
    
    assert session.time == 0.0
    session.step(dt=0.1)
    assert session.time == 0.1
    assert session.get("Voltage") < 4.2
    
    session.reach_steady_state()
    eis = session.solve_eis(frequencies=np.array([10.0]), input_var="i_app", output_var="Voltage")
    assert len(eis) == 1

def test_differentiable_graph_metrics():
    engine = Engine(model=ExponentialDecay(), target="cpu", mock_execution=True)
    res = engine.solve(requires_grad=["k"])
    
    loss = fx.metrics.rmse(res["Voltage [V]"].data, np.array([4.0]*100), engine=engine)
    loss.backward()
    
    # Validate the AD hook properly propagated to the engine's ParamHandle
    assert engine.parameters["k"].grad != 0.0

def test_stateless_binary_deployment(tmp_path):
    import os
    engine = Engine(model=ExponentialDecay(), target="cpu", mock_execution=False)
    if not engine.runtime:
        pytest.skip("Compilation environment absent.")
        
    export_file = tmp_path / "model.so"
    engine.export_binary(str(export_file))
    
    stateless_engine = Engine.load(str(export_file), target="cpu:serial")
    assert stateless_engine.target == "cpu:serial"
    assert stateless_engine.mock_execution is True

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_end_to_end():
    """
    Proves Phase 1 completion: JIT -> Clang -> Rust FFI -> Solver (Native) -> Numpy
    """
    model = ExponentialDecay()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    t_eval = np.linspace(0, 2.0, 50)
    res = engine.solve(parameters={"k": 1.5}, t_eval=t_eval)
    
    # Analytical solution: y(t) = y_0 * e^{-kt}
    y_analytical = 5.0 * np.exp(-1.5 * t_eval)
    y_simulated = res["y"].data
    
    np.testing.assert_allclose(y_simulated, y_analytical, rtol=1e-3, atol=1e-4)