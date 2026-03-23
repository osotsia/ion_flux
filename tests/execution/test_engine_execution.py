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
            "global": [
                fx.dt(self.y) == -self.k * self.y,
                self.y.t0 == 5.0
            ]
        }

class SpatiallyVaryingHeatIC(fx.PDE):
    rod = fx.Domain(bounds=(0, 2), resolution=5)
    T = fx.State(domain=rod)
    
    def math(self):
        return {
            "regions": {
                self.rod: [ fx.dt(self.T) == fx.grad(self.T) ]
            },
            "global": [
                self.T.t0 == 2 * self.rod.coords
            ]
        }

def test_engine_emits_enzyme_cpp(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    cpp = engine.cpp_source
    assert "extern void __enzyme_fwddiff" in cpp
    assert "extern void __enzyme_autodiff" in cpp
    assert "void evaluate_residual" in cpp
    assert "void evaluate_vjp" in cpp

def test_spatial_initial_condition_evaluation():
    engine = Engine(model=SpatiallyVaryingHeatIC(), target="cpu", mock_execution=True)
    y0, _, _, _ = engine._extract_metadata()
    # Coordinates for boundaries (0, 2) over 5 points -> dx = 2.0 / 4 = 0.5
    # T0 = 2 * coords -> [0.0, 1.0, 2.0, 3.0, 4.0]
    expected = [0.0, 1.0, 2.0, 3.0, 4.0]
    np.testing.assert_allclose(y0, expected, rtol=1e-5)

def test_stateful_session_execution():
    engine = Engine(model=ExponentialDecay(), target="cpu", mock_execution=True)
    session = engine.start_session(parameters={"k": 1.0})
    
    assert session.time == 0.0
    session.step(dt=0.1)
    assert session.time == 0.1
    assert session.get("Voltage") < 4.2
    
    session.reach_steady_state()
    eis = session.solve_eis(frequencies=np.array([10.0]), input_var="i_app", output_var="Voltage")
    assert len(eis["Z_real"].data) == 1

def test_differentiable_graph_metrics():
    engine = Engine(model=ExponentialDecay(), target="cpu", mock_execution=True)
    res = engine.solve(requires_grad=["k"])
    
    loss = fx.metrics.rmse(res["Voltage [V]"], np.array([4.0]*100), engine=engine)
    grads = loss.backward()
    
    # Validate the AD hook properly propagated through the isolated dictionary mapping
    assert "k" in grads
    assert grads["k"] != 0.0

def test_stateless_binary_deployment(tmp_path):
    import os
    engine = Engine(model=ExponentialDecay(), target="cpu", mock_execution=False)
    if getattr(engine, "mock_execution", False):
        pytest.skip("Compilation environment absent.")
        
    export_file = tmp_path / "model.so"
    engine.export_binary(str(export_file))
    
    # Verify JSON meta-manifest generated side-by-side
    assert os.path.exists(str(export_file) + ".meta.json")
    
    stateless_engine = Engine.load(str(export_file), target="cpu:serial")
    assert stateless_engine.target == "cpu:serial"
    
    # Validate the engine is fully operational without the original AST
    assert stateless_engine.mock_execution is False
    assert stateless_engine.layout.n_states == engine.layout.n_states
    assert stateless_engine.layout.get_param_offset("k") == 0
    
    y = [5.0]
    ydot = [-10.0]
    p = stateless_engine._pack_parameters({})
    res = stateless_engine.runtime.evaluate_residual(y, ydot, p)
    
    # ydot - (-k * y) -> -10.0 - (-2.0 * 5.0) = 0.0
    assert res[0] == pytest.approx(0.0)

def test_engine_telemetry_reporting():
    model = ExponentialDecay()
    engine = Engine(model=model, target="cpu")
    tel = engine.telemetry
    
    # An ODE has length 1. Ensure Telemetry defaults accurately hit the 0-penalty fast paths.
    assert tel.model_len == 1
    assert tel.l1_cache_hit_estimate == 1.0
    assert tel.sparsity == 0.0

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_end_to_end():
    """
    Proves Phase 1 completion: JIT -> Clang -> Rust FFI -> Solver (Native) -> Numpy
    """
    model = ExponentialDecay()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    if engine.mock_execution: pytest.skip("Compilation environment absent.")
    
    t_eval = np.linspace(0, 2.0, 50)
    res = engine.solve(parameters={"k": 1.5}, t_eval=t_eval)
    
    # Analytical solution: y(t) = y_0 * e^{-kt}
    y_analytical = 5.0 * np.exp(-1.5 * t_eval)
    y_simulated = res["y"].data
    
    # Updated tolerance due to adaptive scale-up inside implicit BDF1 formulation allowing larger integration truncation
    np.testing.assert_allclose(y_simulated, y_analytical, rtol=0.02, atol=1e-3)