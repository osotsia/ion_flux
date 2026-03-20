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