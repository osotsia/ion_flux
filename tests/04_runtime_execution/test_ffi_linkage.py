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
    

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])