import pytest
import numpy as np
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, CV

class ValidatedSPM(fx.PDE):
    """
    A direct V2 transcription of V1's Single Particle Model core PDE 
    to validate numerical implicit solver behavior.
    """
    r = fx.Domain(bounds=(0.0, 1.0), resolution=30, coord_sys="spherical")
    
    c_s = fx.State(domain=r)
    V_cell = fx.State(domain=None)
    i_app = fx.State(domain=None)
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    D_s = fx.Parameter(default=1e-3)
    
    def math(self):
        flux = -self.D_s * fx.grad(self.c_s)
        
        # Simplified Butler-Volmer / OCV relationship for validation
        ocv = 4.2 - 0.5 * (1000.0 - self.c_s.right) / 1000.0
        
        return {
            "regions": {
                self.r: [ fx.dt(self.c_s) == -fx.div(flux) ]
            },
            "boundaries": [
                flux.left == 0.0,
                flux.right == self.i_app / 20.0
            ],
            "global": [
                self.V_cell == ocv - 0.05 * self.i_app,
                self.c_s.t0 == 500.0,
                self.V_cell.t0 == 3.95,
                self.i_app.t0 == 0.0
            ]
        }

def test_v1_spm_discharge_conservation():
    """
    Translates V1's `TestSPM::test_conservation` and `TestVoltage`.
    Validates that concentration bounds are maintained and voltage drops monotonically.
    """
    model = ValidatedSPM()
    engine = fx.Engine(model=model, target="cpu:serial")
    
    if getattr(engine, "mock_execution", False):
        pytest.skip("Native compiler toolchain missing. Cannot validate execution.")

    res = engine.solve(protocol=Sequence([CC(rate=2.0, time=500)]))
    
    assert res.status == "completed"
    
    V = res["V_cell"].data
    c_s = res["c_s"].data
    
    # 1. Voltage should monotonically decrease during CC discharge
    assert np.all(np.diff(V) <= 1e-10)
    
    # 2. Surface concentration should deplete faster than core concentration
    surface_c = c_s[:, -1]
    core_c = c_s[:, 0]
    assert np.all(surface_c <= core_c)
    
    # 3. Overall concentration must strictly decrease
    assert surface_c[-1] < surface_c[0]

def test_v1_experiment_cccv_hot_swapping():
    """
    Translates V1's `TestExperimentSteps::test_cccv`.
    Validates that the native solver can hot-swap differential/algebraic constraints mid-solve.
    """
    model = ValidatedSPM()
    engine = fx.Engine(model=model, target="cpu:serial")
    
    if getattr(engine, "mock_execution", False):
        pytest.skip("Native compiler toolchain missing.")

    # Added fallback `time` bounds to completely eliminate the risk of infinite loops.
    # Time limits expanded to 5000s to allow the 1000s diffusion time constant to naturally hit the -0.1A trigger.
    protocol = Sequence([
        CC(rate=-2.0, until=fx.Condition("V_cell >= 4.2"), time=5000),
        CV(voltage=4.2, until=fx.Condition("i_app >= -0.1"), time=5000)
    ])
    
    res = engine.solve(protocol=protocol)
    
    assert res.status == "completed"
    
    V = res["V_cell"].data
    i_app = res["i_app"].data
    
    # Validate that voltage successfully clamped at 4.2V without overshooting
    assert np.max(V) <= 4.2 + 1e-3
    assert np.isclose(V[-1], 4.2, atol=1e-3)
    
    # Validate the CV taper naturally triggered the -0.1A threshold
    assert i_app[-1] >= -0.1 - 1e-3