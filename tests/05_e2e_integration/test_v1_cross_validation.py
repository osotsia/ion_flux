"""
E2E Integration: V1 Cross Validation

Direct transcription of legacy PyBaMM/V1 logic mapped to V2 to mathematically
validate backwards-compatible conservation and CCCV handling.
Cross-validates the native solver engine against SUNDIALS.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, CV
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE

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


@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native C++ toolchain.")
def test_v1_spm_discharge_conservation_cross_validation():
    """
    Translates V1's `TestSPM::test_conservation` and `TestVoltage`.
    Validates that concentration bounds are maintained, voltage drops monotonically,
    and the native solver trajectory perfectly aligns with SUNDIALS.
    """
    model = ValidatedSPM()
    engine_native = Engine(model=model, target="cpu:serial", solver_backend="native")
    engine_sundials = Engine(model=model, target="cpu:serial", solver_backend="sundials")
    
    protocol = Sequence([CC(rate=2.0, time=500)])

    res_native = engine_native.solve(protocol=protocol)
    res_sundials = engine_sundials.solve(protocol=protocol)
    
    assert res_native.status == "completed"
    assert res_sundials.status == "completed"
    
    V_n = res_native["V_cell"].data
    c_s_n = res_native["c_s"].data
    
    # 1. Voltage should monotonically decrease during CC discharge
    assert np.all(np.diff(V_n) <= 1e-10)
    
    # 2. Surface concentration should deplete faster than core concentration
    surface_c = c_s_n[:, -1]
    core_c = c_s_n[:, 0]
    assert np.all(surface_c <= core_c)
    
    # 3. Overall concentration must strictly decrease
    assert surface_c[-1] < surface_c[0]
    
    # 4. Cross-Validate Native vs. SUNDIALS IDAS
    np.testing.assert_allclose(res_native["V_cell"].data, res_sundials["V_cell"].data, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(res_native["c_s"].data, res_sundials["c_s"].data, rtol=1e-3, atol=1e-2)


@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native C++ toolchain.")
def test_v1_experiment_cccv_hot_swapping_cross_validation():
    """
    Translates V1's `TestExperimentSteps::test_cccv`.
    Validates that both the native and SUNDIALS solvers can successfully hot-swap 
    differential/algebraic constraints mid-solve and land on the same asymptotes.
    """
    model = ValidatedSPM()
    engine_native = Engine(model=model, target="cpu:serial", solver_backend="native")
    engine_sundials = Engine(model=model, target="cpu:serial", solver_backend="sundials")
    
    # Time limits expanded to 5000s to allow the 1000s diffusion time constant 
    # to naturally hit the -0.1A trigger via the bisection event catcher.
    protocol = Sequence([
        CC(rate=-2.0, until=fx.Condition("V_cell >= 4.2"), time=5000),
        CV(voltage=4.2, until=fx.Condition("i_app >= -0.1"), time=5000)
    ])
    
    res_native = engine_native.solve(protocol=protocol)
    res_sundials = engine_sundials.solve(protocol=protocol)
    
    assert res_native.status == "completed"
    assert res_sundials.status == "completed"
    
    V_n = res_native["V_cell"].data
    i_app_n = res_native["i_app"].data
    
    # Validate that voltage successfully clamped at 4.2V without overshooting
    assert np.max(V_n) <= 4.2 + 1e-3
    assert np.isclose(V_n[-1], 4.2, atol=1e-3)
    
    # Validate the CV taper naturally triggered the -0.1A threshold
    assert i_app_n[-1] >= -0.1 - 1e-3
    
    # Cross-Validate Event Trigger and Final Asymptotes
    # (Checking the terminal boundary variables after bisection searches)
    assert res_native["Time [s]"].data[-1] == pytest.approx(res_sundials["Time [s]"].data[-1], rel=1e-3)
    assert res_native["V_cell"].data[-1] == pytest.approx(res_sundials["V_cell"].data[-1], rel=1e-3)
    assert res_native["i_app"].data[-1] == pytest.approx(res_sundials["i_app"].data[-1], rel=1e-3)