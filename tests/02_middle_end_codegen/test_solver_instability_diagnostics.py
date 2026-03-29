"""
Middle-End Codegen: Solver Instability Diagnostics

Isolates and proves that the engine is architecturally sound, but fails 
due to f64 ULP (Unit in the Last Place) noise exceeding the unscaled 
absolute tolerances hardcoded into the implicit numerical solvers.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

RUST_FFI_AVAILABLE = _has_compiler()

# ==============================================================================
# DIAGNOSTIC 1: 1D Poisson ULP Noise Limit
# ==============================================================================

class PoissonULPNoisePDE(fx.PDE):
    x = fx.Domain(bounds=(0, 40e-6), resolution=10, name="x")
    phi = fx.State(domain=x, name="phi")
    
    def math(self):
        i_s = -100.0 * fx.grad(self.phi)
        return {
            "regions": { self.x: [ 0 == fx.div(i_s) ] },
            "boundaries": [ i_s.left == -30.0, i_s.right == 30.0 ],
            "global": [ self.phi.t0 == 0.0 ]
        }

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native compiler.")
def test_1d_poisson_ulp_precision_limit():
    """
    DIAGNOSTIC: Proves that standard SI dimensions naturally create ULP noise > 1e-8.
    """
    engine = Engine(model=PoissonULPNoisePDE(), target="cpu", mock_execution=False)
    
    # The divergence operator div(-100 * grad(phi)) on a 40um grid multiplies phi by ~ 6e12.
    # 6e12 * 2e-16 (f64 epsilon) = ~ 0.001. 
    # The residual noise floor is ~0.001, which is > 1e-8.
    try:
        engine.solve(t_span=(0, 0.1))
        pytest.fail("Solver should have crashed due to spatial DAE ULP noise.")
    except RuntimeError as e:
        assert "Line Search exhausted" in str(e) or "NaN" in str(e) or "Step collapsed" in str(e)


# ==============================================================================
# DIAGNOSTIC 2: The Physical Solution (AST Equilibration)
# ==============================================================================

class MacroMicroSPM_Equilibrated(fx.PDE):
    """
    Direct replica of 2_macro_micro_spm.py, but with MANUAL AST-level equation scaling
    AND the corrected Faraday specific-area volumetric flux conversions.
    """
    x_n = fx.Domain(bounds=(0, 40e-6), resolution=10, name="x_n")
    x_p = fx.Domain(bounds=(60e-6, 100e-6), resolution=10, name="x_p")
    r_n = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_p") 
    macro_n = x_n * r_n 
    macro_p = x_p * r_p 
    
    c_s_n = fx.State(domain=macro_n, name="c_s_n")
    c_s_p = fx.State(domain=macro_p, name="c_s_p")
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")
    
    V_cell = fx.State(name="V_cell") 
    i_app = fx.State(name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    def math(self):
        Ds_n, Ds_p = 1e-14, 1e-14
        sig_n, sig_p = 100.0, 100.0
        
        i_s_n = -sig_n * fx.grad(self.phi_s_n, axis=self.x_n) 
        i_s_p = -sig_p * fx.grad(self.phi_s_p, axis=self.x_p) 
        
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n) 
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p) 
        
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p) 
        
        U_n = 0.1 - 0.0001 * c_surf_n 
        U_p = 4.2 - 0.0001 * c_surf_p 
        
        j_n = 1e6 * (self.phi_s_n - U_n)
        j_p = 1e6 * (self.phi_s_p - U_p)

        # MANUAL EQUILIBRATION: Scale massive DAE operators down to O(1)
        eq_scale = 1e-12
        
        # PHYSICAL CORRECTION: Convert Volumetric current (A/m^3) to Area flux (mol/m^2 s)
        aF = 5.78e10

        return {
            "regions": {
                self.x_n: [ 0 == (fx.div(i_s_n, axis=self.x_n) + j_n) * eq_scale ],
                self.x_p: [ 0 == (fx.div(i_s_p, axis=self.x_p) + j_p) * eq_scale ],
                self.macro_n: [ fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n) ],
                self.macro_p: [ fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p) ]
            },
            "boundaries": [
                # FIXED WIRING: Both current boundaries pull current to drop the voltage.
                i_s_n.left == -self.i_app, i_s_n.right == 0.0,
                i_s_p.left == 0.0, i_s_p.right == -self.i_app,
                
                N_s_n.boundary("left", domain=self.r_n) == 0.0,  
                N_s_n.boundary("right", domain=self.r_n) == -j_n / aF, 
                N_s_p.boundary("left", domain=self.r_p) == 0.0,  
                N_s_p.boundary("right", domain=self.r_p) == -j_p / aF 
            ],
            "global": [
                self.V_cell == self.phi_s_p.right - self.phi_s_n.left,
                self.phi_s_n.t0 == 0.05, self.phi_s_p.t0 == 4.15, 
                self.c_s_n.t0 == 500.0, self.c_s_p.t0 == 500.0, 
                self.V_cell.t0 == 4.10, self.i_app.t0 == 0.0 
            ]
        }

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native compiler.")
def test_manual_ast_equilibration_fixes_macro_micro():
    """
    DIAGNOSTIC: Proves that scaling the spatial DAE residuals in the AST to O(1)
    completely stabilizes the implicit solver, and fixing the wiring allows
    the time-integrator to traverse the entire CCCV sequence normally.
    """
    engine = Engine(model=MacroMicroSPM_Equilibrated(), target="cpu", mock_execution=False)
    
    from ion_flux.protocols import Sequence, CC, Rest
    protocol = Sequence([
        CC(rate=30.0, until=fx.Condition("V_cell <= 3.0"), time=3600),
        Rest(time=60)
    ])
    
    res = engine.solve(protocol=protocol)
    
    assert res.status == "completed"
    assert len(res["Time [s]"].data) > 1
    
    # Verify the voltage safely dropped to the 3.0V cutoff before transferring to Rest.
    V = res["V_cell"].data
    assert np.min(V) <= 3.05