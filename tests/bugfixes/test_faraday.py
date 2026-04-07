"""
O'Regan Paper Bug Oracle: Hardcoded Faraday Specific Area

This suite isolates a severe bug in the `8_OReaganPaper.py` example script.
The script hardcodes the Faraday specific surface area conversions (`aF_n` and `aF_p`)
using a simplified radius (`5e-6`) and completely omits the porosity (`eps_s`).
This causes the flux boundary condition `j_n / aF_n` to severely underestimate 
the true lithium depletion rate of the particles, violating mass conservation 
and artificially preventing surface saturation at high C-rates (causing the 
underprediction of heat at 2C observed in the PNG).
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

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)

class FaradayBugOracle(fx.PDE):
    x = fx.Domain(bounds=(0, 1.0), resolution=5, name="x")
    r = fx.Domain(bounds=(0, 5.86e-6), resolution=5, coord_sys="spherical", name="r")
    
    c_s = fx.State(domain=x * r, name="c_s")
    c_e = fx.State(domain=x, name="c_e")
    
    def math(self):
        F = 96485.0
        
        # FIXED: Include eps_sn (0.75) and use the exact radius 5.86e-6
        a_n = 3.0 * 0.75 / 5.86e-6
        aF_n_correct = a_n * F 
        
        j_n = 10000.0 # Constant volumetric current [A/m^3]
        
        N_s_n = -1e-14 * fx.grad(self.c_s, axis=self.r)
        
        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(N_s_n, axis=self.r),
                # Electrolyte perfectly conserves the true faradaic current
                self.c_e: fx.dt(self.c_e) == j_n / F
            },
            "boundaries": {
                # Solid boundary is subjected to the correct scaling
                N_s_n: {"left": 0.0, "right": j_n / aF_n_correct}
            },
            "initial_conditions": {
                self.c_s: 1000.0,
                self.c_e: 0.0
            }
        }

@REQUIRES_RUNTIME
def test_faraday_mass_conservation_bug():
    """
    PROBE: Proves that the correct `aF_n` scaling guarantees global lithium 
    mass conservation. The amount of lithium entering the electrolyte will 
    perfectly match the amount leaving the solid particles.
    """
    model = FaradayBugOracle()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    # 1. Total Lithium added to the electrolyte
    c_e_final = res["c_e"].data[-1]
    Li_added_e = np.mean(c_e_final) * 1.0 # length is 1.0
    
    # 2. Total Lithium removed from the solid
    c_s_initial = res["c_s"].data[0]
    c_s_final = res["c_s"].data[-1]
    
    # Calculate exact FVM volumes for spherical coordinates
    R_p = 5.86e-6
    dr = R_p / 4.0 # res=5 -> 4 intervals
    r_faces = np.linspace(0, R_p, 5)
    
    V_cells = np.zeros(5)
    for i in range(5):
        r_right = r_faces[i] + 0.5*dr if i < 4 else r_faces[i]
        r_left = r_faces[i] - 0.5*dr if i > 0 else 0.0
        V_cells[i] = (4.0/3.0) * np.pi * (r_right**3 - r_left**3)
        
    V_particle = (4.0/3.0) * np.pi * R_p**3
    
    # Calculate average concentration drop per particle
    c_s_drop = c_s_initial - c_s_final
    c_s_drop_2d = c_s_drop.reshape((5, 5))
    
    avg_drop_per_particle = np.sum(c_s_drop_2d * V_cells, axis=1) / V_particle
    
    # Total Li removed = avg_drop * eps_s * V_macro
    # The true porosity used in the paper for the negative electrode
    eps_s = 0.75 
    Li_removed_s = np.mean(avg_drop_per_particle) * eps_s * 1.0
    
    # Conservation must hold exactly
    assert np.isclose(Li_added_e, Li_removed_s, rtol=1e-3), \
        f"Faraday Bug Confirmed! Electrolyte received {Li_added_e:.5f} mol, " \
        f"but Solid only lost {Li_removed_s:.5f} mol. Mass is not conserved. " \
        f"Ratio: {Li_removed_s / Li_added_e:.3f}"