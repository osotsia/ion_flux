"""
TSPMe Diagnostic Oracles (Round 2)

This suite probes the parameter scaling, AST spatial broadcasting, and 
electrolyte depletion limits to identify the root cause of the erratic 
voltage jumps and premature 1.6 Ah capacity termination.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

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

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

# ==============================================================================
# ORACLE 1: Particle Depletion Rate (The 1.6 Ah Limit)
# ==============================================================================

class DepletionRateProbe(fx.PDE):
    """
    Simulates a single particle undergoing 1C discharge to measure exact depletion time.
    Compares the geometric a_n derivation vs the Table 1 paper value.
    """
    r = fx.Domain(bounds=(0, 5.86e-6), resolution=10, coord_sys="spherical")
    c_geo = fx.State(domain=r, name="c_geo")
    c_tab = fx.State(domain=r, name="c_tab")
    
    c_geo_avg = fx.State(domain=None, name="c_geo_avg")
    c_tab_avg = fx.State(domain=None, name="c_tab_avg")
    
    def math(self):
        F = 96485.0
        A_elec = 0.10269  # m^2 (Derived from 5Ah / 48.69 A/m^2)
        L_n = 85.2e-6
        i_app = 5.0       # 1C discharge
        j_vol = (i_app / A_elec) / L_n
        
        # Area scaling values
        a_geo = 3.0 * 0.25 / 5.86e-6  # 1.28e5 (My previous script)
        a_tab = 3.84e5                # (Paper Table 1)
        
        flux_geo = -1e-14 * fx.grad(self.c_geo, axis=self.r)
        flux_tab = -1e-14 * fx.grad(self.c_tab, axis=self.r)
        
        vol = (4.0/3.0) * np.pi * (5.86e-6)**3
        
        return {
            "equations": {
                self.c_geo: fx.dt(self.c_geo) == -fx.div(flux_geo, axis=self.r),
                self.c_tab: fx.dt(self.c_tab) == -fx.div(flux_tab, axis=self.r),
                self.c_geo_avg: self.c_geo_avg == fx.integral(self.c_geo, over=self.r) / vol,
                self.c_tab_avg: self.c_tab_avg == fx.integral(self.c_tab, over=self.r) / vol
            },
            "boundaries": {
                flux_geo: {"left": 0.0, "right": j_vol / (a_geo * F)},
                flux_tab: {"left": 0.0, "right": j_vol / (a_tab * F)}
            },
            "initial_conditions": {
                self.c_geo: 29866.0,
                self.c_tab: 29866.0,
                self.c_geo_avg: 29866.0,
                self.c_tab_avg: 29866.0
            }
        }

@REQUIRES_COMPILER
def test_depletion_rate_scaling():
    """
    PROBE: Proves that the geometric calculation of `a_n` forces the particle 
    to deplete ~3x faster than the paper's tabulated value, causing the 1.6 Ah crash.
    """
    engine = Engine(model=DepletionRateProbe(), target="cpu", mock_execution=False)

    # 1.8 Ah at 1C (5 Amps) takes 1.8 Ah / 5 A = 0.36 hours.
    t_eval = np.linspace(0, 0.36 * 3600, 50)
    res = engine.solve(t_eval=t_eval)

    c_geo_final = res["c_geo_avg"].data[-1]
    c_tab_final = res["c_tab_avg"].data[-1]

    # c_geo depletes 3x faster, hitting 0 before 1.8 Ah.
    assert c_geo_final < 0.0, "Geometric scaling failed to fully deplete the particle."
    # c_tab correctly holds ~2/3 of its concentration.
    assert c_tab_final > 10000.0, "Tabulated scaling unexpectedly depleted."
    assert c_tab_final > c_geo_final * 2.5, "The depletion rates do not diverge by the expected 3x factor."


# ==============================================================================
# ORACLE 2: AST Spatial Broadcasting (The Erradic Voltage)
# ==============================================================================

class DimensionalBroadcastingProbe(fx.PDE):
    """
    Isolates what the AST compiler does when a 1D spatial array is directly 
    assigned to a 0D scalar state without an integration wrapper.
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    spatial_field = fx.State(domain=x, name="spatial_field")
    
    # 0D target states
    v_target = fx.State(domain=None, name="v_target")
    
    def math(self):
        return {
            "equations": {
                self.spatial_field: fx.dt(self.spatial_field) == 0.0,
                # BUG INJECTION: Directly assigning 1D to 0D
                self.v_target: self.v_target == self.spatial_field
            },
            "boundaries": {},
            "initial_conditions": {
                self.spatial_field: self.x.coords * 10.0,  # Field is [0.0, 1.1, 2.2, ..., 10.0]
                self.v_target: 0.0
            }
        }

@REQUIRES_COMPILER
def test_ast_1d_to_0d_broadcasting_behavior():
    """
    PROBE: Determines if the compiler silently extracts the first index (0.0) 
    when broadcasting 1D to 0D. This proves why the overpotential in the previous 
    script produced wild jumps, as it only tracked the boundary node of the electrolyte.
    """
    engine = Engine(model=DimensionalBroadcastingProbe(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0)
    
    off_v, _ = engine.layout.state_offsets["v_target"]
    
    # Residual = ydot - rhs = 0.0 - rhs. Therefore rhs = -residual.
    rhs_eval = -res[off_v]
    
    # If rhs_eval is exactly 0.0, the compiler silently grabbed index 0 of the spatial field.
    assert rhs_eval == pytest.approx(0.0), \
        f"Compiler Broadcast Logic: Expected silent index-0 extraction, got {rhs_eval}."


# ==============================================================================
# ORACLE 3: Electrolyte Depletion Limits
# ==============================================================================

class ElectrolyteDepletionProbe(fx.PDE):
    """
    Runs the pure SPMe electrolyte diffusion equations at 2C to check if 
    it hits absolute zero.
    """
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=50)
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=25, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=5, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=20, name="x_p")
    
    c_e = fx.State(domain=cell, name="c_e")
    
    def math(self):
        F, t_plus = 96485.0, 0.2594
        De = 3e-10
        
        # 2C Current Density
        A_elec = 0.10269
        i_app = 10.0
        i_den = i_app / A_elec
        
        j_n = i_den / 85.2e-6
        j_p = -i_den / 75.6e-6
        
        flux_n = -De * (0.25**1.5) * fx.grad(self.c_e)
        flux_s = -De * (0.47**1.5) * fx.grad(self.c_e)
        flux_p = -De * (0.335**1.5) * fx.grad(self.c_e)
        
        return {
            "equations": {
                self.c_e: fx.Piecewise({
                    self.x_n: 0.25 * fx.dt(self.c_e) == -fx.div(flux_n) + (1.0 - t_plus) * j_n / F,
                    self.x_s: 0.47 * fx.dt(self.c_e) == -fx.div(flux_s),
                    self.x_p: 0.335 * fx.dt(self.c_e) == -fx.div(flux_p) + (1.0 - t_plus) * j_p / F
                })
            },
            "boundaries": {
                flux_n: {"left": 0.0},
                flux_p: {"right": 0.0}
            },
            "initial_conditions": {
                self.c_e: 1000.0
            }
        }

@REQUIRES_COMPILER
def test_electrolyte_depletion_at_2c():
    """
    PROBE: Checks if the electrolyte concentration physically hits 0.0 during a 2C 
    discharge. If it does, it explains the catastrophic jump in reaction heating 
    and validates why SPMe models generally fail at high C-rates.
    """
    engine = Engine(model=ElectrolyteDepletionProbe(), target="cpu", mock_execution=False)
    
    # Run for 15 minutes (0.5 Ah at 2C, exactly where the jump occurred in the plot)
    t_eval = np.linspace(0, 900, 50)
    res = engine.solve(t_eval=t_eval)
    
    c_e_final = res["c_e"].data[-1]
    min_c_e = np.min(c_e_final)
    
    # If the electrolyte drops below 10.0, the `sqrt(c_e)` in the exchange current 
    # density will cause extreme overpotentials.
    assert min_c_e > 10.0, f"Electrolyte depleted to {min_c_e:.1f} mol/m^3. SPMe is breaking down at 2C."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])