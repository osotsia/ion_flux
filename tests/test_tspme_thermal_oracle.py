"""
TSPMe Thermal Source Oracle

Extracts the exact evaluated magnitudes of Q_s, Q_e, Q_irr, and Q_tot 
from the AST at t=0 for a 2C discharge to determine why the temperature 
only reaches 31.5°C instead of the physically accurate 48°C.
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
# Thermal Source Probe Model
# ==============================================================================

class ThermalSourceProbe(fx.PDE):
    """
    Isolates the exact overpotential and heating calculations from the TSPMe 
    script to evaluate their raw magnitudes at t=0.
    """
    x_n = fx.Domain(bounds=(0, 85.2e-6), resolution=71, name="x_n")
    x_p = fx.Domain(bounds=(97.2e-6, 172.8e-6), resolution=63, name="x_p")
    r_n = fx.Domain(bounds=(0, 5.86e-6), resolution=15, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5.22e-6), resolution=15, coord_sys="spherical", name="r_p") 
    
    # We must bind states to the domains to ensure fx.integral() retains spatial context
    c_e_n = fx.State(domain=x_n, name="c_e_n")
    c_e_p = fx.State(domain=x_p, name="c_e_p")
    c_s_n = fx.State(domain=r_n, name="c_s_n")
    c_s_p = fx.State(domain=r_p, name="c_s_p")
    
    T_cell = fx.State(domain=None, name="T_cell")
    
    # 0D Output Trackers
    Q_s_out = fx.State(domain=None, name="Q_s_out")
    Q_e_out = fx.State(domain=None, name="Q_e_out")
    Q_irr_out = fx.State(domain=None, name="Q_irr_out")
    Q_tot_out = fx.State(domain=None, name="Q_tot_out")
    eta_r_out = fx.State(domain=None, name="eta_r_out")
    
    def math(self):
        F, R_const = 96485.0, 8.314
        L_n, L_s, L_p = 85.2e-6, 12.0e-6, 75.6e-6
        L_cell = L_n + L_s + L_p
        
        eps_n, eps_s, eps_p = 0.25, 0.47, 0.335
        eps_s_n, eps_s_p = 0.75, 0.665
        b_brug = 1.5
        
        a_n = 3.0 * eps_s_n / 5.86e-6
        a_p = 3.0 * eps_s_p / 5.22e-6
        c_max_n, c_max_p = 33133.0, 63104.0
        sig_n, sig_p = 215.0, 0.18
        m_n, m_p = 6.48e-7, 3.42e-6
        
        sig_e_ref, t_plus = 1.0, 0.2594
        
        # 2C Discharge
        i_app = 10.0
        A_elec = 0.1024 
        i_den = i_app / A_elec
        
        c_surf_n = fx.min(fx.max(self.c_s_n.boundary("right", domain=self.r_n), 10.0), c_max_n - 10.0)
        c_surf_p = fx.min(fx.max(self.c_s_p.boundary("right", domain=self.r_p), 10.0), c_max_p - 10.0)
        
        ce_safe_n = fx.max(self.c_e_n, 1.0)
        ce_safe_p = fx.max(self.c_e_p, 1.0)
        
        def arcsinh_ast(x):
            return fx.log(x + (x**2 + 1.0)**0.5)

        j0_n = m_n * (ce_safe_n * c_surf_n * (c_max_n - c_surf_n))**0.5
        j0_p = m_p * (ce_safe_p * c_surf_p * (c_max_p - c_surf_p))**0.5
        
        term_n = i_den / (a_n * L_n * j0_n)
        term_p = i_den / (a_p * L_p * j0_p)
        
        eta_r_n = - (2.0 * R_const * self.T_cell / F) * (fx.integral(arcsinh_ast(term_n), over=self.x_n) / L_n)
        eta_r_p = - (2.0 * R_const * self.T_cell / F) * (fx.integral(arcsinh_ast(term_p), over=self.x_p) / L_p)
        eta_r = eta_r_n + eta_r_p
        
        eta_e = (1.0 - t_plus) * (2.0 * R_const * self.T_cell / F) * (
            fx.integral(fx.log(ce_safe_p), over=self.x_p) / L_p - 
            fx.integral(fx.log(ce_safe_n), over=self.x_n) / L_n
        )
        
        R_s_ohm = (L_n / sig_n + L_p / sig_p) / 3.0
        term_n_e = L_n / (eps_n ** b_brug)
        term_s_e = 3.0 * L_s / (eps_s ** b_brug)
        term_p_e = L_p / (eps_p ** b_brug)
        R_e_ohm = (term_n_e + term_s_e + term_p_e) / (3.0 * sig_e_ref)
        
        Q_s = (i_den ** 2) * R_s_ohm / L_cell
        Q_e = (i_den ** 2) * R_e_ohm / L_cell - (i_den * eta_e / L_cell)
        
        # CRITICAL TEST: Does fx.abs() annihilate the double?
        Q_irr = i_den * fx.abs(eta_r) / L_cell 
        Q_tot = Q_s + Q_e + Q_irr
        
        return {
            "equations": {
                self.Q_s_out: self.Q_s_out == Q_s,
                self.Q_e_out: self.Q_e_out == Q_e,
                self.Q_irr_out: self.Q_irr_out == Q_irr,
                self.Q_tot_out: self.Q_tot_out == Q_tot,
                self.eta_r_out: self.eta_r_out == eta_r,
                
                # Dummy equations to satisfy the solver rank validation
                self.c_e_n: fx.dt(self.c_e_n) == 0.0,
                self.c_e_p: fx.dt(self.c_e_p) == 0.0,
                self.c_s_n: fx.dt(self.c_s_n) == 0.0,
                self.c_s_p: fx.dt(self.c_s_p) == 0.0,
                self.T_cell: fx.dt(self.T_cell) == 0.0
            },
            "boundaries": {},
            "initial_conditions": {
                self.c_e_n: 1000.0, self.c_e_p: 1000.0,     
                self.c_s_n: 29866.0, self.c_s_p: 17038.0,
                self.T_cell: 298.15,
                self.Q_s_out: 0.0, self.Q_e_out: 0.0,
                self.Q_irr_out: 0.0, self.Q_tot_out: 0.0, self.eta_r_out: 0.0
            }
        }

@REQUIRES_COMPILER
def test_thermal_source_magnitudes():
    """
    PROBE: Mathematically forces the AST to reveal its internal calculations.
    If Q_irr drops below 50,000 W/m3 while eta_r remains negative, fx.abs() 
    is silently casting the overpotential to an integer 0 in the C++ layer.
    """
    engine = Engine(model=ThermalSourceProbe(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _, _ = engine._extract_metadata()
    # Evaluate instantaneous residual (Res = ydot - rhs = 0.0 - rhs -> rhs = -Res)
    res = engine.evaluate_residual(y0, ydot0, parameters={})
    
    Q_s = -res[engine.layout.state_offsets["Q_s_out"][0]]
    Q_e = -res[engine.layout.state_offsets["Q_e_out"][0]]
    Q_irr = -res[engine.layout.state_offsets["Q_irr_out"][0]]
    Q_tot = -res[engine.layout.state_offsets["Q_tot_out"][0]]
    eta_r = -res[engine.layout.state_offsets["eta_r_out"][0]]
    
    print(f"\n--- Extracted AST Constants ---")
    print(f"eta_r: {eta_r:.3f} V")
    print(f"Q_s:   {Q_s:.1f} W/m3")
    print(f"Q_e:   {Q_e:.1f} W/m3")
    print(f"Q_irr: {Q_irr:.1f} W/m3")
    print(f"Q_tot: {Q_tot:.1f} W/m3")
    
    # 1. Ensure overpotential evaluates correctly (~ -0.22V)
    assert eta_r < -0.1, f"Overpotential evaluated incorrectly: {eta_r}V"
    
    # 2. Ensure Q_irr is correctly utilizing the absolute value
    # Expected Q_irr = 97.6 A/m2 * 0.22V / 172.8e-6 m = ~124,000 W/m3
    assert Q_irr > 50000.0, f"AST Mismatch: Q_irr is abnormally low ({Q_irr}). fx.abs() compilation failed!"
    
    # 3. Ensure Total Heat matches the physical bounds for 48°C
    assert Q_tot > 100000.0, f"Total heat {Q_tot} is too low to drive the battery to 48°C."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])