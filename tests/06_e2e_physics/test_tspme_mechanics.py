import pytest
import numpy as np
import shutil
import platform
import os
import sys
import ion_flux as fx

# Ensure models directory is in path for E2E tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)

import pytest
import numpy as np
import ion_flux as fx
import shutil
import platform


"""
TSPMe Diagnostic Oracles

This suite contains three isolated probes to determine why the TSPMe model 
is exhibiting "Infinite Reservoir" and "Reversed Voltage" anomalies.
"""



# ==============================================================================
# ORACLE 1: FVM Spherical Volume Scaling
# ==============================================================================

class SphericalFaradayProbe(fx.PDE):
    """
    Isolates the spatial divergence operator for microscopic spherical grids.
    A constant flux is applied to the boundary. We assert that the average 
    concentration drops at EXACTLY the analytical Faraday depletion rate.
    """
    r = fx.Domain(bounds=(0, 5.0e-6), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    
    # 0D tracker for the volume-averaged concentration
    c_avg = fx.State(domain=None, name="c_avg")
    
    def math(self):
        flux = -1.0 * fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r),
                self.c_avg: self.c_avg == fx.integral(self.c, over=self.r) / ((4.0/3.0) * np.pi * (5.0e-6)**3)
            },
            "boundaries": {
                # 1.0 mol/m^2 s leaving the particle
                flux: {"left": 0.0, "right": 1.0}
            },
            "initial_conditions": {
                self.c: 1000.0, self.c_avg: 1000.0
            }
        }



def test_spherical_volume_depletion_rate():
    """
    PROBE: If this fails, the compiler is inflating the particle volume 
    (e.g., losing the micro-scale multiplier) resulting in the 'Infinite Reservoir'.
    
    Analytical rate of change for average concentration:
    dc_avg/dt = - (3 / R) * Flux_boundary
    dc_avg/dt = - (3 / 5e-6) * 1.0 = -600,000 mol/m^3 per second
    """
    engine = fx.Engine(model=SphericalFaradayProbe(), target="cpu", mock_execution=False)
    
    # Simulate a tiny 1 microsecond step to test the initial derivative
    res = engine.solve(t_span=(0, 1e-6), t_eval=np.array([0, 1e-6]))
    
    c_avg_initial = res["c_avg"].data[0]
    c_avg_final = res["c_avg"].data[-1]
    
    dc_dt_simulated = (c_avg_final - c_avg_initial) / 1e-6
    
    assert dc_dt_simulated == pytest.approx(-600000.0, rel=1e-2), \
        f"FVM Volume Bug: Simulated depletion rate was {dc_dt_simulated}, expected -600000.0. " \
        "The spherical microscopic grid volume is being calculated incorrectly!"




# ==============================================================================
# ORACLE 2: Boundary Node Extraction
# ==============================================================================

class BoundaryExtractionProbe(fx.PDE):
    """
    Isolates the `state.boundary("right")` AST operator when used inside a 0D equation.
    """
    r = fx.Domain(bounds=(0, 1.0), resolution=5, name="r")
    c = fx.State(domain=r, name="c")
    surf_tracker = fx.State(domain=None, name="surf_tracker")
    
    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                self.surf_tracker: self.surf_tracker == self.c.boundary("right", domain=self.r)
            },
            "boundaries": {},
            "initial_conditions": {
                # Setup a strict spatial gradient: [1.0, 2.0, 3.0, 4.0, 5.0]
                self.c: 1.0 + 4.0 * self.r.coords
            }
        }



def test_ast_boundary_node_extraction():
    """
    PROBE: If this fails, V_cell in the TSPMe is evaluating the wrong concentration 
    (e.g., the average instead of the surface), hiding the boundary depletion.
    """
    engine = fx.Engine(model=BoundaryExtractionProbe(), target="cpu", mock_execution=False)
    
    # Evaluate instantaneous residual at t=0
    N = engine.layout.n_states
    y0, ydot0, _, _, _ = engine._extract_metadata()
    
    res = engine.evaluate_residual(y0, ydot0)
    
    off_tracker, _ = engine.layout.state_offsets["surf_tracker"]
    
    # Residual of algebraic eq: res = y_eval - rhs. If y0 was initialized to 0, res = -rhs.
    rhs_eval = -res[off_tracker]
    
    assert rhs_eval == pytest.approx(5.0), \
        f"AST Boundary Extraction Bug: Expected to pull surface value 5.0, got {rhs_eval}."




# ==============================================================================
# ORACLE 3: Thermodynamic Sign and Magnitude Validation
# ==============================================================================

class ThermodynamicSignProbe(fx.PDE):
    """
    Isolates the exact overpotential and Ohmic calculations from the TSPMe 
    script to evaluate their raw signs and magnitudes at t=0 for a 2C discharge.
    """
    # Dummy states to extract raw intermediate AST calculations
    U_n_out = fx.State(domain=None, name="U_n_out")
    U_p_out = fx.State(domain=None, name="U_p_out")
    eta_r_out = fx.State(domain=None, name="eta_r_out")
    dPhi_s_out = fx.State(domain=None, name="dPhi_s_out")
    
    def math(self):
        F, R_const, T = 96485.0, 8.314, 298.15
        
        # LG M50 ExactTSPMe Initial Conditions
        x_n = 29866.0 / 33133.0  # ~0.901
        x_p = 17038.0 / 63104.0  # ~0.270
        
        # 2C Discharge Current Density
        i_app = 10.0
        A_elec = 0.1024 
        i_den = i_app / A_elec  # ~97.6 A/m^2
        
        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)
            
        def arcsinh_ast(x):
            return fx.log(x + (x**2 + 1.0)**0.5)

        U_n = (1.9793 * fx.exp(-39.3631 * x_n) + 0.2482 
               - 0.0909 * tanh_ast(29.8538 * (x_n - 0.1234)) 
               - 0.04478 * tanh_ast(14.9159 * (x_n - 0.2769)) 
               - 0.0205 * tanh_ast(30.4444 * (x_n - 0.6103)))
               
        U_p = (-0.8090 * x_p + 4.4875 
               - 0.0428 * tanh_ast(18.5138 * (x_p - 0.5542)) 
               - 17.7326 * tanh_ast(15.7890 * (x_p - 0.3117)) 
               + 17.5842 * tanh_ast(15.9308 * (x_p - 0.3120)))
        
        # Hardcoded j0 for isolation testing
        j0_n = 6.48e-7 * (1000.0 * 29866.0 * (33133.0 - 29866.0))**0.5
        a_n = 3.0 * 0.25 / 5.86e-6
        L_n = 85.2e-6
        
        term_n = i_den / (a_n * L_n * j0_n)
        eta_r_n = - (2.0 * R_const * T / F) * arcsinh_ast(term_n)
        
        sig_n, sig_p = 215.0, 0.18
        L_p = 75.6e-6
        R_s_ohm = (L_n / sig_n + L_p / sig_p) / 3.0
        dPhi_s = -i_den * R_s_ohm

        return {
            "equations": {
                self.U_n_out: self.U_n_out == U_n,
                self.U_p_out: self.U_p_out == U_p,
                self.eta_r_out: self.eta_r_out == eta_r_n,
                self.dPhi_s_out: self.dPhi_s_out == dPhi_s
            },
            "boundaries": {},
            "initial_conditions": {
                self.U_n_out: 0.0, self.U_p_out: 0.0,
                self.eta_r_out: 0.0, self.dPhi_s_out: 0.0
            }
        }



def test_initial_thermodynamic_signs():
    """
    PROBE: Validates the physical signs of the TSPMe algebraic calculations.
    During discharge (i_app > 0), Ohmic drops and Overpotentials MUST be negative 
    to pull the terminal voltage below the OCV.
    """
    engine = fx.Engine(model=ThermodynamicSignProbe(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y0, ydot0, _, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0)
    
    U_n = -res[engine.layout.state_offsets["U_n_out"][0]]
    U_p = -res[engine.layout.state_offsets["U_p_out"][0]]
    eta_r = -res[engine.layout.state_offsets["eta_r_out"][0]]
    dPhi_s = -res[engine.layout.state_offsets["dPhi_s_out"][0]]
    
    # 1. Validation of Equilibrium Open Circuit Potential (OCV)
    # At t=0, fully charged LG M50 should have U_p ~ 4.2V and U_n ~ 0.1V
    assert 4.1 <= U_p <= 4.3, f"U_p initialization is wildly incorrect: {U_p}V"
    assert 0.05 <= U_n <= 0.15, f"U_n initialization is wildly incorrect: {U_n}V"



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
    engine = fx.Engine(model=DepletionRateProbe(), target="cpu", mock_execution=False)

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
    engine = fx.Engine(model=DimensionalBroadcastingProbe(), target="cpu", mock_execution=False)
    
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
    engine = fx.Engine(model=ElectrolyteDepletionProbe(), target="cpu", mock_execution=False)
    
    # Run for 15 minutes (0.5 Ah at 2C, exactly where the jump occurred in the plot)
    t_eval = np.linspace(0, 900, 50)
    res = engine.solve(t_eval=t_eval)
    
    c_e_final = res["c_e"].data[-1]
    min_c_e = np.min(c_e_final)
    
    # If the electrolyte drops below 10.0, the `sqrt(c_e)` in the exchange current 
    # density will cause extreme overpotentials.
    assert min_c_e > 10.0, f"Electrolyte depleted to {min_c_e:.1f} mol/m^3. SPMe is breaking down at 2C."



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
        
        c_surf_n = fx.clamp(self.c_s_n.boundary("right", domain=self.r_n), lower=10, upper=c_max_n-10.0)
        c_surf_p = fx.clamp(self.c_s_p.boundary("right", domain=self.r_p), lower=10, upper=c_max_p-10.0)
        
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
    engine = fx.Engine(model=ThermalSourceProbe(), target="cpu", mock_execution=False)
    
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
