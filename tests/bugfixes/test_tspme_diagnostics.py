"""
TSPMe Diagnostic Oracles

This suite contains three isolated probes to determine why the TSPMe model 
is exhibiting "Infinite Reservoir" and "Reversed Voltage" anomalies.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine

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
    engine = Engine(model=SphericalFaradayProbe(), target="cpu", mock_execution=False)
    
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
    engine = Engine(model=BoundaryExtractionProbe(), target="cpu", mock_execution=False)
    
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
    engine = Engine(model=ThermodynamicSignProbe(), target="cpu", mock_execution=False)
    
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