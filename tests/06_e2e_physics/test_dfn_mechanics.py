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
import shutil
import platform
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, CV, Rest, CurrentProfile
from ion_flux._core import solve_ida_native


"""
E2E Integration: Industry Models & Battery Library API

End-to-End validation of the implicit solver against highly coupled PDE-ODE-DAE
industry architectures (e.g., lumped thermals).
Cross-validates the custom `native` backend against the `sundials` backend.
Validates the public Battery Library (`DFN`), flat-string parameter overrides,
and dynamic drive cycle evaluation.
"""



# ==============================================================================
# Industry Models
# ==============================================================================

class ThermallyCoupledSPM(fx.PDE):
    """
    Industry-grade Single Particle Model coupled with a lumped thermal ODE.
    Validates PDE (Diffusion), ODE (Temperature), and DAE (Voltage/Current) interaction.
    """
    r = fx.Domain(bounds=(0, 5e-6), resolution=15, coord_sys="spherical")
    
    c_s = fx.State(domain=r, name="c_s")
    T_cell = fx.State(domain=None, name="T_cell") # 0D Lumped ODE
    V_cell = fx.State(domain=None, name="V_cell") # 0D Algebraic DAE
    i_app = fx.State(domain=None, name="i_app")   # Cycler controlled State
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    h_conv = fx.Parameter(default=10.0)
    T_amb = fx.Parameter(default=298.15)
    
    def math(self):
        # Temperature-dependent diffusion (Mild Arrhenius to prevent instant collapse at 30A)
        D_s = 1e-14 * fx.exp(0.01 * (self.T_cell - 298.15))
        flux = -D_s * fx.grad(self.c_s, axis=self.r)
        
        # Faraday's law at particle surface
        j_surf = self.i_app / 96485.0
        
        # Lumped thermal generation (Simplified Joule heating + Reaction heat)
        Q_gen = fx.abs(self.i_app) * (4.2 - self.V_cell) * 0.1
        Q_cool = self.h_conv * (self.T_cell - self.T_amb)
        
        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(flux, axis=self.r),
                self.T_cell: fx.dt(self.T_cell) == (Q_gen - Q_cool) / 1000.0,
                # Pure Algebraic Voltage Map
                self.V_cell: self.V_cell == 4.2 - 1e-5 * self.c_s.right - 0.001 * self.i_app
            },
            "boundaries": {
                # A positive j_surf (discharge) flows IN to the particle, increasing concentration 
                # and correctly lowering the cell voltage.
                flux: {"left": 0.0, "right": -j_surf} 
            },
            "initial_conditions": {
                self.c_s: 10000.0,
                self.T_cell: 298.15,
                self.V_cell: 4.1,
                self.i_app: 0.0
            }
        }



# ==============================================================================
# Cross-Validation & Execution Tests
# ==============================================================================

@REQUIRES_RUNTIME
def test_industry_physics_cccv_cross_validation():
    """
    Proves the native implicit solver perfectly matches SUNDIALS IDAS tracking 
    highly coupled PDE-ODE-DAE systems through multi-stage CCCV protocols.
    """
    model = ThermallyCoupledSPM()
    
    engine_native = fx.Engine(model=model, target="cpu", solver_backend="native")
    engine_sundials = fx.Engine(model=model, target="cpu", solver_backend="sundials")
    
    # Aggressive CCCV sequence
    protocol = Sequence([
        CC(rate=30.0, until=model.V_cell <= 3.0),
        CV(voltage=3.0, until=model.i_app <= 1.0),
        Rest(time=600)
    ])
    
    res_native = engine_native.solve(protocol=protocol)
    res_sundials = engine_sundials.solve(protocol=protocol)
    
    assert res_native.status == "completed"
    assert res_sundials.status == "completed"
    
    # Physics Validation:
    # 1. Temperature must have increased due to 30A discharge Joule heating
    assert res_native["T_cell"].data[-1] > 298.15
    # 2. Voltage must have perfectly landed and held at the 3.0V asymptote during CV
    V_n = res_native["V_cell"].data
    assert np.min(V_n) >= 3.0 - 1e-3
    
    # Cross-Validate Native vs. SUNDIALS IDAS Output Trajectories
    # Ensures step bounds, algebraic jumps, and tolerances behave identically
    t_n = res_native["Time [s]"].data
    t_s = res_sundials["Time [s]"].data
    
    V_s_interp = np.interp(t_n, t_s, res_sundials["V_cell"].data)
    T_s_interp = np.interp(t_n, t_s, res_sundials["T_cell"].data)
    
    # Continuous states can be safely interpolated
    np.testing.assert_allclose(res_native["V_cell"].data, V_s_interp, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_native["T_cell"].data, T_s_interp, rtol=1e-3, atol=1e-3)
    
    # Discontinuous cycler state (i_app) should be checked at the end to prevent alignment artifacting
    np.testing.assert_allclose(res_native["i_app"].data[-1], res_sundials["i_app"].data[-1], rtol=1e-3, atol=1e-3)




@REQUIRES_RUNTIME
def test_extreme_algebraic_discontinuity_t0():
    """
    Validates that both the Native and SUNDIALS solvers successfully traverse a 
    massive t=0 algebraic discontinuity (e.g., i_app jumping from 0 to 100A instantly)
    without suffering from IDACalcIC initialization failures or divergent initial steps.
    """
    model = ThermallyCoupledSPM()
    engine_native = fx.Engine(model=model, target="cpu", solver_backend="native")
    engine_sundials = fx.Engine(model=model, target="cpu", solver_backend="sundials")
    
    # 10 second instantaneous 100A pulse tests the solver's algebraic snap at t=0
    protocol = Sequence([ CC(rate=100.0, time=10.0) ])
    
    res_native = engine_native.solve(protocol=protocol)
    res_sundials = engine_sundials.solve(protocol=protocol)
    
    assert res_native.status == "completed"
    assert res_sundials.status == "completed"
    
    i_app_native = res_native["i_app"].data
    assert i_app_native[0] == pytest.approx(100.0, rel=1e-3), "Native solver failed t=0 algebraic snap."
    
    i_app_sundials = res_sundials["i_app"].data
    assert i_app_sundials[0] == pytest.approx(100.0, rel=1e-3), "SUNDIALS failed t=0 algebraic snap (IDACalcIC failure)."




# ==============================================================================
# Public Battery Library & API Tests
# ==============================================================================
@pytest.mark.skip(reason="Might revisit")
def test_library_drive_cycle_array_injection(dfn_engine):
    """
    Validates dynamic time array evaluation (`t_eval`).
    Tests how users inject real-world telemetry arrays seamlessly.
    """
    base_params = parameters.Chen2020()
    
    # Mock a high-frequency 100-second drive cycle sampling array
    time_array = np.linspace(0, 100, 1000)
    current_array = np.sin(time_array)
    protocol = CurrentProfile(time=time_array, current=current_array)
    
    result = dfn_engine.solve(protocol=protocol, parameters=base_params)
    
    assert result.status == "completed"
    # Verify the Engine successfully evaluated the continuous array dimensions without clipping
    assert len(result["Time [s]"].data) == 1000



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
    engine = fx.Engine(model=FaradayBugOracle(), target="cpu", mock_execution=False)
    
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



# ==============================================================================
# ORACLE 1: Piecewise Capacity Mass Conservation
# ==============================================================================

class PiecewisePorosityOracle(fx.PDE):
    """
    In battery models, porosity (eps) multiplies the time derivative: eps * dt(c) = div(flux).
    If a shared interface node is overwritten by the Piecewise region iterator instead 
    of averaging the volumetric capacity, mass will leak.
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=20) 
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A") # Indices 0-9
    reg_B = cell.region(bounds=(1.0, 2.0), resolution=10, name="reg_B") # Indices 10-19
    
    c = fx.State(domain=cell, name="c")
    
    def math(self):
        flux = fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_A: 1.0 * fx.dt(self.c) == fx.div(flux),
                    self.reg_B: 5.0 * fx.dt(self.c) == fx.div(flux)
                })
            },
            "boundaries": {
                # Inject 1.0 unit of mass per second
                flux: {"left": -1.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c: 0.0
            }
        }



@REQUIRES_RUNTIME
def test_oracle_piecewise_porosity_conservation():
    """
    PROBE: Ensures the AST compiler correctly handles discontinuous LHS multipliers 
    at FVM faces. A failure here explains capacity drift in full-cell models.
    """
    engine = fx.Engine(model=PiecewisePorosityOracle(), target="cpu", mock_execution=False)
    
    # Run for 2.0 seconds. Total injected mass = 2.0.
    res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))
    c_final = res["c"].data[-1]
    
    dx = 2.0 / 19.0
    
    mass_integrated = 0.0
    for i in range(20):
        if i == 0:
            mass_integrated += 1.0 * c_final[i] * (0.5 * dx)
        elif 1 <= i <= 8:
            mass_integrated += 1.0 * c_final[i] * dx
        elif i == 9:
            # Node 9 is fully in reg_A. Its right face is the boundary.
            mass_integrated += 1.0 * c_final[i] * dx
        elif i == 10:
            # Node 10 is fully in reg_B. Its left face is the boundary.
            mass_integrated += 5.0 * c_final[i] * dx
        elif 11 <= i <= 18:
            mass_integrated += 5.0 * c_final[i] * dx
        elif i == 19:
            mass_integrated += 5.0 * c_final[i] * (0.5 * dx)
            
    # Allow 1e-4 tolerance for inherent implicit time-stepping errors
    np.testing.assert_allclose(mass_integrated, 2.0, rtol=1e-4, 
        err_msg="Mass leaked at the discontinuous capacity interface! "
                "The Piecewise AST logic is overwriting LHS capacities at shared faces.")




# ==============================================================================
# ORACLE 2: Electrolyte Potential Non-Linear DAE
# ==============================================================================

class ElectrolytePotentialDAEOracle(fx.PDE):
    """
    MMS for the electrolyte DAE: div( k * grad(phi) + k * B * grad(c)/c ) = 0.
    
    Manufactured steady-state solution:
    c(x) = exp(x) -> grad(c)/c = 1.0
    phi(x) = -1.5 * x -> grad(phi) = -1.5
    Let k = 2.0, B = 3.0.
    Flux = 2.0(-1.5) + 2.0 * 3.0 * (1.0) / 2.0  (Wait, k*B*grad(c)/c = 2*3*1 = 6.0)
    Let's set Flux = 2.0 * grad(phi) + 3.0 * (grad(c)/c).
    If phi(x) = -1.5 * x, grad(phi) = -1.5. 2(-1.5) = -3.0.
    Flux = -3.0 + 3.0(1) = 0.0. div(Flux) = 0.0.
    """
    x = fx.Domain(bounds=(0, 2.0), resolution=50, name="x")
    c = fx.State(domain=x, name="c")
    phi = fx.State(domain=x, name="phi")
    
    def math(self):
        # O'Regan paper style log-gradient formulation
        ce_diff_term = fx.grad(self.c) / fx.max(self.c, 1e-3)
        flux_phi = 2.0 * fx.grad(self.phi) + 3.0 * ce_diff_term
        
        return {
            "equations": {
                # Lock concentration to the exact manufactured profile
                self.c: fx.dt(self.c) == 0.0,
                # Pure Spatial DAE
                self.phi: fx.div(flux_phi) == 0.0
            },
            "boundaries": {
                self.phi: {"left": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {
                self.c: fx.exp(self.x.coords),
                self.phi: 0.0 # Force the solver to instantly find the root
            }
        }



@REQUIRES_RUNTIME
def test_oracle_electrolyte_dae_log_gradient_coupling():
    """
    PROBE: Validates that the implicit algebraic root-finder and the discrete 
    FVM formulations of `grad(c)/c` perfectly align to solve the non-linear potential field.
    """
    engine = fx.Engine(model=ElectrolytePotentialDAEOracle(), target="cpu", mock_execution=False)
    
    # 1. Take a single minimal time step to force the DAE root evaluation
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    x_coords = np.linspace(0, 2.0, 50)
    phi_exact = -1.5 * x_coords
    phi_sim = res["phi"].data[-1]
    
    # FVM `grad(c)/c` exhibits discrete truncation error distinct from analytical `1.0`.
    # O(dx^2) error is expected. We assert the trajectory profile matches closely.
    np.testing.assert_allclose(phi_sim, phi_exact, atol=2e-2, 
        err_msg="Electrolyte potential DAE root-finder deviated from exact thermodynamic gradient coupling.")




# ==============================================================================
# ORACLE 3: Butler-Volmer Thermodynamic Symmetry
# ==============================================================================

class ButlerVolmerSymmetryOracle(fx.PDE):
    """
    Tests perfect equilibrium stability. 
    If eta = phi_s - phi_e - U(c_s) == 0.0, the Butler-Volmer equation must 
    evaluate to exactly 0.0, and no drift should occur over thousands of seconds.
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    c_s = fx.State(domain=x, name="c_s")
    phi_e = fx.State(domain=x, name="phi_e")
    phi_s = fx.State(domain=x, name="phi_s")
    
    def math(self):
        # Arbitrary OCP function
        U_ocp = 0.5 + 0.1 * self.c_s
        
        eta = self.phi_s - self.phi_e - U_ocp
        
        # Exchange current density
        i_0 = fx.abs(self.c_s) ** 0.5
        
        def sinh_ast(x_val):
            e2x = fx.exp(2.0 * x_val)
            return (e2x - 1.0) / (e2x + 1.0) # tanh used to bound explosive limits, scaling matches symmetry
            
        j_faraday = i_0 * sinh_ast(eta * 38.9) # F / RT approx 38.9
        
        return {
            "equations": {
                # If j_faraday is exactly 0, states remain absolutely static
                self.c_s: fx.dt(self.c_s) == -j_faraday,
                # Ground the DAEs to prevent singular matrices at equilibrium
                self.phi_e: self.phi_e == 0.0,
                self.phi_s: self.phi_s == 0.55
            },
            "boundaries": {},
            "initial_conditions": {
                self.c_s: 0.5,   # U_ocp = 0.5 + 0.1(0.5) = 0.55
                self.phi_e: 0.0,
                self.phi_s: 0.55 # eta = 0.55 - 0.0 - 0.55 = 0.0
            }
        }



@REQUIRES_RUNTIME
def test_oracle_butler_volmer_equilibrium_drift():
    """
    PROBE: Ensures the solver's non-linear root finder doesn't introduce 
    floating point "creep" during stiff DAE integration at thermodynamic equilibrium.
    """
    engine = fx.Engine(model=ButlerVolmerSymmetryOracle(), target="cpu", mock_execution=False)
    
    # Integrate for an extreme duration
    res = engine.solve(t_span=(0, 3600.0), t_eval=np.array([0.0, 3600.0]))
    
    c_s_initial = res["c_s"].data[0]
    c_s_final = res["c_s"].data[-1]
    
    drift = np.max(np.abs(c_s_final - c_s_initial))
    
    assert drift < 1e-12, f"Thermodynamic drift detected! System lost equilibrium (Drift: {drift:.3e})."




# ==============================================================================
# ORACLE 4: The Integrated Hierarchical DFN Crucible
# ==============================================================================

class HierarchicalDFNCrucibleOracle(fx.PDE):
    """
    Manufactured Solution testing the fully integrated DFN architecture.
    
    Let x_dom = [0, 2]. Define X = x_dom.coords + 1.0 (ranges 1.0 to 3.0).
    Manufactured Truth:
      c_s(X, r, t) = t + X^2 * r^2
      phi_e(X, t) = t * X
      c_e(X, t) = t + X
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=20, name="cell")
    x_n = cell.region(bounds=(0, 1.0), resolution=10, name="x_n")
    x_p = cell.region(bounds=(1.0, 2.0), resolution=10, name="x_p")
    
    r = fx.Domain(bounds=(0, 1.0), resolution=5, coord_sys="spherical", name="r")
    
    c_s = fx.State(domain=cell * r, name="c_s")
    phi_e = fx.State(domain=cell, name="phi_e")
    c_e = fx.State(domain=cell, name="c_e")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        # Spatially varying field to stress test the IndexManager's context routing
        X = self.cell.coords + 1.0
        
        flux_cs = fx.grad(self.c_s, axis=self.r)
        flux_phi = fx.grad(self.phi_e)
        flux_ce = fx.grad(self.c_e)
        
        c_surf = self.c_s.boundary("right", domain=self.r)
        
        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                
                # dt(c_s) = 1.0. div_r(grad_r) = 6 * X^2.
                self.c_s: fx.dt(self.c_s) == fx.div(flux_cs, axis=self.r) + 1.0 - 6.0 * (X**2),
                
                # 0 = div_x(grad_x) + phi_e / X - (c_surf - X^2)
                # Evaluates analytically to: 0 = 0 + (t*X)/X - (t + X^2 - X^2) = t - t = 0.
                self.phi_e: 0.0 == fx.div(flux_phi) + self.phi_e / X - (c_surf - X**2),
                
                # Piecewise testing concurrent with spatial DAEs
                self.c_e: fx.Piecewise({
                    self.x_n: 2.0 * fx.dt(self.c_e) == fx.div(flux_ce) + 2.0,
                    self.x_p: 3.0 * fx.dt(self.c_e) == fx.div(flux_ce) + 3.0
                })
            },
            "boundaries": {
                # grad_r(t + X^2 r^2) at r=1.0 is 2*X^2
                flux_cs: {"left": 0.0, "right": 2.0 * (X**2)},
                
                # phi_e = t*X. At x=0 (X=1.0), phi_e = t.
                self.phi_e: {"left": fx.Dirichlet(self.t_var)},
                
                # grad_x(t*X) = t.
                flux_phi: {"right": self.t_var},
                
                # grad_x(t + X) = 1.0
                flux_ce: {"left": 1.0, "right": 1.0}
            },
            "initial_conditions": {
                self.t_var: 0.0,
                self.c_s: (X**2) * (self.r.coords**2),
                self.phi_e: 0.0,
                self.c_e: X
            }
        }



@REQUIRES_RUNTIME
def test_oracle_integrated_hierarchical_dfn():
    """
    PROBE: Proves the compiler correctly merges piecewise logic, hierarchical boundaries, 
    spatial DAEs, and spherical geometric dilution into a stable native execution graph.
    """
    engine = fx.Engine(model=HierarchicalDFNCrucibleOracle(), target="cpu", mock_execution=False)
    
    # Assert analytical exactness over a 1.0 second integration window
    # We pass "dummy" bc the engine only populates the res.trajectory metadata dictionary 
    # (which contains the raw, flattened C-arrays like _y_raw) when an adjoint pass is anticipated
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]), requires_grad=["dummy"])
    
    # Compute exact analytical fields at t=1.0
    dx_cell = 2.0 / 19.0
    X_coords = np.linspace(0, 2.0, 20) + 1.0
    
    dx_r = 1.0 / 4.0
    r_coords = np.linspace(0, 1.0, 5)
    
    phi_e_exact = 1.0 * X_coords
    c_e_exact = 1.0 + X_coords
    
    # Broadcast X^2 and r^2 to formulate the flattened 2D expected array
    c_s_exact = 1.0 + (X_coords[:, None]**2) * (r_coords[None, :]**2)
    c_s_exact = c_s_exact.flatten()
    
    np.testing.assert_allclose(
        res["phi_e"].data[-1], phi_e_exact, rtol=1e-3, atol=1e-5,
        err_msg="Spatial DAE failed to correctly extract and couple to the hierarchical micro-surface boundary."
    )
    
    np.testing.assert_allclose(
        res["c_e"].data[-1], c_e_exact, rtol=1e-3, atol=1e-5,
        err_msg="Piecewise equation diverged under the presence of a spatial DAE in the same macroscopic domain."
    )
    
    np.testing.assert_allclose(
        res["c_s"].data[-1], c_s_exact, rtol=1e-3, atol=1e-5,
        err_msg="Spherical hierarchy failed to integrate non-linear coordinates injected from the outer domain."
    )

    # Validate that Enzyme perfectly constructed the non-linear Dense Jacobian
    y_final = res.trajectory["_y_raw"][-1]
    ydot_final = np.zeros_like(y_final)
    J = np.array(engine.evaluate_jacobian(y_final.tolist(), ydot_final.tolist(), c_j=1.0, parameters={}))
    
    assert np.isfinite(J).all(), "Enzyme LLVM AD generated NaNs evaluating the coupled Jacobian."
    assert np.linalg.matrix_rank(J) == engine.layout.n_states, "Jacobian contains structurally singular rows."

    
