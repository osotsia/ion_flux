"""
End-to-End Industry Oracles: The DFN Crucible

This suite provides Method of Manufactured Solutions (MMS) and exact 
thermodynamic stability oracles specifically tailored for the highly coupled, 
non-linear phenomena observed in Doyle-Fuller-Newman (DFN) and TSPMe models.

It rigorously probes:
1. Piecewise Capacity Mass Conservation (Discontinuous Porosity).
2. Non-Linear Electrolyte Potential DAEs (Log-Gradient formulations).
3. Butler-Volmer Thermodynamic Symmetry and Root-Finding Drift.
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

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)

# ==============================================================================
# ORACLE 1: Piecewise Capacity Mass Conservation
# ==============================================================================

class PiecewisePorosityOracle(fx.PDE):
    """
    In battery models, porosity (eps) multiplies the time derivative: eps * dt(c) = div(flux).
    If a shared interface node is overwritten by the Piecewise region iterator instead 
    of averaging the volumetric capacity, mass will leak.
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=21) # dx = 0.1
    reg_A = cell.region(bounds=(0, 1.0), resolution=11, name="reg_A") # Indices 0-10
    reg_B = cell.region(bounds=(1.0, 2.0), resolution=11, name="reg_B") # Indices 10-20
    
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
    at overlapping FVM boundaries. A failure here explains capacity drift in full-cell models.
    """
    engine = Engine(model=PiecewisePorosityOracle(), target="cpu", mock_execution=False)
    
    # Run for 2.0 seconds. Total injected mass = 2.0.
    res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))
    c_final = res["c"].data[-1]
    
    dx = 0.1
    # Integrate mass = sum(eps_i * c_i * V_i)
    # Node 0: V = 0.5*dx. eps = 1.0.
    # Nodes 1-9: V = dx. eps = 1.0.
    # Node 10 (Interface): V = dx. Half belongs to A, Half to B. Effective eps = (1.0 + 5.0)/2 = 3.0.
    # Nodes 11-19: V = dx. eps = 5.0.
    # Node 20: V = 0.5*dx. eps = 5.0.
    
    mass_integrated = 0.0
    for i in range(21):
        if i == 0:
            mass_integrated += 1.0 * c_final[i] * (0.5 * dx)
        elif 1 <= i <= 9:
            mass_integrated += 1.0 * c_final[i] * dx
        elif i == 10:
            mass_integrated += 3.0 * c_final[i] * dx
        elif 11 <= i <= 19:
            mass_integrated += 5.0 * c_final[i] * dx
        elif i == 20:
            mass_integrated += 5.0 * c_final[i] * (0.5 * dx)
            
    # Allow 1e-4 tolerance for inherent implicit time-stepping errors
    np.testing.assert_allclose(mass_integrated, 2.0, rtol=1e-4, 
        err_msg="Mass leaked at the discontinuous capacity interface! "
                "The Piecewise AST logic is overwriting LHS capacities at shared nodes.")


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
    engine = Engine(model=ElectrolytePotentialDAEOracle(), target="cpu", mock_execution=False)
    
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
    engine = Engine(model=ButlerVolmerSymmetryOracle(), target="cpu", mock_execution=False)
    
    # Integrate for an extreme duration
    res = engine.solve(t_span=(0, 3600.0), t_eval=np.array([0.0, 3600.0]))
    
    c_s_initial = res["c_s"].data[0]
    c_s_final = res["c_s"].data[-1]
    
    drift = np.max(np.abs(c_s_final - c_s_initial))
    
    assert drift < 1e-12, f"Thermodynamic drift detected! System lost equilibrium (Drift: {drift:.3e})."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])