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
from ion_flux._core import solve_ida_native
from ion_flux.protocols import Sequence, CC, CV


"""
Analytical Oracles: Establishing Absolute Ground Truth

Before modifying the Rust native solver's initialization or step-clamping logic,
this suite proves that the engine currently achieves exact mathematical accuracy 
against known analytical solutions for ODEs, coupled ODEs, PDEs, and non-linear DAEs.
"""



# ==============================================================================
# Model 1: Simple ODE (Exponential Decay)
# dy/dt = -k * y  |  y(0) = 1.0
# Exact: y(t) = exp(-k * t)
# ==============================================================================
class ExponentialDecay(fx.PDE):
    y = fx.State(domain=None, name="y")
    k = fx.Parameter(default=0.5, name="k")

    def math(self):
        return {
            "equations": { self.y: fx.dt(self.y) == -self.k * self.y },
            "boundaries": {},
            "initial_conditions": { self.y: 1.0 }
        }



@REQUIRES_RUNTIME
def test_oracle_1_exponential_decay():
    model = ExponentialDecay()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    t_eval = np.linspace(0, 10, 100)
    k_val = 0.5
    res = engine.solve(t_eval=t_eval, parameters={"k": k_val})
    
    # Exact Analytical Solution
    y_exact = np.exp(-k_val * t_eval)
    
    # Compare with high strictness
    np.testing.assert_allclose(res["y"].data, y_exact, rtol=1e-4, atol=1e-5)




# ==============================================================================
# Model 2: Coupled ODEs (Harmonic Oscillator)
# dx/dt = v
# dv/dt = -omega^2 * x  | x(0)=1, v(0)=0
# Exact: x(t) = cos(omega*t), v(t) = -omega*sin(omega*t)
# ==============================================================================
class HarmonicOscillator(fx.PDE):
    x = fx.State(domain=None, name="x")
    v = fx.State(domain=None, name="v")
    omega = fx.Parameter(default=2.0, name="omega")

    def math(self):
        return {
            "equations": {
                self.x: fx.dt(self.x) == self.v,
                self.v: fx.dt(self.v) == -(self.omega**2) * self.x
            },
            "boundaries": {},
            "initial_conditions": { self.x: 1.0, self.v: 0.0 }
        }



@REQUIRES_RUNTIME
def test_oracle_2_harmonic_oscillator():
    model = HarmonicOscillator()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    t_eval = np.linspace(0, 5, 200)
    omega_val = 2.0
    res = engine.solve(t_eval=t_eval, parameters={"omega": omega_val})
    
    # Exact Analytical Solution
    x_exact = np.cos(omega_val * t_eval)
    v_exact = -omega_val * np.sin(omega_val * t_eval)
    
    np.testing.assert_allclose(res["x"].data, x_exact, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res["v"].data, v_exact, rtol=1e-3, atol=1e-3)




# ==============================================================================
# Model 3: 1D PDE (Heat Equation / Diffusion)
# dT/dt = alpha * d2T/dx2  | T(0)=0, T(L)=0, T(x,0) = sin(pi*x/L)
# Exact: T(x,t) = sin(pi*x/L) * exp(-alpha * (pi/L)^2 * t)
# ==============================================================================
class HeatEquationPDE(fx.PDE):
    # High resolution to minimize spatial discretization (Finite Volume) error
    x_dom = fx.Domain(bounds=(0, 1.0), resolution=100, name="x_dom")
    T = fx.State(domain=x_dom, name="T")
    alpha = fx.Parameter(default=0.1, name="alpha")

    def math(self):
        flux = -self.alpha * fx.grad(self.T)
        return {
            "equations": {
                self.T: fx.dt(self.T) == -fx.div(flux)
            },
            "boundaries": {
                # Dirichlet Boundaries
                self.T: {"left": fx.Dirichlet(0.0), "right": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {
                # Map initial condition to spatial coordinate geometry
                self.T: fx.sin(np.pi * self.x_dom.coords)
            }
        }



@REQUIRES_RUNTIME
def test_oracle_3_heat_equation_pde():
    model = HeatEquationPDE()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    t_eval = np.linspace(0, 1.0, 50)
    alpha_val = 0.1
    L = 1.0
    res = engine.solve(t_eval=t_eval, parameters={"alpha": alpha_val})
    
    x_coords = np.linspace(0, L, 100)
    
    # Check the solution at the final time step
    t_final = t_eval[-1]
    T_exact_final = np.sin(np.pi * x_coords / L) * np.exp(-alpha_val * (np.pi / L)**2 * t_final)
    T_sim_final = res["T"].data[-1]
    
    # 1D Spatial discretizations typically have O(dx^2) error, so we use a slightly looser tolerance
    np.testing.assert_allclose(T_sim_final, T_exact_final, rtol=1e-2, atol=1e-3)




# ==============================================================================
# Model 4: Non-Linear DAE
# dc/dt = -k * c        (ODE)
# V = ln(c) + I * R     (Algebraic Constraint)
# Exact: c(t) = exp(-k*t), V(t) = -k*t + I*R
# ==============================================================================
class NonLinearDAE(fx.PDE):
    c = fx.State(domain=None, name="c")
    V = fx.State(domain=None, name="V")
    
    k = fx.Parameter(default=0.5, name="k")
    I_app = fx.Parameter(default=2.0, name="I_app")
    R = fx.Parameter(default=0.1, name="R")

    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == -self.k * self.c,
                # Pure Algebraic DAE with a non-linear log operator
                self.V: self.V == fx.log(self.c) + self.I_app * self.R
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 1.0,
                self.V: 0.2  # ln(1.0) + 2.0 * 0.1
            }
        }



@REQUIRES_RUNTIME
def test_oracle_4_nonlinear_dae():
    model = NonLinearDAE()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    t_eval = np.linspace(0, 5, 100)
    k_val, I_val, R_val = 0.5, 2.0, 0.1
    res = engine.solve(t_eval=t_eval, parameters={"k": k_val, "I_app": I_val, "R": R_val})
    
    c_exact = np.exp(-k_val * t_eval)
    V_exact = -k_val * t_eval + (I_val * R_val)
    
    np.testing.assert_allclose(res["c"].data, c_exact, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res["V"].data, V_exact, rtol=1e-4, atol=1e-5)




# ==============================================================================
# Model 5: ALE Moving Boundary Mass Conservation (Stefan Problem)
# Domain stretches over time. Mass injected at boundary.
# Proves ALE upwinding and dynamic `dx` expansion perfectly conserves mass.
# ==============================================================================
class ALESwellingOracle(fx.PDE):
    x = fx.Domain(bounds=(0, 1.0), resolution=20, name="x")
    
    c = fx.State(domain=x, name="c")
    L = fx.State(domain=None, name="L")
    Mass_calc = fx.State(domain=None, name="Mass_calc")
    
    v_expand = fx.Parameter(default=0.5, name="v_expand")
    j_flux = fx.Parameter(default=1.0, name="j_flux")
    D = fx.Parameter(default=0.1, name="D")

    def math(self):
        flux = -self.D * fx.grad(self.c)
        return {
            "equations": {
                # Bulk diffusion
                self.c: fx.dt(self.c) == -fx.div(flux),
                # Domain expansion ODE
                self.L: fx.dt(self.L) == self.v_expand,
                # Algebraic tracker of total mass in the expanding domain
                self.Mass_calc: self.Mass_calc == fx.integral(self.c, over=self.x)
            },
            "boundaries": {
                # Bind the boundary natively to the moving state (Triggers ALE mode)
                self.x: {"right": self.L},
                # Negative flux injects mass INWARD from the right
                flux: {"left": 0.0, "right": -self.j_flux} 
            },
            "initial_conditions": {
                self.c: 1.0,
                self.L: 1.0,
                self.Mass_calc: 1.0
            }
        }



@REQUIRES_RUNTIME
def test_oracle_5_ale_mass_conservation():
    model = ALESwellingOracle()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    t_eval = np.linspace(0, 2.0, 50)
    
    # Simulate a domain expanding at 0.5 m/s, with an inward mass flux of 1.0 mol/s
    res = engine.solve(t_eval=t_eval, parameters={"v_expand": 0.5, "j_flux": 1.0})
    
    # Exact Analytical Mass Truth: 
    # Initial mass = 1.0 (L=1.0, uniform c=1.0)
    # Flux inward = 1.0 (mol/s)
    # Total mass over time = 1.0 + 1.0 * t
    mass_exact = 1.0 + 1.0 * t_eval
    
    # The algebraic tracker evaluates `fx.integral(c, over=x)` natively inside the solver
    mass_sim = res["Mass_calc"].data
    
    # A standard 5% temporal truncation error exists natively in time-integrated moving meshes.
    np.testing.assert_allclose(mass_sim, mass_exact, rtol=5e-2, atol=1e-3)




# ==============================================================================
# Model 6: ALE Spherical Contraction (Zero-Flux Geometric Concentration)
# Domain shrinks over time. No mass leaves the system.
# Proves spherical geometric dilution (-3 * c * v / R) correctly concentrates mass.
# ==============================================================================
class ALESphericalContractionOracle(fx.PDE):
    r = fx.Domain(bounds=(0, 1.0), resolution=20, coord_sys="spherical", name="r")
    
    c = fx.State(domain=r, name="c")
    R = fx.State(domain=None, name="R")
    
    v_shrink = fx.Parameter(default=-0.1, name="v_shrink")
    D = fx.Parameter(default=10.0, name="D") # Fast diffusion keeps concentration perfectly uniform

    def math(self):
        flux = -self.D * fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                # Bulk diffusion
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r),
                # Domain contraction ODE
                self.R: fx.dt(self.R) == self.v_shrink
            },
            "boundaries": {
                # Bind the boundary natively to the moving state (Triggers ALE mode)
                self.r: {"right": self.R},
                # Hermetically sealed particle (No mass leaves)
                flux: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c: 1.0,
                self.R: 1.0
            }
        }



@REQUIRES_RUNTIME
def test_oracle_6_ale_spherical_contraction():
    model = ALESphericalContractionOracle()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    # Simulate a particle shrinking from R=1.0 down to R=0.5
    t_eval = np.linspace(0, 5.0, 50) 
    
    res = engine.solve(t_eval=t_eval, parameters={"v_shrink": -0.1})
    
    # Exact Analytical Truth: 
    # Sphere volume V(t) = (4/3) * pi * R(t)^3
    # Total mass is strictly conserved (Zero Flux Boundary). M_0 = V_0 * c_0 = V(t) * c(t)
    # Therefore, concentration must spike cubically: c(t) = c_0 * (R_0 / R(t))^3
    
    R_t = 1.0 - 0.1 * t_eval
    c_exact = 1.0 * (1.0 / R_t)**3
    
    # Extract simulated concentration. Fast diffusion (D=10) ensures it remains 
    # uniform across the particle, so we can just take the mean of the spatial array.
    c_sim = np.mean(res["c"].data, axis=1)
    
    # If the ALE compiler failed to detect `coord_sys="spherical"`, it would scale 
    # linearly (resulting in 2.0x concentration) instead of cubically (8.0x concentration).
    # Allows 5% temporal truncation drift inherent to moving grids.
    np.testing.assert_allclose(c_sim, c_exact, rtol=5e-2, atol=1e-3)



# ------------------------------------------------------------------------------
# SECTION 1: TOPOLOGY & INDEXING ORACLES
# ------------------------------------------------------------------------------

class SphericalPolynomialMMSOracle(fx.PDE):
    """
    Manufactured Solution: c(r, t) = t + r^2
    - dt(c) = 1.0
    - grad(c) = 2*r
    - Spherical div(grad(c)) = (1/r^2) * d/dr(r^2 * 2r) = 6.0
    
    PDE to solve: dt(c) = div(grad(c)) - 5.0
    """
    r = fx.Domain(bounds=(0, 1.0), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    t_var = fx.State(domain=None, name="t_var") # Tracker for explicit time dependency

    def math(self):
        flux = fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.c: fx.dt(self.c) == fx.div(flux, axis=self.r) - 5.0
            },
            "boundaries": {
                # grad(r^2) = 2r. At r=0 -> 0.0. At r=1.0 -> 2.0.
                flux: {"left": 0.0, "right": 2.0}
            },
            "initial_conditions": {
                self.t_var: 0.0,
                self.c: self.r.coords**2
            }
        }



@REQUIRES_RUNTIME
def test_oracle_spherical_geometry_and_origin_limits():
    """Proves FVM scaling for spherical volumes is analytically exact and L'Hopital safety holds."""
    engine = fx.Engine(model=SphericalPolynomialMMSOracle(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    r_coords = np.linspace(0, 1.0, 10)
    c_exact = 1.0 + r_coords**2
    
    # Assert exactness (allowing for standard implicit time-integration tolerances)
    np.testing.assert_allclose(res["c"].data[-1], c_exact, rtol=1e-4, atol=1e-5)




class CoupledPiecewiseMMSOracle(fx.PDE):
    """
    Manufactured Solution: 
      c(x, t) = t + x^2   (Piecewise PDE)
      phi(x, t) = t - x^2 (Spatial DAE)
      
    DAE: 0 = div(grad(phi)) + 2.0 + {Coupling Term -> 0}
    PDE: dt(c) = div(grad(c)) - 1.0 + {Coupling Term -> 0}
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=20, name="cell")
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 2.0), resolution=10, name="reg_B")

    c = fx.State(domain=cell, name="c")
    phi = fx.State(domain=cell, name="phi")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        flux_c = fx.grad(self.c)
        flux_phi = fx.grad(self.phi)

        # Coupling terms that mathematically evaluate to 0 based on the exact solution
        coupling_to_c = self.phi - (self.t_var - self.cell.coords**2)
        coupling_to_phi = self.c - (self.t_var + self.cell.coords**2)

        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.c: fx.Piecewise({
                    self.reg_A: fx.dt(self.c) == fx.div(flux_c) + coupling_to_c - 1.0,
                    self.reg_B: fx.dt(self.c) == fx.div(flux_c) + coupling_to_c - 1.0
                }),
                self.phi: 0.0 == fx.div(flux_phi) + 2.0 + coupling_to_phi
            },
            "boundaries": {
                flux_c: {"left": 0.0, "right": 4.0},   # grad(x^2) = 2x -> 2(2) = 4
                flux_phi: {"left": 0.0, "right": -4.0} # grad(-x^2) = -2x -> -2(2) = -4
            },
            "initial_conditions": {
                self.t_var: 0.0,
                self.c: self.cell.coords**2,
                self.phi: -(self.cell.coords**2)
            }
        }



@REQUIRES_RUNTIME
def test_oracle_piecewise_stitching_and_dae_coupling():
    """Proves interface flux continuity across piecewise sub-regions and Jacobian DAE masks."""
    engine = fx.Engine(model=CoupledPiecewiseMMSOracle(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    x_coords = np.linspace(0, 2.0, 20)
    c_exact = 1.0 + x_coords**2
    phi_exact = 1.0 - x_coords**2
    
    np.testing.assert_allclose(res["c"].data[-1], c_exact, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res["phi"].data[-1], phi_exact, rtol=1e-4, atol=1e-5)




class MacroMicroMMSOracle(fx.PDE):
    """
    Manufactured Solution:
      c_micro(x, r, t) = t
      phi_macro(x, t) = -x^2
      
    This proves that the composite 2D domain safely unrolls and couples its 
    micro-surface boundary explicitly back into the 1D macroscopic DAE.
    """
    x_dom = fx.Domain(bounds=(0, 1.0), resolution=5, name="x_dom")
    r_dom = fx.Domain(bounds=(0, 1.0), resolution=5, coord_sys="spherical", name="r_dom")
    
    macro_micro = x_dom * r_dom

    c = fx.State(domain=macro_micro, name="c")
    phi = fx.State(domain=x_dom, name="phi")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        flux_c = fx.grad(self.c, axis=self.r_dom)
        flux_phi = fx.grad(self.phi, axis=self.x_dom)

        # Micro surface boundary mapping into macro DAE
        # Since c = t, c.boundary("right") = t.
        # Term mathematically evaluates to 0.0
        coupling_from_micro = self.c.boundary("right", domain=self.r_dom) - self.t_var

        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.c: fx.dt(self.c) == fx.div(flux_c, axis=self.r_dom) + 1.0,
                self.phi: 0.0 == fx.div(flux_phi, axis=self.x_dom) + 2.0 + coupling_from_micro
            },
            "boundaries": {
                flux_c: {"left": 0.0, "right": 0.0},
                # Ground the DAE with a Dirichlet anchor at x=0 to prevent a singular Jacobian!
                self.phi: {"left": fx.Dirichlet(0.0)},
                flux_phi: {"right": -2.0} # grad(-x^2) at x=1.0 is -2.0
            },
            "initial_conditions": {
                self.t_var: 0.0,
                self.c: 0.0,
                self.phi: -(self.x_dom.coords**2)
            }
        }



@REQUIRES_RUNTIME
def test_oracle_macro_micro_domain_unrolling():
    """Proves hierarchical composite topologies safely evaluate boundaries across dimensions."""
    engine = fx.Engine(model=MacroMicroMMSOracle(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    np.testing.assert_allclose(res["c"].data[-1].reshape((5, 5)), np.ones((5, 5)), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res["phi"].data[-1], -(np.linspace(0, 1.0, 5)**2), rtol=1e-4, atol=1e-5)




# ------------------------------------------------------------------------------
# SECTION 2: ALGORITHMIC EXACTNESS ORACLES
# ------------------------------------------------------------------------------

class NonLinearTransportMMSOracle(fx.PDE):
    """
    The AD Chain-Rule Crucible.
    Validates Enzyme Automatic Differentiation (AD) correctly applies the 
    chain-rule through a state-dependent non-linear transport parameter: D(c) = c.
    
    Manufactured Solution: c(x, t) = t + x
    - Flux N = -D(c) * grad(c) = -c * 1 = -(t + x)
    - div(N) = d/dx[-(t + x)] = -1.0
    - dt(c) = 1.0
    
    PDE to solve: dt(c) = -div(N)  =>  1.0 = -(-1.0)
    (No forcing function required! It is a naturally perfect manufactured solution).
    """
    x = fx.Domain(bounds=(1.0, 2.0), resolution=11, name="x")
    c = fx.State(domain=x, name="c")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        # D(c) = c. The flux tensor requires AD to differentiate BOTH terms in the product!
        flux = -self.c * fx.grad(self.c)
        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.c: fx.dt(self.c) == -fx.div(flux)
            },
            "boundaries": {
                # At x=1.0, N = -(t + 1.0). At x=2.0, N = -(t + 2.0).
                flux: {"left": -(self.t_var + 1.0), "right": -(self.t_var + 2.0)}
            },
            "initial_conditions": {
                self.t_var: 0.0,
                self.c: self.x.coords
            }
        }



@REQUIRES_RUNTIME
def test_oracle_nonlinear_state_dependent_ad_chain_rule():
    """Proves Enzyme AD correctly formulates Jacobians for state-dependent parameters (D(c)*grad(c))."""
    engine = fx.Engine(model=NonLinearTransportMMSOracle(), target="cpu", mock_execution=False)
    res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))
    
    x_coords = np.linspace(1.0, 2.0, 11)
    c_exact_t2 = 2.0 + x_coords
    
    np.testing.assert_allclose(res["c"].data[-1], c_exact_t2, rtol=1e-4, atol=1e-5)




class StatefulMultiplexerMMSOracle(fx.PDE):
    """
    The CCCV Multiplexer.
    Proves the implicit BDF solver's event locator mathematically lands on 
    trigger asymptotes and hot-swaps algebraic constraints seamlessly.
    
    Model:
    dt(SOC) = I
    V = SOC + I * R  (where R=1.0)
    
    Protocol: 
    CC at 1.0A until V=5.0V, then CV at 5.0V for 1 second.
    """
    soc = fx.State(domain=None, name="soc")
    V = fx.State(domain=None, name="V")
    i_app = fx.State(domain=None, name="i_app")
    
    terminal = fx.Terminal(current=i_app, voltage=V)

    def math(self):
        return {
            "equations": {
                self.soc: fx.dt(self.soc) == self.i_app,
                self.V: self.V == self.soc + self.i_app * 1.0
            },
            "boundaries": {},
            "initial_conditions": { self.soc: 0.0, self.V: 0.0, self.i_app: 0.0 }
        }



@REQUIRES_RUNTIME
def test_oracle_cccv_state_machine_asymptote_timing():
    """Proves the dense root-finder hits discrete trigger asymptotes exactly and re-inverts the Jacobian."""
    model = StatefulMultiplexerMMSOracle()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    protocol = Sequence([
        CC(rate=1.0, until=model.V >= 5.0), # Phase 1
        CV(voltage=5.0, time=1.0)           # Phase 2
    ])
    
    res = engine.solve(protocol=protocol)
    t_history = res["Time [s]"].data
    v_history = res["V"].data
    i_history = res["i_app"].data
    
    # Phase 1 Analytical Math (CC):
    # I = 1.0 -> dt(SOC) = 1.0 -> SOC(t) = t
    # V(t) = SOC + I*R = t + 1.0. 
    # Target V=5.0 is reached EXACTLY at t = 4.0s.
    
    # Robustly isolate the transition step boundary by searching for the voltage asymptote
    transition_idx = np.argmax(v_history >= 4.99)
    t_transition = t_history[transition_idx]
    
    assert t_transition == pytest.approx(4.0, abs=1e-2), "Event locator missed the exact CC to CV trigger time!"
    
    # Phase 2 Analytical Math (CV):
    # V = 5.0 -> 5.0 = SOC + I -> I = 5.0 - SOC
    # dt(SOC) = 5.0 - SOC. Solved with SOC(4.0) = 4.0:
    # SOC(t) = 5.0 - exp(-(t - 4.0))
    # I(t) = exp(-(t - 4.0))
    # After exactly 1 second in CV (t=5.0), I(5.0) = exp(-1)
    i_final = i_history[-1]
    expected_i_final = np.exp(-1.0)
    
    assert t_history[-1] == pytest.approx(5.0, abs=1e-2)
    assert i_final == pytest.approx(expected_i_final, rel=1e-2)




# Unstructured Tetrahedron with 4 nodes
tetrahedron_mesh = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "elements": [[0, 1, 2, 3]]
}



class UnstructuredGraphConservationOracle(fx.PDE):
    """
    The Topological Graph Oracle.
    Proves that the C++ translation of unstructured meshes (via Compressed Sparse Row CSR formats) 
    perfectly conserves mass during Matrix-Free FVM evaluations.
    """
    mesh = fx.Domain.from_mesh(tetrahedron_mesh, name="mesh", surfaces={"top": [2, 3]})
    c = fx.State(domain=mesh, name="c")
    mass_tracker = fx.State(domain=None, name="mass_tracker")

    def math(self):
        # We explicitly extract `flux` to a shared Python variable. This ensures the AST 
        # tags the exact same node reference with the Boundary ID that is passed into `div`.
        flux = -fx.grad(self.c)
        
        return {
            "equations": {
                # Stable Laplacian Diffusion: dt(c) = -div(-grad(c)) = div(grad(c))
                self.c: fx.dt(self.c) == -fx.div(flux),
                # fx.integral seamlessly sums (Volume * Concentration) for all nodes in the mesh
                self.mass_tracker: self.mass_tracker == fx.integral(self.c, over=self.mesh)
            },
            "boundaries": {
                # Because unstructured FVM uses (bulk_div + bc_val), a NEGATIVE bc_val 
                # evaluated against dt(c) = -div(flux) translates to a POSITIVE inward mass injection.
                flux: {"top": -10.0} 
            },
            "initial_conditions": {
                self.c: 1.0, self.mass_tracker: 0.0
            }
        }



@REQUIRES_RUNTIME
def test_oracle_unstructured_csr_mass_conservation():
    """Proves the FVM element volumes and CSR integration weights are perfectly symmetric and conservative."""
    engine = fx.Engine(model=UnstructuredGraphConservationOracle(), target="cpu", mock_execution=False)
    
    # We solve over 2 seconds. Total injected mass = 10.0 * 2 nodes * 2 seconds = 40.0.
    res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))
    
    # The actual geometric volume of the tetrahedron is (1/3) * Base * Height = 1/6.
    # Initial mass = 1.0 (concentration) * (1/6) (volume) = 1/6.
    # Expected final mass = 1/6 + 40.0 = 40.166666...
    exact_final_mass = (1.0 / 6.0) + 40.0
    
    # The mass_tracker integrates the mesh volumes natively via the C++ graph
    simulated_final_mass = res["mass_tracker"].data[-1]
    
    np.testing.assert_allclose(simulated_final_mass, exact_final_mass, rtol=1e-4, atol=1e-5)
