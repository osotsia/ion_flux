"""
Analytical Oracles: Establishing Absolute Ground Truth

Before modifying the Rust native solver's initialization or step-clamping logic,
this suite proves that the engine currently achieves exact mathematical accuracy 
against known analytical solutions for ODEs, coupled ODEs, PDEs, and non-linear DAEs.
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
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
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
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
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
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
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
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
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
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
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
    
    # If the ALE upwinding drifts, or the diffusion gradients fail to stretch with `dx`, 
    # mass conservation will immediately fail.
    np.testing.assert_allclose(mass_sim, mass_exact, rtol=1e-3, atol=1e-3)