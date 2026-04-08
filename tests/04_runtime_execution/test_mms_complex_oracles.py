"""
Method of Manufactured Solutions (MMS) Oracles

This suite establishes absolute analytical ground truth for the most complex
topological and coupled features of the engine. By manufacturing polynomial 
solutions of degree 2 or less (which Finite Volume Methods evaluate with zero 
spatial truncation error), we achieve rigorous, machine-precision oracles for:

--- SECTION 1: TOPOLOGY & INDEXING ---
1. Spherical coordinate L'Hopital limits and geometric scaling.
2. Multi-region Piecewise spatial stitching with Spatial DAE coupling.
3. Multi-scale (Macro x Micro) hierarchical unrolling and boundary extraction.

--- SECTION 2: ALGORITHMIC EXACTNESS ---
4. The AD Chain-Rule Crucible: Exact Jacobians for non-linear, state-dependent transport.
5. The CCCV Multiplexer: Exact analytical timing of discrete root-finding asymptotes.
6. The Topological Graph Oracle: Exact mass conservation on unstructured Matrix-Free meshes.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine
from ion_flux.protocols import Sequence, CC, CV

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
    engine = Engine(model=SphericalPolynomialMMSOracle(), target="cpu", mock_execution=False)
    
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
    cell = fx.Domain(bounds=(0, 2.0), resolution=21, name="cell")
    reg_A = cell.region(bounds=(0, 1.0), resolution=11, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 2.0), resolution=11, name="reg_B")

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
    engine = Engine(model=CoupledPiecewiseMMSOracle(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    x_coords = np.linspace(0, 2.0, 21)
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
    engine = Engine(model=MacroMicroMMSOracle(), target="cpu", mock_execution=False)
    
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
                # Explicitly add 1.0 to map `coords` (0.0->1.0) to the physical bounds (1.0->2.0).
                # Prevents a singular D(c) = 0 condition at the left boundary.
                self.c: self.x.coords + 1.0 
            }
        }

@REQUIRES_RUNTIME
def test_oracle_nonlinear_state_dependent_ad_chain_rule():
    """Proves Enzyme AD correctly formulates Jacobians for state-dependent parameters (D(c)*grad(c))."""
    engine = Engine(model=NonLinearTransportMMSOracle(), target="cpu", mock_execution=False)
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
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
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
    engine = Engine(model=UnstructuredGraphConservationOracle(), target="cpu", mock_execution=False)
    
    # We solve over 2 seconds. Total injected mass = 10.0 * 2 nodes * 2 seconds = 40.0.
    res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))
    
    # The actual geometric volume of the tetrahedron is (1/3) * Base * Height = 1/6.
    # Initial mass = 1.0 (concentration) * (1/6) (volume) = 1/6.
    # Expected final mass = 1/6 + 40.0 = 40.166666...
    exact_final_mass = (1.0 / 6.0) + 40.0
    
    # The mass_tracker integrates the mesh volumes natively via the C++ graph
    simulated_final_mass = res["mass_tracker"].data[-1]
    
    np.testing.assert_allclose(simulated_final_mass, exact_final_mass, rtol=1e-4, atol=1e-5)

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])