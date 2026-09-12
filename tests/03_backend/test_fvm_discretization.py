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


"""
Middle-End Codegen: Piecewise Auto-Stitching & Conservation

This suite acts as a Test-Driven Development (TDD) oracle for the DSL's 
ability to automatically detect and fix disjointed fluxes in piecewise spatial 
domains. It proves that explicitly separated fluxes passed to `fx.div()` 
within a `fx.Piecewise` block are automatically stitched into a conservative 
field at the boundaries.
"""




# ==============================================================================
# Models for Testing
# ==============================================================================

class BiMaterialDiffusion(fx.PDE):
    """
    2-Region model. The user defines `flux_left` and `flux_right` separately.
    The DSL must auto-stitch them at x=1.0 to prevent mass accumulation.
    """
    bulk = fx.Domain(bounds=(0, 2.0), resolution=20)
    reg_L = bulk.region(bounds=(0, 1.0), resolution=10, name="reg_L")
    reg_R = bulk.region(bounds=(1.0, 2.0), resolution=10, name="reg_R")
    
    c = fx.State(domain=bulk, name="c")
    
    D_L = fx.Parameter(default=1.0)
    D_R = fx.Parameter(default=5.0)

    def math(self):
        # Disjointed flux definitions
        flux_left = -self.D_L * fx.grad(self.c)
        flux_right = -self.D_R * fx.grad(self.c)
        
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_left),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_right)
                })
            },
            "boundaries": {
                # Sealing the outer edges to strictly test internal interface conservation
                flux_left: {"left": 0.0},
                flux_right: {"right": 0.0}
            },
            "initial_conditions": {
                self.c: 1.0
            }
        }




class TriRegionElectrolyte(fx.PDE):
    """
    3-Region model mirroring the exact bug structure from the O'Regan paper.
    If not auto-stitched, the interfaces at x=1 and x=2 will leak.
    """
    cell = fx.Domain(bounds=(0, 3.0), resolution=30)
    x_n = cell.region(bounds=(0, 1.0), resolution=10, name="x_n")
    x_s = cell.region(bounds=(1.0, 2.0), resolution=10, name="x_s")
    x_p = cell.region(bounds=(2.0, 3.0), resolution=10, name="x_p")
    
    c_e = fx.State(domain=cell, name="c_e")
    
    def math(self):
        # Three entirely disjointed flux branches
        flux_n = -1.0 * fx.grad(self.c_e)
        flux_s = -0.5 * fx.grad(self.c_e)
        flux_p = -2.0 * fx.grad(self.c_e)
        
        return {
            "equations": {
                self.c_e: fx.Piecewise({
                    self.x_n: fx.dt(self.c_e) == -fx.div(flux_n),
                    self.x_s: fx.dt(self.c_e) == -fx.div(flux_s),
                    self.x_p: fx.dt(self.c_e) == -fx.div(flux_p)
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




class CoupledTensorFlux(fx.PDE):
    """
    Ensures that multi-state, complex flux tensors (e.g. Diffusion + Migration)
    are successfully auto-stitched without destroying the AST evaluation logic.
    """
    bulk = fx.Domain(bounds=(0, 2.0), resolution=20)
    reg_L = bulk.region(bounds=(0, 1.0), resolution=10, name="reg_L")
    reg_R = bulk.region(bounds=(1.0, 2.0), resolution=10, name="reg_R")
    
    c = fx.State(domain=bulk, name="c")
    phi = fx.State(domain=bulk, name="phi")
    
    def math(self):
        # Complex tensor dependent on both 'c' and 'phi'
        flux_L = -1.0 * fx.grad(self.c) + self.c * fx.grad(self.phi)
        flux_R = -5.0 * fx.grad(self.c) + self.c * fx.grad(self.phi)
        
        # Pure DAE for phi to force coupling
        i_e = -fx.grad(self.phi)
        
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_L),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_R)
                }),
                self.phi: fx.div(i_e) == 0.0
            },
            "boundaries": {
                flux_L: {"left": 0.0},
                flux_R: {"right": 0.0},
                i_e: {"left": 0.0, "right": 0.0},
                self.phi: {"left": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {
                self.c: 1.0,
                self.phi: 0.0
            }
        }



def _get_volumes(resolution: int, bounds: tuple) -> np.ndarray:
    dx = (bounds[1] - bounds[0]) / max(resolution - 1, 1)
    v = np.ones(resolution) * dx
    v[0] = 0.5 * dx
    v[-1] = 0.5 * dx
    return v



def _get_exact_volumes(resolutions_list: list, dx_list: list) -> np.ndarray:
    v_all = []
    for res, dx in zip(resolutions_list, dx_list):
        v = np.ones(res) * dx
        v[0] = 0.5 * dx
        v[-1] = 0.5 * dx
        v_all.append(v)
    return np.concatenate(v_all)



@REQUIRES_COMPILER
def test_bimaterial_mass_conservation():
    """
    Proves that a 2-region piecewise evaluation conserves mass globally.
    If the interface at x=1.0 is not stitched, the divergence residuals
    will accumulate/leak mass due to differing diffusion coefficients.
    """
    engine = fx.Engine(model=BiMaterialDiffusion(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    
    # Create an arbitrary concentration gradient
    np.random.seed(42)
    y = np.random.uniform(1.0, 5.0, size=N).tolist()
    ydot = np.zeros(N).tolist()
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    dx_bulk = 2.0 / 19.0
    V_cells = _get_exact_volumes([20], [dx_bulk])
    
    total_mass_drift = np.sum(np.array(res) * V_cells)
    assert np.isclose(total_mass_drift, 0.0, atol=1e-10), f"Leaked mass! Drift: {total_mass_drift}"



@REQUIRES_COMPILER
def test_bimaterial_jacobian_interface_coupling():
    """
    Proves that the Analytical Jacobian establishes cross-boundary coupling.
    If fluxes are disjoint, Node 9 (left of interface) will have a 0.0 derivative
    with respect to Node 10 (right of interface).
    """
    engine = fx.Engine(model=BiMaterialDiffusion(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    y = np.linspace(1.0, 5.0, N).tolist()
    ydot = np.zeros(N).tolist()
    
    J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    
    # Interface nodes for a 20-resolution parent divided 10/10
    left_node = 9
    right_node = 10
    
    # Derivative of Left Node's residual with respect to Right Node's state
    # Must be non-zero to prove the gradient stencil crossed the boundary.
    coupling_L_to_R = J[left_node, right_node]
    coupling_R_to_L = J[right_node, left_node]
    
    assert abs(coupling_L_to_R) > 1e-8, "Jacobian is disjoint! Left region does not depend on Right region."
    assert abs(coupling_R_to_L) > 1e-8, "Jacobian is disjoint! Right region does not depend on Left region."




@REQUIRES_COMPILER
def test_triregion_mass_conservation():
    """
    Proves conservation scales safely to 3+ regions (like the DFN electrolyte).
    """
    engine = fx.Engine(model=TriRegionElectrolyte(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y = np.linspace(100.0, 200.0, N).tolist() # Smooth gradient across all 3 regions
    ydot = np.zeros(N).tolist()
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    vol_off = engine.layout.mesh_offsets["cell"]["w_V_nodes"]
    m_list = engine.layout.get_mesh_data()
    V_cells = np.array(m_list[vol_off : vol_off + N]) * 3.0
    
    total_mass_drift = np.sum(np.array(res) * V_cells)
    assert np.isclose(total_mass_drift, 0.0, atol=1e-10), f"Leaked mass! Drift: {total_mass_drift}"



@REQUIRES_COMPILER
def test_complex_tensor_piecewise_stitching():
    """
    Ensures that auto-stitching doesn't break when fluxes contain complex
    multi-state dependencies (like migration `c * grad(phi)`).
    """
    engine = fx.Engine(model=CoupledTensorFlux(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    off_c, size_c = engine.layout.state_offsets["c"]
    off_phi, size_phi = engine.layout.state_offsets["phi"]
    
    y = np.zeros(N)
    # Establish gradients for both 'c' and 'phi'
    y[off_c:off_c+size_c] = np.linspace(1.0, 5.0, size_c)
    y[off_phi:off_phi+size_phi] = np.linspace(0.0, 10.0, size_phi)
    
    res = engine.evaluate_residual(y.tolist(), np.zeros(N).tolist(), parameters={})
    res_c = np.array(res[off_c:off_c+size_c])
    
    dx_bulk = 2.0 / 19.0
    V_cells = _get_exact_volumes([20], [dx_bulk])
    
    total_mass_drift = np.sum(res_c * V_cells)
    
    assert np.isclose(total_mass_drift, 0.0, atol=1e-10), f"Leaked mass! Drift: {total_mass_drift}"



class SphericalFVMOracle(fx.PDE):
    """
    Manufactured Analytical Solution for Spherical Diffusion.
    If c(r, t) = r^2 + 6*t
    Then:
    dt(c) = 6.0
    grad(c) = 2*r
    
    The divergence in spherical coordinates is:
    div(grad(c)) = (1/r^2) * d/dr( r^2 * 2r ) = (1/r^2) * d/dr( 2r^3 ) = 6.0
    
    Therefore, dt(c) == div(grad(c)) is an EXACT mathematical truth.
    If the compiler mistakenly treats the sphere as a 1D slab (Cartesian), 
    div(grad(c)) would equal 2.0, and the residual would fail massively.
    """
    r = fx.Domain(bounds=(0, 1.0), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    
    # 0D target to test the boundary extraction AST logic
    surface_val = fx.State(domain=None, name="surface_val")

    def math(self):
        flux = -fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                # dt(c) - div(grad(c)) = 0 -> dt(c) - (-div(flux)) = 0
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r),
                
                # Check if boundary extraction grabs the exact edge node correctly
                self.surface_val: self.surface_val == self.c.boundary("right", domain=self.r)
            },
            "boundaries": {
                # grad(r^2) at r=1.0 is 2.0. So flux = -2.0.
                flux: {"left": 0.0, "right": -2.0}
            },
            "initial_conditions": {
                # Initialize c(r, 0) = r^2
                self.c: self.r.coords ** 2,
                self.surface_val: 0.0
            }
        }



def test_spherical_fvm_and_boundary_extraction_exactness():
    engine = fx.Engine(model=SphericalFVMOracle(), target="cpu", mock_execution=False)
    
    # We want to check the instantaneous residual at t=0.
    y0, ydot0, _, _, _ = engine._extract_metadata()
    y0 = np.array(y0)
    ydot0 = np.zeros_like(y0)
    
    # Because c(r,t) = r^2 + 6t, the true derivative dt(c) MUST be 6.0 everywhere.
    off_c, size_c = engine.layout.state_offsets["c"]
    ydot0[off_c : off_c + size_c] = 6.0
    
    res = engine.evaluate_residual(y0.tolist(), ydot0.tolist(), parameters={})
    
    # 1. Check Spherical FVM Divergence
    # Residual F = ydot - rhs. If the compiler is perfectly exact, F == 0.0
    c_residuals = res[off_c : off_c + size_c]
    np.testing.assert_allclose(
        c_residuals, 0.0, atol=1e-12, 
        err_msg="COMPILER BUG: Spherical FVM divergence failed! The compiler is calculating the wrong cell volumes/areas."
    )
    
    # 2. Check Boundary Node Extraction
    # The true analytical value of c at the right boundary (r=1.0) is 1.0^2 = 1.0.
    off_surf, _ = engine.layout.state_offsets["surface_val"]
    
    # Residual of the algebraic observer eq: F = 0.0 - surface_val_extracted. 
    # Therefore, the extracted value is -F.
    extracted_surface = -res[off_surf]
    
    assert extracted_surface == pytest.approx(1.0, rel=1e-12), \
        "COMPILER BUG: Boundary extraction `c.boundary('right')` grabbed the wrong node or failed to evaluate!"



# ==============================================================================
# ORACLE 1: Integral Context Propagation (The Structural Singularity Bug)
# ==============================================================================

class IntegralJacobianSingularityOracle(fx.PDE):
    """
    Exposes the bug where `fx.integral()` failed to pass `self.current_axis` 
    to its child AST nodes. This caused nested `fx.grad()` operators to lose 
    their spatial dimension and fall back to evaluating `0.0`.
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=20, name="cell")
    reg = cell.region(bounds=(0, 2.0), resolution=20, name="reg")
    
    phi = fx.State(domain=cell, name="phi")
    T_var = fx.State(domain=None, name="T_var")
    
    def math(self):
        return {
            "equations": {
                # Lock phi to the exact manufactured profile
                self.phi: fx.dt(self.phi) == 0.0,
                
                # dt(T) = integral(grad(phi)) - T
                # If context is lost, grad(phi) -> 0.0, and T decays to 0.0.
                self.T_var: fx.dt(self.T_var) == fx.integral(fx.grad(self.phi), over=self.reg) - self.T_var
            },
            "boundaries": {
                self.phi: {"left": fx.Dirichlet(0.0), "right": fx.Dirichlet(2.0)}
            },
            "initial_conditions": {
                self.phi: (self.cell.coords ** 2) / 2.0, # Manufactured: phi = x^2 / 2
                self.T_var: 0.0
            }
        }



@REQUIRES_RUNTIME
def test_oracle_integral_jacobian_singularity():
    """
    PROBE: Manufactured Truth: 
    phi = x^2 / 2 -> grad(phi) = x
    integral(grad(phi), over=cell) = integral(x) from 0 to 2 = 2.0.
    Thus, dt(T) = 2.0 - T. T should asymptote exactly to 2.0.
    """
    engine = fx.Engine(model=IntegralJacobianSingularityOracle(), target="cpu", mock_execution=False)
    
    # 1. Assert the Jacobian is strictly full-rank and fully coupled
    y0, ydot0, _, _, _ = engine._extract_metadata()
    J = np.array(engine.evaluate_jacobian(y0, ydot0, c_j=1.0, parameters={}))
    
    # Ensure no rows/cols are sheared to 0.0 identically (except fixed boundaries)
    assert np.linalg.matrix_rank(J) >= engine.layout.n_states - 2, \
        "Structural Singularity Detected! Enzyme failed to emit cross-coupled sensitivities " \
        "because `fx.integral()` wiped the spatial context of `fx.grad()`."

    # 2. Assert Dynamic Trajectory Truth
    res = engine.solve(t_span=(0, 15.0), t_eval=np.array([0.0, 15.0]))
    
    T_final = res["T_var"].data[-1]
    
    # Relaxed rtol to 5% to account for the discrete FVM truncation error on a coarse N=20 mesh. 
    # The true discrete value is exactly 37/19 (~1.947). The bug previously evaluated to exactly 0.0.
    np.testing.assert_allclose(T_final, 2.0, rtol=0.05,
        err_msg=f"Integral Context Loss Detected! Expected T to reach ~2.0, but got {T_final:.3f}. "
                "The nested spatial gradient evaluated to a static 0.0.")




# ==============================================================================
# ORACLE 2: AST Fragmentation & Geometric `KeyError` Aliasing
# ==============================================================================

class SubregionGeometricScalingOracle(fx.PDE):
    """
    Exposes the deepcopy AST fragmentation where sub-regions lose their parent 
    links, causing missing volume geometry arrays and raising a `KeyError`.
    """
    cell = fx.Domain(bounds=(0, 3.0), resolution=30, name="cell")
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 3.0), resolution=20, name="reg_B")
    
    c = fx.State(domain=cell, name="c")
    mass_A = fx.State(domain=None, name="mass_A")
    mass_B = fx.State(domain=None, name="mass_B")
    
    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                # Pure algebraic integrals
                self.mass_A: self.mass_A == fx.integral(self.c, over=self.reg_A),
                self.mass_B: self.mass_B == fx.integral(self.c, over=self.reg_B)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: self.cell.coords, # Manufactured: c(x) = x
                self.mass_A: 0.0,
                self.mass_B: 0.0
            }
        }



@REQUIRES_RUNTIME
def test_oracle_subregion_geometric_scaling():
    """
    PROBE: Manufactured Truth:
    mass_A = integral(x) from 0 to 1 = 0.5
    mass_B = integral(x) from 1 to 3 = (3^2 / 2) - (1^2 / 2) = 4.5 - 0.5 = 4.0
    
    If the TopologyAnalyzer fails to link subregions, `Engine()` instantiation 
    will crash violently with a KeyError during code-generation.
    """
    # The instantiation itself is part of the test (verifies the missing KeyError fix)
    engine = fx.Engine(model=SubregionGeometricScalingOracle(), target="cpu", mock_execution=False)
    
    # Take a single short step to trigger the algebraic evaluation
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    mass_A_sim = res["mass_A"].data[-1]
    mass_B_sim = res["mass_B"].data[-1]
    
    # Relaxed rtol to 1% to account for FVM midpoint-rule quadrature truncation error on a coarse mesh.
    # The primary success metric is that the Engine instantiated without a KeyError crash.
    np.testing.assert_allclose(mass_A_sim, 0.5, rtol=0.01, err_msg="Volume Scaling mapped to incorrect L_phys for reg_A.")
    np.testing.assert_allclose(mass_B_sim, 4.0, rtol=0.01, err_msg="Volume Scaling mapped to incorrect L_phys for reg_B.")




# ==============================================================================
# ORACLE 3: Piecewise Harmonic Mean Conservation
# ==============================================================================

class HarmonicMeanDiscontinuityOracle(fx.PDE):
    """
    Exposes implicit instability and flux non-conservation at sharp material 
    interfaces (Piecewise domains). 
    """
    cell = fx.Domain(bounds=(0, 2.0), resolution=40, name="cell")
    reg_L = cell.region(bounds=(0, 1.0), resolution=20, name="reg_L")
    reg_R = cell.region(bounds=(1.0, 2.0), resolution=20, name="reg_R")
    
    c = fx.State(domain=cell, name="c")
    
    def math(self):
        # Extreme jump in material property (e.g., solid vs electrolyte conductivity)
        flux_L = -10.0 * fx.grad(self.c)
        flux_R = -0.1 * fx.grad(self.c)
        
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_L),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_R)
                })
            },
            "boundaries": {
                # Force a steady-state profile across the cell
                self.c: {"left": fx.Dirichlet(100.0), "right": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {
                self.c: 50.0
            }
        }



@REQUIRES_RUNTIME
def test_oracle_harmonic_mean_discontinuity():
    """
    PROBE: In steady-state, flux must be perfectly continuous at x=1.0.
    -10.0 * grad_L = -0.1 * grad_R
    Analytical Truth at Interface (c_int):
    10.0 * (100 - c_int) / 1.0 = 0.1 * (c_int - 0) / 1.0
    1000 - 10 c_int = 0.1 c_int  ->  10.1 c_int = 1000  -> c_int = 99.0099
    """
    engine = fx.Engine(model=HarmonicMeanDiscontinuityOracle(), target="cpu", mock_execution=False)
    
    # Integrate to extreme steady state
    res = engine.solve(t_span=(0, 500.0), t_eval=np.array([0.0, 500.0]))
    c_final = res["c"].data[-1]
    
    # Interface nodes (indices 19 and 20 for a 40-node mesh sliced down the middle)
    c_int_L = c_final[19]
    c_int_R = c_final[20]
    
    # Because of the 100x discrepancy in diffusivity, the arithmetic mean is physically invalid.
    # We reconstruct the exact interface value by weighting nodes by their diffusivities
    # to explicitly invert the flux conservation: D_L * (c_L - c_int) = D_R * (c_int - c_R)
    D_L, D_R = 10.0, 0.1
    c_int_exact = (D_L * c_int_L + D_R * c_int_R) / (D_L + D_R)
    
    np.testing.assert_allclose(
        c_int_exact, 99.0099, rtol=1e-2,
        err_msg=f"Harmonic Mean Interface Failure! Flux is leaking across the Piecewise boundary. "
                f"Expected interface concentration ~99.01, got {c_int_exact:.2f}."
    )



# ==============================================================================
# ORACLE 1: Memory Corruption / Misrouting in Cross-Domain Boundaries
# ==============================================================================

class HierarchicalMemoryCorruptionOracle(fx.PDE):
    """
    Isolates the out-of-bounds memory fetch when a micro-particle boundary 
    depends on a global macroscopic state array.
    """
    cell = fx.Domain(bounds=(0, 10), resolution=10, name="cell")
    x_n = cell.region(bounds=(0, 5), resolution=5, name="x_n")
    x_p = cell.region(bounds=(5, 10), resolution=5, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 1), resolution=100, coord_sys="spherical", name="r_n")
    
    # 2D Field (Size: 5 * 100 = 500 nodes)
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n") 
    # 1D Global Field (Size: 10 nodes)
    phi_e = fx.State(domain=cell, name="phi_e")      
    
    def math(self):
        flux = -fx.grad(self.c_s_n, axis=self.r_n)
        
        return {
            "equations": {
                self.phi_e: fx.dt(self.phi_e) == 0.0,
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(flux, axis=self.r_n)
            },
            "boundaries": {
                # Evaluated over x_n * r_n, but referencing cell.
                flux: {"left": 0.0, "right": self.phi_e}
            },
            "initial_conditions": {
                self.c_s_n: 0.0,
                self.phi_e: 0.0
            }
        }



@REQUIRES_COMPILER
def test_cross_domain_memory_corruption():
    """
    PROBE: Proves that `phi_e` is incorrectly indexed using the flat 2D `c_s_n` 
    index. Because of the C++ CLAMP macro, the index does not segfault but maps 
    to the entirely wrong physical location.
    """
    engine = fx.Engine(model=HierarchicalMemoryCorruptionOracle(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y0, ydot0, _, _, _ = engine._extract_metadata()
    y0 = np.array(y0)
    
    off_phi, size_phi = engine.layout.state_offsets["phi_e"]
    off_c, size_c = engine.layout.state_offsets["c_s_n"]
    
    # Establish a known linear gradient across phi_e: [0, 1, 2, ..., 9]
    y0[off_phi : off_phi + size_phi] = np.arange(10.0)
    
    res = engine.evaluate_residual(y0.tolist(), ydot0, parameters={})
    c_residuals = np.array(res[off_c : off_c + size_c])
    
    # Analyze the residual at the rightmost boundary of the FIRST particle (x_n node 0).
    # Its flat index in the c_s_n array is 99.
    res_0 = c_residuals[99]
    
    # Mathematical Truth: The first particle sits at x_n[0], which corresponds to cell[0].
    # Therefore, phi_e should evaluate to 0.0, yielding a residual of exactly 0.0.
    expected_res_correct = 0.0
    
    assert res_0 == pytest.approx(expected_res_correct, abs=1e-5), \
        f"Memory Misrouting Confirmed: Expected the correct mapping to fetch phi_e[0] (Residual 0.0), " \
        f"but got a residual of {res_0:.1f}. The compiler is fetching out-of-bounds memory."




# ==============================================================================
# ORACLE 2: Flat-Line Composite Integration
# ==============================================================================

class CompositeIntegrationOracle(fx.PDE):
    """
    Isolates the integration scaling of multi-scale domains.
    Mathematically: ∫ 1.0 dV = V_macro * V_micro
    """
    x = fx.Domain(bounds=(0, 2.0), resolution=10, name="x")
    r = fx.Domain(bounds=(0, 1.0), resolution=10, coord_sys="spherical", name="r")
    
    c = fx.State(domain=x * r, name="c")
    total_mass = fx.State(domain=None, name="total_mass")

    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                self.total_mass: self.total_mass == fx.integral(self.c, over=self.x * self.r)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 1.0,
                self.total_mass: 0.0
            }
        }



@REQUIRES_COMPILER
def test_composite_domain_integration_failure():
    """
    PROBE: Proves the compiler lacks support for 2D composite integration. 
    It will currently crash with a KeyError during AST translation, and must be 
    patched to support nested nested loops yielding `V_macro * V_micro`.
    """
    engine = fx.Engine(model=CompositeIntegrationOracle(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0, parameters={})
    
    off_mass, _ = engine.layout.state_offsets["total_mass"]
    
    # The evaluated integral is `-res` of the algebraic equation
    simulated_integral = -res[off_mass]
    
    # Exact Analytical Volume:
    # V_x = 2.0
    # V_r = (4/3) * pi * (1.0)^3 = 4.18879
    # Total volume = 2.0 * 4.18879 = 8.37758
    exact_integral = 2.0 * (4.0/3.0) * np.pi * (1.0)**3
    
    assert simulated_integral == pytest.approx(exact_integral, rel=1e-2), \
        f"Composite Integration Bug Confirmed! Expected analytical mass {exact_integral:.3f}, " \
        f"but got {simulated_integral:.3f}."



@REQUIRES_COMPILER
def test_spherical_fvm_volume_exactness():
    """Ensures the compiler has no off-by-one errors regarding Spherical geometry arrays."""
    engine = fx.Engine(model=SphericalFVMOracle(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _, _ = engine._extract_metadata()
    y0 = np.array(y0)
    ydot0 = np.zeros_like(y0)
    
    off_c, size_c = engine.layout.state_offsets["c"]
    ydot0[off_c : off_c + size_c] = 6.0
    
    res = engine.evaluate_residual(y0.tolist(), ydot0.tolist(), parameters={})
    c_residuals = res[off_c : off_c + size_c]
    
    np.testing.assert_allclose(
        c_residuals, 0.0, atol=1e-12, 
        err_msg="COMPILER BUG: Spherical FVM divergence failed."
    )




# ==============================================================================
# ORACLE 4: Non-Linear Staggered Grid DAE Crucible
# ==============================================================================

class StaggeredNonLinearMMSOracle(fx.PDE):
    """
    Manufactures a highly non-linear spatial DAE designed to expose face-interpolation 
    errors in the FVM lowering pass.
    """
    x = fx.Domain(bounds=(1.0, 2.0), resolution=50, name="x")
    
    c = fx.State(domain=x, name="c")
    phi = fx.State(domain=x, name="phi")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        coords = self.x.coords
        t = self.t_var
        
        kappa_e = self.c
        kappa_D = self.c ** 2
        
        i_e = kappa_e * fx.grad(self.phi) + kappa_D * fx.grad(self.c) / self.c
        div_source = 4.0 * coords * (t ** 2) + (t ** 2) + 2.0 * t

        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.c: fx.dt(self.c) == coords, 
                self.phi: fx.div(i_e) == div_source
            },
            "boundaries": {
                self.c: {
                    "left": fx.Dirichlet(1.0 * t + 1.0), 
                    "right": fx.Dirichlet(2.0 * t + 1.0)
                },
                self.phi: {
                    "left": fx.Dirichlet(1.0 * t), 
                    "right": fx.Dirichlet(4.0 * t)
                }
            },
            "initial_conditions": {
                self.t_var: 1.0, 
                self.c: coords * 1.0 + 1.0,
                self.phi: (coords ** 2) * 1.0
            }
        }



@REQUIRES_COMPILER
def test_oracle_staggered_grid_nonlinear_dae_interpolation():
    """
    PROBE: Fails if the AST-to-C++ translator misapplies cell-center states to 
    cell-face flux evaluations during non-linear tensor assembly.
    """
    engine = fx.Engine(model=StaggeredNonLinearMMSOracle(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(1.0, 2.0), t_eval=np.array([1.0, 2.0]))
    
    x_coords = np.linspace(1.0, 2.0, 50)
    c_exact = x_coords * 2.0 + 1.0
    phi_exact = (x_coords ** 2) * 2.0
    
    np.testing.assert_allclose(res["c"].data[-1], c_exact, rtol=1e-3)
    np.testing.assert_allclose(res["phi"].data[-1], phi_exact, rtol=1e-3,
        err_msg="Non-Linear DAE Failed: Face interpolation for kappa_e or grad(c)/c is flawed.")




# ==============================================================================
# ORACLE 5: Hierarchical Inter-Domain Mass Mapping Crucible
# ==============================================================================

class HierarchicalMassCouplingOracle(fx.PDE):
    """
    A minimal topology tracking lithium flux extracted from a micro-spherical 
    particle and injected into a macro-Cartesian mesh. Evaluates if the FVM 
    geometric translation factor (a_s) is mathematically conserved in the native arrays.
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    r = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r")
    macro_micro = x * r
    
    c_s = fx.State(domain=macro_micro, name="c_s")
    c_e = fx.State(domain=x, name="c_e")

    def math(self):
        j_flux = 100.0 
        
        flux_s = -1e-14 * fx.grad(self.c_s, axis=self.r)
        flux_e = -1e-10 * fx.grad(self.c_e, axis=self.x)
        
        eps_s = 0.5
        R_p = 5e-6
        a_s = 3.0 * eps_s / R_p
        
        j_volumetric = j_flux * a_s
        
        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(flux_s, axis=self.r),
                self.c_e: fx.dt(self.c_e) == -fx.div(flux_e, axis=self.x) + j_volumetric
            },
            "boundaries": {
                flux_s: {"left": 0.0, "right": j_flux},
                flux_e: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c_s: 1000.0,
                self.c_e: 0.0
            }
        }



@REQUIRES_COMPILER
def test_oracle_hierarchical_mass_coupling_crucible():
    """
    PROBE: Directly calculates exact NumPy FVM volumes and dots them against the 
    simulated arrays to strictly account for every single mole of lithium transferred 
    across the scale gap.
    """
    engine = fx.Engine(model=HierarchicalMassCouplingOracle(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    c_e_final = res["c_e"].data[-1]
    c_s_initial = res["c_s"].data[0].reshape((10, 10))
    c_s_final = res["c_s"].data[-1].reshape((10, 10))
    
    # 1. Exact Macro Volumes (dx = 1.0 / 9)
    dx_macro = 1.0 / 9.0
    V_macro = np.ones(10) * dx_macro
    V_macro[0] *= 0.5
    V_macro[-1] *= 0.5
    
    # 2. Exact Micro Volumes (Spherical, dr = 5e-6 / 9)
    R_p = 5e-6
    dr_micro = R_p / 9.0
    r_faces = np.linspace(0, R_p, 10)
    
    V_micro = np.zeros(10)
    for i in range(10):
        r_right = r_faces[i] + 0.5*dr_micro if i < 9 else r_faces[i]
        r_left = r_faces[i] - 0.5*dr_micro if i > 0 else 0.0
        V_micro[i] = (4.0/3.0) * np.pi * (r_right**3 - r_left**3)
        
    V_particle_total = (4.0/3.0) * np.pi * R_p**3
    
    # 3. Mass Balance Accounting
    Li_added_e = np.sum(c_e_final * V_macro)
    
    eps_s = 0.5
    c_s_drop = c_s_initial - c_s_final
    avg_drop_per_particle = np.sum(c_s_drop * V_micro, axis=1) / V_particle_total
    Li_removed_s = np.sum(avg_drop_per_particle * (eps_s * V_macro))
    
    np.testing.assert_allclose(Li_added_e, Li_removed_s, rtol=1e-8,
        err_msg="Hierarchical Mass Leak Detected! The mass of lithium entering the macroscopic "
                "mesh does not equal the mass leaving the microscopic mesh.")




# ==============================================================================
# ORACLE 6: Chen2020 Topological Overlap (The Off-By-One Bug)
# ==============================================================================

class Chen2020TopologicalOverlapOracle(fx.PDE):
    """
    Exposes the floating-point `int(round(...))` bug in Domain.region.
    Using the exact mesh sizing from the Chen2020 DFN model (35, 6, 31 nodes).
    """
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=72, name="cell")
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=35, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=6, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=31, name="x_p")
    
    c = fx.State(domain=cell, name="c")

    def math(self):
        return {
            "equations": {
                # Assign a distinct prime number to each region's derivative
                self.c: fx.Piecewise({
                    self.x_n: fx.dt(self.c) == 2.0,
                    self.x_s: fx.dt(self.c) == 3.0,
                    self.x_p: fx.dt(self.c) == 5.0
                })
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 0.0
            }
        }



@REQUIRES_COMPILER
def test_oracle_chen2020_topological_overlap_off_by_one():
    """
    PROBE: Fails if the `Domain.region` math creates overlapping indices 
    or orphans nodes at the right boundary due to Python 3 float rounding.
    """
    engine = fx.Engine(model=Chen2020TopologicalOverlapOracle(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0, parameters={})
    
    off_c, _ = engine.layout.state_offsets["c"]
    c_residuals = res[off_c : off_c + 72]
    
    # Residual = ydot - RHS = 0.0 - RHS
    # Total sum of RHS should be 35*2 + 6*3 + 31*5 = 70 + 18 + 155 = 243
    total_rhs = -np.sum(c_residuals)
    
    assert total_rhs == pytest.approx(243.0), \
        f"Topological Off-By-One Bug Confirmed! Expected total derivative sum of 243.0, " \
        f"but got {total_rhs}. The sub-mesh indices are overlapping and overwriting each other!"
        
    # Specifically check the final node. If it's orphaned, it evaluates to 0.0 instead of 5.0
    assert c_residuals[-1] == pytest.approx(-5.0), \
        "The rightmost node of the cell was orphaned (never evaluated) because the cathode region shifted left!"




# ==============================================================================
# ORACLE 7: EIS Mass Matrix Extraction (Engine Bug)
# ==============================================================================

class CapacitiveImpedanceOracle(fx.PDE):
    """
    Proves the Engine's Analytical EIS solver incorrectly extracts the Mass Matrix.
    By allowing non-unit multipliers on the time derivative (C * dt(V)), the 
    true mass matrix M = C. The engine currently hardcodes M = id_arr (1.0).
    """
    V = fx.State(domain=None, name="V")
    C_cap = fx.Parameter(default=5.0, name="C_cap")
    R = fx.Parameter(default=2.0, name="R")
    i_app = fx.Parameter(default=1.0, name="i_app")
    
    def math(self):
        return {
            "equations": {
                # Implicit capacity: M = 5.0
                self.V: self.C_cap * fx.dt(self.V) == self.i_app - self.V / self.R
            },
            "boundaries": {},
            "initial_conditions": {self.V: 0.0}
        }



@REQUIRES_COMPILER
def test_oracle_eis_mass_matrix_extraction():
    """
    PROBE: Compares the simulated EIS against the exact analytical Transfer Function.
    """
    engine = fx.Engine(model=CapacitiveImpedanceOracle(), target="cpu", mock_execution=False)
    session = engine.start_session(parameters={"C_cap": 5.0, "R": 2.0, "i_app": 1.0})
    session.reach_steady_state()
    
    w_arr = np.array([0.1, 1.0, 10.0])
    eis_res = session.solve_eis(w_arr, input_var="i_app", output_var="V")
    
    Z_sim = eis_res["Z_real"].data + 1j * eis_res["Z_imag"].data
    
    # Exact Analytical Transfer Function: Z(w) = R / (1 + j * w * R * C)
    # Note: solve_eis treats the input array as frequencies in Hz, converting 
    # to rad/s natively. We must do the same for the analytical truth.
    w_rad = w_arr * 2 * np.pi
    Z_exact = 2.0 / (1.0 + 1j * w_rad * 2.0 * 5.0)
    
    np.testing.assert_allclose(
        np.real(Z_sim), np.real(Z_exact), rtol=1e-4,
        err_msg="EIS Mass Matrix Bug! The engine is hardcoding M=1.0 instead of extracting M=5.0."
    )




# ==============================================================================
# ORACLE 8: Continuous Adjoint VJP Sensitivities (Engine Bug)
# ==============================================================================

class AdjointCapacityOracle(fx.PDE):
    """
    Proves the Continuous Adjoint solver (adjoint.rs) also suffers from the 
    hardcoded Mass Matrix bug.
    """
    y = fx.State(domain=None, name="y")
    C_cap = fx.Parameter(default=2.0, name="C_cap")
    k = fx.Parameter(default=1.0, name="k")
    
    def math(self):
        return {
            "equations": {
                self.y: self.C_cap * fx.dt(self.y) == -self.k * self.y
            },
            "boundaries": {},
            "initial_conditions": {self.y: 1.0}
        }



@REQUIRES_COMPILER
def test_oracle_adjoint_mass_matrix_vjp():
    """
    PROBE: Compares the Enzyme-derived continuous Adjoint gradient to an exact 
    Scipy-derived analytical ground truth.
    """
    engine = fx.Engine(model=AdjointCapacityOracle(), target="cpu", mock_execution=False)
    t_eval = np.linspace(0, 5.0, 50)
    
    # Forward Pass
    res = engine.solve(t_eval=t_eval, parameters={"C_cap": 2.0, "k": 1.0}, requires_grad=["C_cap"])
    
    # Loss = Sum( y^2 )
    y_sim = res["y"].data
    loss_val = float(np.sum(y_sim ** 2))
    
    # Manual backprop to inject into the engine
    dl_dy = 2.0 * y_sim
    res.trajectory["requires_grad"] = ["C_cap"]
    
    # Trigger native Adjoint pass
    loss_obj = fx.metrics.Loss(loss_val, engine=engine, trajectory=res.trajectory, dl_dy_mapped=np.expand_dims(dl_dy, axis=1))
    grads = loss_obj.backward()
    
    simulated_grad = grads["C_cap"]
    
    # Exact Analytical Oracle
    # y(t) = exp(-k * t / C)
    # dLoss/dC = sum( d/dC [exp(-2 * k * t / C)] )
    # dLoss/dC = sum( exp(-2 * k * t / C) * (2 * k * t / C^2) )
    exact_grad = np.sum( np.exp(-2.0 * 1.0 * t_eval / 2.0) * (2.0 * 1.0 * t_eval / (2.0**2)) )
    
    np.testing.assert_allclose(
        simulated_grad, exact_grad, rtol=2e-2,
        err_msg="Adjoint Mass Matrix Bug! The VJP loop in Rust is likely ignoring the capacity multiplier."
    )



# ==============================================================================
# ORACLE 9: Variable Transference Mass Leak (Literature Inconsistency)
# ==============================================================================

class VariableTransferenceLeakProbe(fx.PDE):
    """
    Proves that `(1 - t_+) * j` leaks mass when t_plus is a spatial field.
    The strictly conservative form is: -div(-D*grad(c) + t_+*i_e / F) + j/F
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    
    c_leaky = fx.State(domain=x, name="c_leaky")
    c_strict = fx.State(domain=x, name="c_strict")
    
    def math(self):
        F = 96485.0
        j_val = 1000.0
        
        # Create a variable t_plus field
        t_plus = 0.2 + 0.1 * self.x.coords 
        
        # Manufactured i_e gradient to trigger the leak
        i_e = 10.0 * self.x.coords 
        
        # 1. The published formulation from Table S2 of O'Regan 2022
        flux_leaky = -1e-10 * fx.grad(self.c_leaky)
        eq_leaky = -fx.div(flux_leaky) + (1.0 - t_plus) * j_val / F
        
        # 2. The mathematically strict, physically conservative formulation
        flux_strict = -1e-10 * fx.grad(self.c_strict) + (t_plus * i_e) / F
        eq_strict = -fx.div(flux_strict) + j_val / F
        
        return {
            "equations": {
                self.c_leaky: fx.dt(self.c_leaky) == eq_leaky,
                self.c_strict: fx.dt(self.c_strict) == eq_strict
            },
            "boundaries": {
                flux_leaky: {"left": 0.0, "right": 0.0},
                flux_strict: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c_leaky: 1000.0,
                self.c_strict: 1000.0
            }
        }



@pytest.mark.skip(reason="The O'Regan 2022 paper publishes a non-conservative electrolyte mass equation.")
@REQUIRES_COMPILER
def test_oracle_literature_transference_mass_leak():
    """
    PROBE: Integrates total electrolyte mass. If it drifts from the exact analytical 
    influx of `j / F`, mass has leaked. 
    
    This test is skipped. The user script faithfully reproduces the DFN equations
    published in the O'Regan 2022 paper, but the paper itself contains an error.
    """
    engine = fx.Engine(model=VariableTransferenceLeakProbe(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 10.0), t_eval=np.array([0.0, 10.0]))
    
    # Exact analytical mass added = j_val / F * time * length
    exact_mass_added = (1000.0 / 96485.0) * 10.0 * 1.0
    
    mass_leaky = np.mean(res["c_leaky"].data[-1]) - 1000.0
    mass_strict = np.mean(res["c_strict"].data[-1]) - 1000.0
    
    np.testing.assert_allclose(mass_strict, exact_mass_added, rtol=1e-4)
    
    np.testing.assert_allclose(
        mass_leaky, exact_mass_added, rtol=1e-4,
        err_msg="Literature Inconsistency! The user faithfully implemented Table S2 of O'Regan 2022, "
                "which uses the simplified source term `(1 - t_plus) * j / F`. However, because Eq 21 "
                "defines t_plus as a spatial variable, factoring it outside the divergence operator "
                "mathematically omits the `i_e * grad(t_plus)` term, causing a global mass leak."
    )
