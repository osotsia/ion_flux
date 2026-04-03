"""
Middle-End Codegen: Piecewise Auto-Stitching & Conservation

This suite acts as a Test-Driven Development (TDD) oracle for the DSL's 
ability to automatically detect and fix disjointed fluxes in piecewise spatial 
domains. It proves that explicitly separated fluxes passed to `fx.div()` 
within a `fx.Piecewise` block are automatically stitched into a conservative 
field at the boundaries.
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


# ==============================================================================
# TDD Test Cases
# ==============================================================================

@REQUIRES_COMPILER
def test_bimaterial_mass_conservation():
    """
    Proves that a 2-region piecewise evaluation conserves mass globally.
    If the interface at x=1.0 is not stitched, the divergence residuals
    will accumulate/leak mass due to differing diffusion coefficients.
    """
    engine = Engine(model=BiMaterialDiffusion(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    
    # Create an arbitrary concentration gradient
    np.random.seed(42)
    y = np.random.uniform(1.0, 5.0, size=N).tolist()
    ydot = np.zeros(N).tolist()
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # Total mass change (sum of residuals) must be exactly zero in a closed system
    total_mass_drift = np.sum(res)
    assert np.isclose(total_mass_drift, 0.0, atol=1e-12), \
        f"Piecewise domain leaked mass! Interface flux was not auto-stitched. Drift: {total_mass_drift}"


@REQUIRES_COMPILER
def test_bimaterial_jacobian_interface_coupling():
    """
    Proves that the Analytical Jacobian establishes cross-boundary coupling.
    If fluxes are disjoint, Node 9 (left of interface) will have a 0.0 derivative
    with respect to Node 10 (right of interface).
    """
    engine = Engine(model=BiMaterialDiffusion(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
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
    engine = Engine(model=TriRegionElectrolyte(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y = np.linspace(100.0, 200.0, N).tolist() # Smooth gradient across all 3 regions
    ydot = np.zeros(N).tolist()
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    total_mass_drift = np.sum(res)
    assert np.isclose(total_mass_drift, 0.0, atol=1e-12), \
        f"3-Region Piecewise domain leaked mass! Drift: {total_mass_drift}"


@REQUIRES_COMPILER
def test_complex_tensor_piecewise_stitching():
    """
    Ensures that auto-stitching doesn't break when fluxes contain complex
    multi-state dependencies (like migration `c * grad(phi)`).
    """
    engine = Engine(model=CoupledTensorFlux(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    off_c, size_c = engine.layout.state_offsets["c"]
    off_phi, size_phi = engine.layout.state_offsets["phi"]
    
    y = np.zeros(N)
    # Establish gradients for both 'c' and 'phi'
    y[off_c:off_c+size_c] = np.linspace(1.0, 5.0, size_c)
    y[off_phi:off_phi+size_phi] = np.linspace(0.0, 10.0, size_phi)
    
    ydot = np.zeros(N).tolist()
    y = y.tolist()
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # Extract 'c' residuals only (since 'phi' is an open system with Dirichlet bounds)
    res_c = res[off_c:off_c+size_c]
    
    total_mass_drift = np.sum(res_c)
    assert np.isclose(total_mass_drift, 0.0, atol=1e-12), \
        f"Coupled tensor Piecewise domain leaked mass! Drift: {total_mass_drift}"
        
    # Verify Jacobian coupling across the interface (Node 9 and Node 10)
    J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    assert abs(J[off_c + 9, off_c + 10]) > 1e-8, "Tensor Jacobian Disjoint: c(left) -> c(right)"
    assert abs(J[off_c + 9, off_phi + 10]) > 1e-8, "Tensor Jacobian Disjoint: c(left) -> phi(right)"