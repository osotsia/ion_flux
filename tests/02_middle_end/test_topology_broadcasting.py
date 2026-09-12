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


"""
Runtime Execution: Topological Masking and Boundary Translation

This suite acts as an oracle against index shadowing and coordinate space 
mismatches during C++ lowering. It ensures that boundary conditions applied 
to sub-meshes, hierarchical cross-products, and heavily wrapped AST nodes 
are accurately translated to the correct spatial indices.
"""



# ==============================================================================
# Models
# ==============================================================================

class PiecewiseShadowingPDE(fx.PDE):
    """
    Category 1: Piecewise Index Shadowing.
    If the compiler uses local sub-mesh sizing (e.g. N=5) to check the boundary 
    (idx == N-1), but the loop iterates in global space (idx = 5 to 9), the 
    boundary will be silently bypassed.
    """
    cell = fx.Domain(bounds=(0, 10.0), resolution=10)
    reg_A = cell.region(bounds=(0, 5.0), resolution=5, name="reg_A")
    reg_B = cell.region(bounds=(5.0, 10.0), resolution=5, name="reg_B")
    
    c = fx.State(domain=cell, name="c")
    
    def math(self):
        flux_A = -fx.grad(self.c)
        flux_B = -fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_A: fx.dt(self.c) == -fx.div(flux_A),
                    self.reg_B: fx.dt(self.c) == -fx.div(flux_B)
                })
            },
            "boundaries": {
                flux_B: {"right": 100.0} # Applied to the right of B
            },
            "initial_conditions": {self.c: 0.0}
        }




class HierarchicalMicroMaskingPDE(fx.PDE):
    """
    Category 2: Macro-Micro Indexing.
    Proves that a boundary applied to the 'micro' domain evaluates at the edge 
    of EVERY micro-particle across the macroscopic dimension, not just the 
    absolute last node in the flattened array.
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=3, name="x")
    r = fx.Domain(bounds=(0, 1.0), resolution=4, name="r")
    
    c = fx.State(domain=x * r, name="c")
    
    def math(self):
        flux = -fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r)
            },
            "boundaries": {
                flux: {"right": 50.0} # Applied to the surface of the particle
            },
            "initial_conditions": {self.c: 0.0}
        }




class ASTTagStrippingPDE(fx.PDE):
    """
    Category 3: AST Operator Wrapping.
    Proves that heavily modifying a tensor (flux * 2.0 + 1.0) does not strip 
    the internal `_bc_id` required for the compiler to intercept the evaluation.
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=6, name="x")
    c = fx.State(domain=x, name="c")
    
    def math(self):
        base_flux = -fx.grad(self.c)
        # Deeply wrap the tensor in the AST
        complex_flux = base_flux * 2.0 + 1.0 
        
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(complex_flux)
            },
            "boundaries": {
                complex_flux: {"right": 42.0}
            },
            "initial_conditions": {self.c: 0.0}
        }




class ExplicitSurfaceAPIPDE(fx.PDE):
    """
    Category 4: The Escape Hatch API.
    Validates that explicitly calling `tensor.surface(domain, side)` forcibly 
    overrides contextual domain inference and evaluates the boundary correctly.
    """
    cell = fx.Domain(bounds=(0, 10.0), resolution=10)
    reg_A = cell.region(bounds=(0, 5.0), resolution=5, name="reg_A")
    reg_B = cell.region(bounds=(5.0, 10.0), resolution=5, name="reg_B")
    
    c = fx.State(domain=cell, name="c")
    
    def math(self):
        flux = -fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_A: fx.dt(self.c) == -fx.div(flux),
                    self.reg_B: fx.dt(self.c) == -fx.div(flux)
                })
            },
            "boundaries": {
                # Force the compiler to recognize the boundary of reg_B
                flux.surface(domain=self.reg_B, side="right"): 99.0
            },
            "initial_conditions": {self.c: 0.0}
        }



# ==============================================================================
# Tests
# ==============================================================================

@REQUIRES_COMPILER
def test_piecewise_index_shadowing():
    engine = fx.Engine(model=PiecewiseShadowingPDE(), target="cpu", mock_execution=False)
    N = engine.layout.n_states
    y, ydot = np.zeros(N), np.zeros(N)
    
    # Evaluate instantaneous residual
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    # If y=0, grad(c)=0. 
    # div(flux) = (flux_out - flux_in) / (0.5 * dx). 
    # At the right boundary, flux_out = 100.0, flux_in = 0.0.
    # Residual F = ydot - (-div(flux)) = 0 + (100.0 / (0.5 * dx))
    dx = 10.0 / 9.0  # Strict Face-Sharing staggered boundary centers
    expected_res = 100.0 / (0.5 * dx)
    
    # The rightmost node globally is index 9. 
    assert res[-1] == pytest.approx(expected_res), \
        "Piecewise boundary was silently bypassed due to global/local index shadowing!"
    
    # Ensure it wasn't accidentally applied to the end of reg_A (index 4)
    assert res[4] == pytest.approx(0.0)




@REQUIRES_COMPILER
def test_hierarchical_micro_masking():
    engine = fx.Engine(model=HierarchicalMicroMaskingPDE(), target="cpu", mock_execution=False)
    N = engine.layout.n_states # x.res(3) * r.res(4) = 12 states
    y, ydot = np.zeros(N), np.zeros(N)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    # Micro grid has res=4. The 'right' boundaries are indices 3, 7, 11
    # dx for r = 1.0 / 3 = 0.33333
    dx_r = 1.0 / 3.0
    expected_res = 50.0 / (0.5 * dx_r)
    
    # Check all macro boundaries
    for bnd_idx in [3, 7, 11]:
        assert res[bnd_idx] == pytest.approx(expected_res), \
            f"Micro-domain boundary bypassed at flat index {bnd_idx}. Macro-tiling failed."
            
    # Check a random internal node
    assert res[5] == pytest.approx(0.0)




@REQUIRES_COMPILER
def test_ast_operator_tag_stripping():
    engine = fx.Engine(model=ASTTagStrippingPDE(), target="cpu", mock_execution=False)
    N = engine.layout.n_states
    y, ydot = np.zeros(N), np.zeros(N)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    # dx = 1.0 / 5 = 0.2
    # math evaluates to (42.0 - 1.0) / (0.5 * dx) = 410.0 
    # (since the complex_flux is evaluated on the inner node face at 1.0).
    dx = 0.2
    expected_res = (42.0 - 1.0) / (0.5 * dx)
    
    assert res[-1] == pytest.approx(expected_res), \
        "AST Operator wrapping stripped the _bc_id tag. The boundary was not injected."




@REQUIRES_COMPILER
def test_explicit_surface_api_override():
    engine = fx.Engine(model=ExplicitSurfaceAPIPDE(), target="cpu", mock_execution=False)
    N = engine.layout.n_states
    y, ydot = np.zeros(N), np.zeros(N)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    dx = 10.0 / 9.0
    expected_res = 99.0 / (0.5 * dx)
    
    assert res[-1] == pytest.approx(expected_res), \
        "The explicit `tensor.surface(domain, side)` API failed to override the context masking."



# ==============================================================================
# Bug Isolation Model
# ==============================================================================

class CompositeCoordsOracle(fx.PDE):
    """
    Manufactures a 2D composite domain with highly predictable integer coordinates.
    - Outer Domain (y): bounds=(0, 2), resolution=3  --> dx_y = 1.0. Coords: [0, 1, 2]
    - Inner Domain (x): bounds=(0, 3), resolution=4  --> dx_x = 1.0. Coords: [0, 1, 2, 3]
    
    The composite domain (y * x) flattens into an array of size 12.
    Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    """
    dom_y = fx.Domain(bounds=(0, 2.0), resolution=3, name="dom_y")
    dom_x = fx.Domain(bounds=(0, 3.0), resolution=4, name="dom_x")
    
    comp_domain = dom_y * dom_x
    
    # Track the coordinates of the inner and outer dimensions independently
    c_inner_coords = fx.State(domain=comp_domain, name="c_inner_coords")
    c_outer_coords = fx.State(domain=comp_domain, name="c_outer_coords")

    def math(self):
        return {
            "equations": {
                # dt(c) = coords. Evaluated at y=0, ydot=0, the residual F = ydot - rhs = -coords
                self.c_inner_coords: fx.dt(self.c_inner_coords) == self.dom_x.coords,
                self.c_outer_coords: fx.dt(self.c_outer_coords) == self.dom_y.coords
            },
            "boundaries": {},
            "initial_conditions": {
                self.c_inner_coords: 0.0,
                self.c_outer_coords: 0.0
            }
        }



# ==============================================================================
# The Test
# ==============================================================================

@REQUIRES_COMPILER
def test_composite_coords_broadcasting_bug():
    """
    PROBE: Evaluates the instantaneous coordinate maps generated by the AST compiler.
    If the bug is present, the compiler evaluates `flat_index * dx`, resulting in 
    both arrays reading `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`.
    """
    engine = fx.Engine(model=CompositeCoordsOracle(), target="cpu", mock_execution=False)
    
    # Extract metadata and create zeroed state arrays
    N = engine.layout.n_states
    y = np.zeros(N).tolist()
    ydot = np.zeros(N).tolist()
    
    # Evaluate the instantaneous residual
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    off_inner, size_inner = engine.layout.state_offsets["c_inner_coords"]
    off_outer, size_outer = engine.layout.state_offsets["c_outer_coords"]
    
    # Because F = ydot - rhs = 0 - coords, the evaluated coordinates are -F
    simulated_inner = -np.array(res[off_inner : off_inner + size_inner])
    simulated_outer = -np.array(res[off_outer : off_outer + size_outer])
    
    # --- Exact Mathematical Truth ---
    # Inner dimension (x) cycles rapidly: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    expected_inner = np.array([0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.])
    
    # Outer dimension (y) cycles slowly: 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2
    expected_outer = np.array([0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.])
    
    # 1. Assert Inner Coordinate Broadcasting
    np.testing.assert_allclose(
        simulated_inner, expected_inner, atol=1e-12,
        err_msg=f"\nBUG DETECTED (Inner Coords): The AST compiler failed to apply the modulo operator "
                f"`(i % res) * dx` for the inner composite dimension.\n"
                f"Expected: {expected_inner}\n"
                f"Got:      {simulated_inner}\n"
                f"This causes severe unphysical scaling in models like the 1+1D SPMe!"
    )
    
    # 2. Assert Outer Coordinate Broadcasting
    np.testing.assert_allclose(
        simulated_outer, expected_outer, atol=1e-12,
        err_msg=f"\nBUG DETECTED (Outer Coords): The AST compiler failed to apply the division operator "
                f"`(i / inner_res) * dx` for the outer composite dimension.\n"
                f"Expected: {expected_outer}\n"
                f"Got:      {simulated_outer}\n"
    )



# ==============================================================================
# Bug Isolation Model
# ==============================================================================

class MissingParentIntegrationOracle(fx.PDE):
    # A parent domain split into two halves
    cell = fx.Domain(bounds=(0, 2.0), resolution=20)
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 2.0), resolution=10, name="reg_B")
    
    # State bound to the global parent
    c_parent = fx.State(domain=cell, name="c_parent")
    
    # 0D target to hold the integral evaluation
    int_val = fx.State(domain=None, name="int_val")

    def math(self):
        return {
            "equations": {
                self.c_parent: fx.dt(self.c_parent) == 0.0,
                
                # We integrate the parent state ONLY over the second half (reg_B).
                # Crucially, NO state is bound to reg_B, forcing the compiler 
                # to use the fallback `MockDomain`.
                self.int_val: self.int_val == fx.integral(self.c_parent, over=self.reg_B)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c_parent: 0.0, 
                self.int_val: 0.0
            }
        }



# ==============================================================================
# The Test
# ==============================================================================

@REQUIRES_COMPILER
def test_unbound_subregion_memory_mapping():
    """
    If the fallback MockDomain lacks a `.parent` attribute, the integral over 
    reg_B will fail to add the `start_idx` offset (10). It will silently read 
    indices 0-9 (reg_A) instead of 10-19 (reg_B).
    """
    engine = fx.Engine(model=MissingParentIntegrationOracle(), target="cpu", mock_execution=False)
    y = np.zeros(engine.layout.n_states)
    ydot = np.zeros(engine.layout.n_states)
    
    # Manually populate the parent state array in memory
    off_c, size_c = engine.layout.state_offsets["c_parent"]
    
    # We set the first half (reg_A) to 0.0, and the second half (reg_B) to 100.0
    y[off_c : off_c + 10] = 0.0 
    y[off_c + 10 : off_c + 20] = 100.0 
    
    # Evaluate the instantaneous residual: F = ydot - rhs = 0.0 - integral
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    off_int, _ = engine.layout.state_offsets["int_val"]
    evaluated_integral = -res[off_int]
    
    # The physical volume of reg_B is exactly 1.0. 
    # The normalized discrete FVM integral of a 100.0 field over a volume of 1.0 is exactly 100.0.
    exact_fvm_integral = 100.0
         
    assert evaluated_integral == pytest.approx(exact_fvm_integral), \
        f"BUG DETECTED: Expected integral {exact_fvm_integral}, but got {evaluated_integral}. " \
        "The compiler dropped the topological start_idx offset and read the wrong memory addresses!"
