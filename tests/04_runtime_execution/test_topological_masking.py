"""
Runtime Execution: Topological Masking and Boundary Translation

This suite acts as an oracle against index shadowing and coordinate space 
mismatches during C++ lowering. It ensures that boundary conditions applied 
to sub-meshes, hierarchical cross-products, and heavily wrapped AST nodes 
are accurately translated to the correct spatial indices.
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

REQUIRES_COMPILER = pytest.mark.skipif(
    not _has_compiler(), 
    reason="Requires native C++ toolchain to evaluate generated boundary loops."
)

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
    cell = fx.Domain(bounds=(0, 10.0), resolution=11)
    reg_A = cell.region(bounds=(0, 5.0), resolution=6, name="reg_A")
    reg_B = cell.region(bounds=(5.0, 10.0), resolution=6, name="reg_B")
    
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
    cell = fx.Domain(bounds=(0, 10.0), resolution=11)
    reg_A = cell.region(bounds=(0, 5.0), resolution=6, name="reg_A")
    reg_B = cell.region(bounds=(5.0, 10.0), resolution=6, name="reg_B")
    
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
    engine = Engine(model=PiecewiseShadowingPDE(), target="cpu", mock_execution=False)
    N = engine.layout.n_states
    y, ydot = np.zeros(N), np.zeros(N)
    
    # Evaluate instantaneous residual
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    # If y=0, grad(c)=0. 
    # div(flux) = (flux_out - flux_in) / (0.5 * dx). 
    # At the right boundary, flux_out = 100.0, flux_in = 0.0.
    # Residual F = ydot - (-div(flux)) = 0 + (100.0 / (0.5 * dx))
    dx = 10.0 / 10.0  # Cell length 10, res 11 -> 10 intervals -> dx=1.0
    expected_res = 100.0 / (0.5 * dx)
    
    # The rightmost node globally is index 10. 
    assert res[-1] == pytest.approx(expected_res), \
        "Piecewise boundary was silently bypassed due to global/local index shadowing!"
    
    # Ensure it wasn't accidentally applied to the end of reg_A (index 5)
    assert res[5] == pytest.approx(0.0)


@REQUIRES_COMPILER
def test_hierarchical_micro_masking():
    engine = Engine(model=HierarchicalMicroMaskingPDE(), target="cpu", mock_execution=False)
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
    engine = Engine(model=ASTTagStrippingPDE(), target="cpu", mock_execution=False)
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
    engine = Engine(model=ExplicitSurfaceAPIPDE(), target="cpu", mock_execution=False)
    N = engine.layout.n_states
    y, ydot = np.zeros(N), np.zeros(N)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    dx = 1.0
    expected_res = 99.0 / (0.5 * dx)
    
    assert res[-1] == pytest.approx(expected_res), \
        "The explicit `tensor.surface(domain, side)` API failed to override the context masking."