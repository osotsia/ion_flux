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
Compiler Bug Oracle: Composite Domain Traversal Failures

This suite isolates two critical failures in the AST-to-C++ lowering pass 
when dealing with 2D Composite Domains (Macro x Micro). It proves that:
1. `Piecewise` equations fail to unroll across the outer dimension.
2. `fx.integral` drops the outer dimension index, evaluating the first slice repeatedly.
"""



# ==============================================================================
# Bug 1: Piecewise Outer-Loop Omission
# ==============================================================================

class PiecewiseCompositeOracle(fx.PDE):
    y = fx.Domain(bounds=(0, 1), resolution=2, name="y")
    x = fx.Domain(bounds=(0, 1), resolution=4, name="x")
    
    x_L = x.region(bounds=(0, 0.5), resolution=2, name="x_L")
    x_R = x.region(bounds=(0.5, 1), resolution=2, name="x_R")
    
    # State is 2D: y * x. Size = 2 * 4 = 8 nodes.
    c = fx.State(domain=y * x, name="c")
    
    def math(self):
        return {
            "equations": {
                # If Piecewise correctly unrolls over the outer domain (y),
                # all 8 nodes should be assigned a residual.
                self.c: fx.Piecewise({
                    self.x_L: fx.dt(self.c) == 1.0,
                    self.x_R: fx.dt(self.c) == 2.0
                })
            },
            "boundaries": {},
            "initial_conditions": {self.c: 0.0}
        }



@REQUIRES_COMPILER
def test_piecewise_composite_loop_unrolling():
    """
    PROBE: Proves that Piecewise blocks on a 2D state emit a 1D loop that 
    orphans the outer dimension.
    """
    engine = fx.Engine(model=PiecewiseCompositeOracle(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y, ydot = np.zeros(N).tolist(), np.zeros(N).tolist()
    
    # Evaluate residual: Res = ydot - rhs = 0.0 - rhs -> rhs = -Res
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # Exact mathematical expectation (All 8 nodes evaluated across both y slices):
    # y=0, x_L (idx 0, 1) -> rhs = 1.0 -> res = -1.0
    # y=0, x_R (idx 2, 3) -> rhs = 2.0 -> res = -2.0
    # y=1, x_L (idx 4, 5) -> rhs = 1.0 -> res = -1.0
    # y=1, x_R (idx 6, 7) -> rhs = 2.0 -> res = -2.0
    expected_res = [-1.0, -1.0, -2.0, -2.0, -1.0, -1.0, -2.0, -2.0]
    
    # If the bug is active, indices 4,5,6,7 will be 0.0 because the naive C++ loop 
    # stops at `end_idx = 4`.
    np.testing.assert_allclose(
        res, expected_res, atol=1e-12,
        err_msg="BUG DETECTED: Piecewise compilation failed to unroll over the outer composite dimension! "
                "Nodes in the outer dimension were completely orphaned (residual 0.0)."
    )



# ==============================================================================
# Bug 2: Partial Integral Context Dropping
# ==============================================================================

class PartialIntegrationOracle(fx.PDE):
    y = fx.Domain(bounds=(0, 2), resolution=2, name="y")
    x = fx.Domain(bounds=(0, 1), resolution=3, name="x")
    
    c = fx.State(domain=y * x, name="c")
    c_avg = fx.State(domain=y, name="c_avg")
    
    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                # c_avg is a 1D state on `y`. It integrates `c` over `x`.
                self.c_avg: self.c_avg == fx.integral(self.c, over=self.x)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 0.0, self.c_avg: 0.0
            }
        }



@REQUIRES_COMPILER
def test_partial_integral_outer_index_dropping():
    """
    PROBE: Proves that `fx.integral` over a 1D sub-domain drops the outer loop index, 
    evaluating the first slice repeatedly for every iteration.
    """
    engine = fx.Engine(model=PartialIntegrationOracle(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y_arr, ydot_arr = np.zeros(N), np.zeros(N)
    
    off_c, _ = engine.layout.state_offsets["c"]
    
    # Populate c such that y=0 has values 10.0, and y=1 has values 20.0
    # x resolution is 3. So indices 0,1,2 are y=0. Indices 3,4,5 are y=1.
    y_arr[off_c : off_c + 3] = 10.0
    y_arr[off_c + 3 : off_c + 6] = 20.0
    
    res = engine.evaluate_residual(y_arr.tolist(), ydot_arr.tolist(), parameters={})
    
    off_avg, _ = engine.layout.state_offsets["c_avg"]
    
    # Res = ydot - rhs = 0.0 - integral -> integral = -Res
    evaluated_integrals = -np.array(res[off_avg : off_avg + 2])
    
    # Mathematical truth:
    # Volume of x cells = dx = 1.0/2 = 0.5 (with 0.25 at faces). Total volume = 1.0.
    # Integral at y=0 is 10.0 * 1.0 = 10.0.
    # Integral at y=1 is 20.0 * 1.0 = 20.0.
    expected_integrals = [10.0, 20.0]
    
    # If the bug is active, the integral C++ lambda drops the `y` index and always 
    # evaluates `c` using `int_idx` (0,1,2). Thus both integrals will equal 10.0!
    np.testing.assert_allclose(
        evaluated_integrals, expected_integrals, atol=1e-12,
        err_msg="BUG DETECTED: Partial integration over a composite domain dropped the outer loop index! "
                "The integral evaluated the first slice repeatedly for all outer dimensions."
    )



# ==============================================================================
# Bug Isolation Model
# ==============================================================================

class CompositeMaskingBugOracle(fx.PDE):
    """
    Constructs a 2D composite domain (y * x) with a total of 3 * 4 = 12 nodes.
    y resolution = 3 (Outer dimension)
    x resolution = 4 (Inner dimension)
    
    Flattened array indices for the 2D grid:
    y=0: [0, 1, 2, 3]
    y=1: [4, 5, 6, 7]
    y=2: [8, 9, 10, 11]
    """
    y = fx.Domain(bounds=(0, 1), resolution=3, name="y")
    x = fx.Domain(bounds=(0, 1), resolution=4, name="x")
    
    # Sub-regions for piecewise logic on the inner dimension
    x_L = x.region(bounds=(0, 0.5), resolution=2, name="x_L")
    x_R = x.region(bounds=(0.5, 1), resolution=2, name="x_R")
    
    # 2D States (Size 12 each)
    c_piece = fx.State(domain=y * x, name="c_piece")
    c_dir = fx.State(domain=y * x, name="c_dir")
    
    def math(self):
        return {
            "equations": {
                # Because fx.dt() is used in both regions, ALL 12 nodes should be
                # marked as differential (1.0) in the id_arr mask.
                self.c_piece: fx.Piecewise({
                    self.x_L: fx.dt(self.c_piece) == 1.0,
                    self.x_R: fx.dt(self.c_piece) == 2.0
                }),
                
                # Standard equation: initializes all 12 nodes as differential (1.0)
                self.c_dir: fx.dt(self.c_dir) == fx.grad(self.c_dir)
            },
            "boundaries": {
                # Dirichlet boundaries force nodes to become algebraic constraints (0.0).
                # This should apply to the left (x=0) and right (x=3) faces for ALL y slices.
                # Left nodes (0.0):  0, 4, 8
                # Right nodes (0.0): 3, 7, 11
                # Bulk nodes (1.0):  1, 2, 5, 6, 9, 10
                self.c_dir: {
                    "left": fx.Dirichlet(0.0), 
                    "right": fx.Dirichlet(1.0)
                }
            },
            "initial_conditions": {
                self.c_piece: 0.0,
                self.c_dir: 0.0
            }
        }



# ==============================================================================
# Tests
# ==============================================================================

def test_piecewise_composite_dae_masking_bug():
    """
    PROBE 1: Validates if `fx.Piecewise` correctly masks outer dimensions.
    If the bug is present, the engine only reads `reg["start_idx"]` (0) and 
    `reg["end_idx"]` (4), masking indices 0-3 as 1.0, but silently leaving 
    indices 4-11 as 0.0 (Algebraic constraints).
    """
    engine = fx.Engine(model=CompositeMaskingBugOracle(), target="cpu", mock_execution=True)
    
    _, _, id_arr, _, _ = engine._extract_metadata()
    id_arr = np.array(id_arr)
    
    off_p, size_p = engine.layout.state_offsets["c_piece"]
    c_piece_mask = id_arr[off_p : off_p + size_p]
    
    # Mathematical Truth: The entire 2D field contains time derivatives.
    expected_mask = np.ones(12)
    
    np.testing.assert_allclose(
        c_piece_mask, expected_mask,
        err_msg="BUG DETECTED: `fx.Piecewise` failed to unroll the ID mask across the outer dimension. "
                "Nodes in y=1 and y=2 were falsely tagged as algebraic DAE constraints (0.0), "
                "which disables the SUNDIALS truncation error solver for those nodes!"
    )



def test_dirichlet_composite_dae_masking_bug():
    """
    PROBE 2: Validates if `fx.Dirichlet` correctly masks outer dimensions.
    If the bug is present, the engine only masks `offset` (0) and 
    `offset + size - 1` (11). Indices 4, 8, 3, and 7 are left as 1.0 (PDEs), 
    causing a structurally singular system or constraint divergence.
    """
    engine = fx.Engine(model=CompositeMaskingBugOracle(), target="cpu", mock_execution=True)
    
    _, _, id_arr, _, _ = engine._extract_metadata()
    id_arr = np.array(id_arr)
    
    off_d, size_d = engine.layout.state_offsets["c_dir"]
    c_dir_mask = id_arr[off_d : off_d + size_d]
    
    # Mathematical Truth: 
    # x=0 -> indices 0, 4, 8 are left boundary (0.0)
    # x=3 -> indices 3, 7, 11 are right boundary (0.0)
    # Bulk -> indices 1, 2, 5, 6, 9, 10 are PDE (1.0)
    expected_mask = np.array([
        0.0, 1.0, 1.0, 0.0,  # y=0
        0.0, 1.0, 1.0, 0.0,  # y=1
        0.0, 1.0, 1.0, 0.0   # y=2
    ])
    
    np.testing.assert_allclose(
        c_dir_mask, expected_mask,
        err_msg="BUG DETECTED: `fx.Dirichlet` failed to apply boundary masking across the outer dimension. "
                "Intermediate boundary nodes (4, 8, 3, 7) were falsely left as differential states (1.0), "
                "causing severe Jacobian rank deficiencies during implicit integration."
    )
