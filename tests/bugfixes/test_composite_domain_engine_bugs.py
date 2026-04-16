"""
Compiler Bug Oracle: Composite Domain Traversal Failures

This suite isolates two critical failures in the AST-to-C++ lowering pass 
when dealing with 2D Composite Domains (Macro x Micro). It proves that:
1. `Piecewise` equations fail to unroll across the outer dimension.
2. `fx.integral` drops the outer dimension index, evaluating the first slice repeatedly.
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
    reason="Requires native C++ toolchain to evaluate the emitted AST loops."
)

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
    engine = Engine(model=PiecewiseCompositeOracle(), target="cpu", mock_execution=False)
    
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
    engine = Engine(model=PartialIntegrationOracle(), target="cpu", mock_execution=False)
    
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

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])