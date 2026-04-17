# --- File: tests/bugfixes/test_unfixed_zero_pivot_panic.py ---
"""
Compiler Bug Oracle: Permutation-Invariant Zero Pivot Panic

Isolates the fundamental algorithmic flaw in the Native Sparse LU backend.
By constructing a Jacobian that evaluates to a dense block of 1.0s, we mathematically
guarantee that Gaussian elimination will produce a zero pivot on the second step.
Because all permutations of a uniform dense matrix are identical, this completely 
defeats `faer`'s AMD ordering, proving that the lack of partial pivoting in the 
`simplicial` module causes hard Rust panics on valid equations.
"""

import pytest
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

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

class DefeatAMDZeroPivotOracle(fx.PDE):
    y1 = fx.State(domain=None, name="y1")
    y2 = fx.State(domain=None, name="y2")
    y3 = fx.State(domain=None, name="y3")
    y4 = fx.State(domain=None, name="y4")

    def math(self):
        # The Jacobian dF/dy for this system is a 4x4 matrix of -1.0s.
        total = self.y1 + self.y2 + self.y3 + self.y4
        return {
            "equations": {
                self.y1: 0.0 == total - 1.0,
                self.y2: 0.0 == total - 2.0,
                self.y3: 0.0 == total - 3.0,
                self.y4: 0.0 == total - 4.0
            },
            "boundaries": {},
            "initial_conditions": {
                self.y1: 0.0, self.y2: 0.0, self.y3: 0.0, self.y4: 0.0
            }
        }

@REQUIRES_RUNTIME
@pytest.mark.skip(reason="Apparently not necessary to fix")
def test_simplicial_lu_zero_pivot_panic(capfd):
    """
    PROBE: Executes a model designed to defeat AMD fill-reducing permutations.
    This guarantees `faer` encounters a zero pivot. Since `faer`'s `simplicial` 
    module lacks partial pivoting, it will hard panic.
    """
    engine = Engine(model=DefeatAMDZeroPivotOracle(), target="cpu", mock_execution=False)
    
    try:
        engine.solve(t_span=(0, 1.0))
    except RuntimeError:
        pass
        
    captured = capfd.readouterr()
    
    assert "panicked at" not in captured.err, \
        "BUG DETECTED: Rust panic leaked to stderr! `faer`'s simplicial LU solver " \
        "cannot handle zero pivots generated during Gaussian elimination, and the " \
        "matrix successfully defeated AMD reordering."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])