"""
test_pipeline_diagnostics.py

Validates core solver diagnostics: Finite volume stencil mathematical accuracy
and Matrix Rank preservation (Boundary Condition collisions).
"""

import pytest
import numpy as np
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

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

class DiffusionModel(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    
    def math(self):
        return {
            "regions": { self.x: [ fx.dt(self.c) == fx.div(fx.grad(self.c)) ] },
            "boundaries": [ self.c.left == 1.0, self.c.right == 0.0 ]
        }

@REQUIRES_COMPILER
def test_stencil_deformation_numerical():
    """Mathematically proves the spatial discretization stencil is exact using a quadratic profile."""
    engine = Engine(model=DiffusionModel(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y = [0.0] * N
    ydot = [0.0] * N
    
    off_c, size_c = engine.layout.state_offsets["c"]
    
    # Inject a quadratic profile: c(x) = x^2
    # Nodes at x = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_coords = np.linspace(0, 1, 5)
    y[off_c : off_c + size_c] = x_coords ** 2
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # Math Oracle: 
    # Equation is dt(c) == div(grad(c)) -> res = ydot - d2c/dx2
    # If c(x) = x^2, then d2c/dx2 = 2.0 everywhere.
    # We strictly check the center node to avoid FVM half-cell boundary truncation artifacts.
    
    # Index 2 is the exact center node (x=0.5)
    center_res = res[off_c + 2] 
    expected_center = -2.0
    
    assert center_res == pytest.approx(expected_center), \
        f"Internal spatial stencil deformed. Expected {expected_center}, got {center_res}"

@REQUIRES_COMPILER
def test_boundary_condition_rank_preservation():
    """Proves boundary conditions successfully mask bulk equations by ensuring full Jacobian rank."""
    engine = Engine(model=DiffusionModel(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y = [0.5] * N
    ydot = [0.0] * N
    
    # Evaluate the numerical Jacobian via the FFI
    # If a boundary condition failed to emit, the first/last row would be all zeros
    # or identical to the adjacent node, causing a rank deficiency.
    J = engine.evaluate_jacobian(y, ydot, parameters={}, c_j=1.0)
    
    # Convert dense representation to numpy array for rank check
    J_matrix = np.array(J).reshape((N, N))
    
    rank = np.linalg.matrix_rank(J_matrix)
    
    assert rank == N, f"Jacobian is rank-deficient (Rank {rank} / {N}). Boundary masking failed."