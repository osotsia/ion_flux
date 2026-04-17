"""
Compiler Bug Oracle: Composite Domain Metadata Masking

This suite isolates the indexing failure in `engine.py`'s `_extract_metadata` pass.
It proves that when a State is evaluated over a 2D composite domain (e.g., y * x),
the Python frontend incorrectly applies 1D flattening logic to `id_arr` (the
differential/algebraic mask). 

Specifically:
1. `fx.Piecewise` equations only mask the first slice of the outer dimension, leaving
   the rest of the spatial field falsely tagged as algebraic DAEs (0.0).
2. `fx.Dirichlet` boundaries only mask the absolute first and last nodes of the entire
   flattened array, leaving the intermediate boundary slices falsely tagged as PDEs (1.0).
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine

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
    engine = Engine(model=CompositeMaskingBugOracle(), target="cpu", mock_execution=True)
    
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
    engine = Engine(model=CompositeMaskingBugOracle(), target="cpu", mock_execution=True)
    
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

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])