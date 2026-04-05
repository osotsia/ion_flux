"""
Topological Sub-Mesh Oracles

This suite proves that `Domain.region` calculations suffer from floating-point 
off-by-one errors, resulting in overlapping physics and orphaned boundary nodes.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine

# ==============================================================================
# Model for Isolation
# ==============================================================================

class LGM50TopologicalProbe(fx.PDE):
    """
    Replicates the exact LG M50 grid spacing to isolate the floating-point 
    misalignment in the AST compiler's sub-mesh calculations.
    """
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=144)
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=71, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=10, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=63, name="x_p")
    
    c = fx.State(domain=cell, name="c")
    
    def math(self):
        return {
            "equations": {
                # Assign a distinct constant derivative to each region to track 
                # exactly which region claims which spatial node in the C++ array.
                self.c: fx.Piecewise({
                    self.x_n: fx.dt(self.c) == 1.0,
                    self.x_s: fx.dt(self.c) == 2.0,
                    self.x_p: fx.dt(self.c) == 3.0
                })
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 0.0
            }
        }

# ==============================================================================
# Tests
# ==============================================================================

def test_submesh_connectivity_and_orphans():
    """
    PROBE 1: Validates that the calculated start indices perfectly align 
    with the resolutions to form a contiguous array without gaps or overlaps.
    """
    model = LGM50TopologicalProbe()
    
    n_start, n_res = model.x_n.start_idx, model.x_n.resolution
    s_start, s_res = model.x_s.start_idx, model.x_s.resolution
    p_start, p_res = model.x_p.start_idx, model.x_p.resolution
    
    # 1. Separator must start exactly where Anode ends
    assert s_start == n_start + n_res, \
        f"Overlap Error: Separator starts at {s_start}, but Anode ends at {n_start + n_res}."
        
    # 2. Cathode must start exactly where Separator ends
    assert p_start == s_start + s_res, \
        f"Overlap Error: Cathode starts at {p_start}, but Separator ends at {s_start + s_res}."
        
    # 3. Cathode must end exactly at the Parent's final node
    assert p_start + p_res == model.cell.resolution, \
        f"Orphan Error: Cathode ends at {p_start + p_res}, but Cell has {model.cell.resolution} nodes."


def test_piecewise_residual_abandonment():
    """
    PROBE 2: Evaluates the instantaneous residual to prove the PDE solver 
    abandons the final node due to the off-by-one error.
    """
    engine = Engine(model=LGM50TopologicalProbe(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y, ydot = np.zeros(N).tolist(), np.zeros(N).tolist()
    
    # Residual = ydot - RHS = 0.0 - RHS
    # Expected RHS for x_p is 3.0. Therefore expected residual is -3.0.
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # Check the final node of the cell (Index 143)
    final_node_res = res[-1]
    
    assert final_node_res == pytest.approx(-3.0), \
        f"Orphaned Node: Expected residual -3.0, got {final_node_res}. " \
        "The PDE loop terminated before reaching the physical edge of the battery."


def test_piecewise_overlap_overwrite():
    """
    PROBE 3: Proves that the overlapping boundary between x_s and x_p 
    causes the latter to silently overwrite the former.
    """
    engine = Engine(model=LGM50TopologicalProbe(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y, ydot = np.zeros(N).tolist(), np.zeros(N).tolist()
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # The true boundary between x_s and x_p in a contiguous mesh should be node 81.
    # Because x_s ends at 80 and x_p starts at 80, node 80 is processed twice.
    # The RHS for x_s is 2.0. The RHS for x_p is 3.0.
    
    node_80_res = res[80]
    
    # If the mesh was contiguous, Node 80 belongs to the Separator (RHS 2.0 -> Res -2.0)
    assert node_80_res == pytest.approx(-2.0), \
        f"Overlap Overwrite: Expected Separator physics (-2.0), got {node_80_res}. " \
        "The Cathode region silently overwrote the shared node."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])