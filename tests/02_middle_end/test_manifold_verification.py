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
import ion_flux as fx
from ion_flux.compiler._2_middle_end.memory_layout import MemoryLayout
from ion_flux.compiler._2_middle_end.verification import verify_manifold, TopologicalError
import numpy as np
import shutil
import platform
from ion_flux._core import solve_ida_native



def test_top_down_grid_assembly():
    cell = fx.Domain(bounds=(0.0, 10.0), name="cell")
    reg_A = cell.region(bounds=(0.0, 4.0), resolution=4, name="reg_A")
    reg_B = cell.region(bounds=(4.0, 10.0), resolution=3, name="reg_B")
    
    assert cell.resolution == 7
    assert reg_B.start_idx == 4
    
    c = fx.State(domain=cell, name="c")
    layout = MemoryLayout(states=[c], parameters=[])
    
    m_list = layout.get_mesh_data()
    offsets = layout.mesh_offsets["cell"]
    
    # 7 nodes -> 6 dx_faces, 7 V_nodes, 8 A_faces
    assert "w_dx_faces" in offsets
    assert "w_V_nodes" in offsets
    assert "w_A_faces" in offsets
    
    # Total normalized volumes must sum exactly to 1.0 for cartesian mappings
    vol_off = offsets["w_V_nodes"]
    vols = m_list[vol_off : vol_off + 7]
    assert sum(vols) == pytest.approx(1.0)
    
    # Reg_A physically takes 40% of physical volume (bounds 0 to 4 out of 10)
    assert sum(vols[0:4]) == pytest.approx(0.4)
    # Reg_B physically takes 60% of physical volume
    assert sum(vols[4:7]) == pytest.approx(0.6)



def test_manifold_verification_catches_physical_gaps():
    cell = fx.Domain(bounds=(0.0, 10.0), name="cell")
    
    # Creating a physical gap from 4.0 to 5.0
    reg_A = cell.region(bounds=(0.0, 4.0), resolution=4, name="reg_A")
    reg_B = cell.region(bounds=(5.0, 10.0), resolution=3, name="reg_B")
    
    class MockModel(fx.PDE):
        c = fx.State(domain=cell, name="c")
        def math(self):
            return {"equations": {self.c: fx.dt(self.c) == 0.0}, "boundaries": {}, "initial_conditions": {}}
            
    model = MockModel()
    with pytest.raises(TopologicalError, match="Topological Gap/Overlap Detected"):
        verify_manifold(model.ast())



def test_domain_addition_deprecation():
    d1 = fx.Domain(bounds=(0, 1), resolution=10)
    d2 = fx.Domain(bounds=(1, 2), resolution=10)
    with pytest.raises(TypeError, match="deprecated"):
        d3 = d1 + d2


"""
O'Regan Crash Oracles: Spatial Scaling & Composite Graph Extraction

This suite isolates the exact compiler and memory-layout bugs that caused 
the `ORegan2022_ThermalDFN` model to catastrophically diverge at the separator 
interface during high C-rate discharges.

1. Spatial Scale Gap: Proves that non-proportional regional resolutions 
   silently distort physical length, violating mass/charge conservation.
2. Composite CSR Extraction: Proves that unstructured `m_list` metadata 
   is silently orphaned when wrapped inside a 2D `CompositeDomain`.
"""




# ==============================================================================
# ORACLE 1: The Spatial Scale Gap (Physical Length Distortion)
# ==============================================================================

class SpatialScaleGapOracle(fx.PDE):
    """
    Mimics the O'Regan uniform resolution (30, 30, 30) applied across 
    vastly different physical lengths (85.2µm, 12.0µm, 75.6µm).
    
    If the compiler forces the parent's `dx` uniformly across all regions, 
    a disproportionate resolution will warp the physical size of the region.
    """
    # Parent cell is 100 units long with 100 nodes. (Base dx ≈ 1.0)
    cell = fx.Domain(bounds=(0, 100.0), resolution=100, name="cell")
    
    # BUG TRAP: The region physically spans 80 units, but is only assigned 10 nodes.
    # The compiler will give it a volume of ~10 units instead of 80!
    reg = cell.region(bounds=(0, 80.0), resolution=10, name="reg")
    reg_rest = cell.region(bounds=(80.0, 100.0), resolution=90, name="reg_rest")
    
    c = fx.State(domain=cell, name="c")
    
    # 0D tracker to measure the perceived physical length of the region
    reg_length = fx.State(domain=None, name="reg_length")
    
    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                # Integrating a constant 1.0 over a domain mathematically yields its exact length/volume.
                self.reg_length: self.reg_length == fx.integral(self.c, over=self.reg)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 1.0,
                self.reg_length: 0.0
            }
        }



@REQUIRES_COMPILER
#@pytest.mark.xfail(reason="Known bug. Will fix later.")
def test_disproportionate_resolution_distorts_physics():
    """
    PROBE: Asserts that integrating over the region yields its true physical bounds.
    If unfixed (or unmitigated in the user's script), this evaluates to ~10.1 
    instead of 80.0, exposing why charge conservation failed in the DFN model.
    """
    engine = fx.Engine(model=SpatialScaleGapOracle(), target="cpu", mock_execution=False)
    
    # Evaluate instantaneous algebraic integration at t=0
    y0, ydot0, _, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0, parameters={})
    
    off_len, _ = engine.layout.state_offsets["reg_length"]
    
    # Residual F = ydot - rhs = 0.0 - integral -> integral = -F
    simulated_length = -res[off_len]
    expected_physical_length = 80.0
    
    assert simulated_length == pytest.approx(expected_physical_length, rel=1e-3), \
        f"Scale Gap Bug Confirmed! The region's physical length should be {expected_physical_length}, " \
        f"but the engine integrated it as {simulated_length:.2f}. " \
        "This coordinate distortion causes massive conservation divergence in coupled PDEs."




# ==============================================================================
# ORACLE 2: Composite CSR Graph Extraction
# ==============================================================================

tetrahedron_mesh = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "elements": [[0, 1, 2, 3]]
}



class CompositeCSRExtractionOracle(fx.PDE):
    """
    Constructs an unstructured 3D mesh wrapped inside a 4D CompositeDomain 
    (Macro x Micro). Evaluates if `memory.py` successfully traverses the AST 
    to extract the underlying CSR sparse structures.
    """
    mesh_3d = fx.Domain.from_mesh(tetrahedron_mesh, name="mesh_3d")
    r_micro = fx.Domain(bounds=(0, 1.0), resolution=5, coord_sys="spherical", name="r_micro")
    
    macro_micro = mesh_3d * r_micro
    
    c = fx.State(domain=macro_micro, name="c")

    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 0.0
            }
        }



@REQUIRES_COMPILER
def test_composite_domain_csr_extraction():
    """
    PROBE: Asserts that the memory layout builder correctly identifies and maps 
    the unstructured CSR graph arrays (`V_nodes`, `row_ptr`, etc.) when they 
    are hidden inside a `CompositeDomain` multiplier.
    """
    engine = fx.Engine(model=CompositeCSRExtractionOracle(), target="cpu", mock_execution=True)
    
    # If the `_get_domains()` recursion fix is absent, the compiler sees `macro_micro`, 
    # checks `getattr(macro_micro, 'csr_data', None)` which is False, and skips 
    # extracting the internal `mesh_3d` arrays entirely.
    is_extracted = "mesh_3d" in engine.layout.mesh_offsets
    
    assert is_extracted, \
        "Composite CSR Extraction Bug Confirmed! The memory builder failed to " \
        "recursively search the `CompositeDomain` for unstructured mesh data. " \
        "The Native C++ matrix-free solver will segfault attempting to read empty pointers."
        
    # Verify that the specific sub-arrays were populated
    offsets = engine.layout.mesh_offsets["mesh_3d"]
    assert "volumes" in offsets
    assert "row_ptr" in offsets
    assert "col_ind" in offsets



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
    engine = fx.Engine(model=LGM50TopologicalProbe(), target="cpu", mock_execution=False)
    
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
    engine = fx.Engine(model=LGM50TopologicalProbe(), target="cpu", mock_execution=False)
    
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
