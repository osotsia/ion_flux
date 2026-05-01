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

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_COMPILER = pytest.mark.skipif(
    not _has_compiler(), 
    reason="Requires native C++ toolchain."
)


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
@pytest.mark.xfail(reason="Known bug. Will fix later.")
def test_disproportionate_resolution_distorts_physics():
    """
    PROBE: Asserts that integrating over the region yields its true physical bounds.
    If unfixed (or unmitigated in the user's script), this evaluates to ~10.1 
    instead of 80.0, exposing why charge conservation failed in the DFN model.
    """
    engine = Engine(model=SpatialScaleGapOracle(), target="cpu", mock_execution=False)
    
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
    engine = Engine(model=CompositeCSRExtractionOracle(), target="cpu", mock_execution=True)
    
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

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])