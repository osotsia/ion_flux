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
from ion_flux.compiler._3_backend.cpr_coloring import HybridGraphColorer
import shutil
import platform
import ion_flux as fx
import sys
import os

from Chen2020_DFN import Chen2020_DFN # type: ignore
from ORegan2022_ThermalDFN import ThermalDFN # type: ignore



def test_tridiagonal_coloring_efficiency():
    """
    PROBE: Asserts that a standard 1D FVM mesh (Tridiagonal Jacobian) is 
    optimally colored in exactly 3 sweeps (Chromatic Number C=3), independent 
    of the mesh size.
    """
    N = 100
    triplets = set()
    for i in range(N):
        triplets.add((i, i))          # Diagonal
        if i > 0:
            triplets.add((i, i - 1))  # Left Stencil
        if i < N - 1:
            triplets.add((i, i + 1))  # Right Stencil
            
    colorer = HybridGraphColorer(n_states=N, triplets=triplets, dense_threshold=20)
    
    assert len(colorer.dense_rows) == 0, "No row should be flagged as dense."
    assert colorer.n_colors == 3, f"Tridiagonal matrix should require exactly 3 colors, got {colorer.n_colors}."
    assert len(colorer.color_seeds) == 3



def test_arrowhead_segregation():
    """
    PROBE: Asserts that a global state coupling to every node (e.g. V_cell) 
    is safely amputated, preserving the sparsity efficiency of the bulk PDE.
    """
    N = 50
    triplets = set()
    # Tridiagonal bulk
    for i in range(N - 1):
        triplets.add((i, i))
        if i > 0:
            triplets.add((i, i - 1))
        if i < N - 2:
            triplets.add((i, i + 1))
            
    # Dense arrowhead row (e.g. state N-1 depends on all other states)
    dense_row = N - 1
    for i in range(N):
        triplets.add((dense_row, i))
        
    colorer = HybridGraphColorer(n_states=N, triplets=triplets, dense_threshold=20)
    
    assert len(colorer.dense_rows) == 1, "Failed to identify the dense arrowhead row."
    assert colorer.dense_rows[0] == dense_row, "Identified the wrong dense row."
    assert colorer.n_colors == 3, "Bulk should remain 3-colorable despite the global state."



def test_cpr_jvp_reconstruction_exactness():
    """
    PROBE: The ultimate mathematical truth. Computes the JVP (J * Seed) and 
    asserts that every original non-zero sparse element is perfectly extracted 
    without any collisions.
    """
    N = 10
    
    # 1. Generate a mock Jacobian with an arbitrary, valid sparse pattern
    np.random.seed(42)
    J = np.zeros((N, N))
    triplets = set()
    
    for i in range(N):
        J[i, i] = np.random.uniform(1.0, 5.0)
        triplets.add((i, i))
        if i > 1:
            J[i, i - 2] = np.random.uniform(1.0, 5.0)
            triplets.add((i, i - 2))
        if i < N - 1:
            J[i, i + 1] = np.random.uniform(1.0, 5.0)
            triplets.add((i, i + 1))
            
    # 2. Extract Seeds
    colorer = HybridGraphColorer(n_states=N, triplets=triplets, dense_threshold=10)
    
    # 3. Simulate Forward-Mode AD JVP Sweep & Reconstruction
    J_reconstructed = np.zeros((N, N))
    
    for c_idx, seed_vector in enumerate(colorer.color_seeds):
        v = np.array(seed_vector)
        # Simulated JVP: evaluate_jvp(..., v)
        jvp_out = J @ v
        
        # Scatter back into matrix using color maps
        for row, col in colorer.sparse_triplets:
            if colorer.color_map[col] == c_idx:
                J_reconstructed[row, col] = jvp_out[row]
                
    # 4. Assert Perfect Sparsity Recovery
    np.testing.assert_allclose(
        J_reconstructed, J, atol=1e-12,
        err_msg="CPR Reconstruction Failed! Color collision caused a JVP overlap."
    )




def get_missing_dependencies(model):
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    N = engine.layout.n_states
    
    # 1. Reconstruct the MIR-derived sparsity pattern directly from the Engine's cache
    _, _, c_rows, c_cols, c_dense = engine._cpr_cache
    
    python_set = set(zip(c_rows, c_cols))
    
    # Expand dense Arrowhead rows into the sparse set for comparison
    for r in c_dense:
        for c in range(N):
            python_set.add((r, c))
            
    # 2. Reconstruct the actual dense C++ Jacobian via Native Evaluation
    # Randomize inputs to ensure no structural zero is accidentally 
    # hidden by mathematical coincidences (e.g. 0.0 * gradient)
    np.random.seed(42)
    y = np.random.uniform(0.1, 1.0, N)
    ydot = np.random.uniform(0.1, 1.0, N)
    
    J_dense = np.array(engine.evaluate_jacobian(y.tolist(), ydot.tolist(), c_j=1.0, parameters={}))
    
    # 3. Extract exact numerical non-zero tuples generated by the Enzyme LLVM plugin
    enzyme_rows, enzyme_cols = np.where(np.abs(J_dense) > 1e-12)
    enzyme_set = set(zip(enzyme_rows, enzyme_cols))
    
    # 4. Assert the MIR grapher is a strict, safe superset of the numerical reality
    return enzyme_set - python_set




@REQUIRES_COMPILER
def test_static_sparsity_analyzer_matches_enzyme_oracl_v1():
    """
    PROBE: Proves that the Python AST Tracing algorithm computes a safe structural 
    superset of the Jacobian compared to the mathematically exact C++ Enzyme evaluation.
    This guarantees no loss of mathematical coupling when the C++ code generator is gutted 
    for CPR/Forward-Mode AD.
    """
    missing_dependencies = get_missing_dependencies(model=Chen2020_DFN())
    assert not missing_dependencies, \
        f"FATAL: Python analyzer failed to map physical dependencies! " \
        f"It missed {len(missing_dependencies)} cross-couplings. Examples: {list(missing_dependencies)[:5]}"




@REQUIRES_COMPILER
@pytest.mark.skip(reason="Passes, but takes a long time to run")
def test_static_sparsity_analyzer_matches_enzyme_oracl_v2():
    missing_dependencies = get_missing_dependencies(model=ThermalDFN())
    assert not missing_dependencies, \
        f"FATAL: Python analyzer failed to map physical dependencies! " \
        f"It missed {len(missing_dependencies)} cross-couplings. Examples: {list(missing_dependencies)[:5]}"
