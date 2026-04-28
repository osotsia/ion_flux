# --- File: tests/03_backend_compilation/test_graph_coloring.py ---
import pytest
import numpy as np
from ion_flux.compiler.coloring import HybridGraphColorer

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

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])