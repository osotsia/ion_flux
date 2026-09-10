import logging
from typing import List, Any, Tuple
from ion_flux.compiler._2_middle_end.memory_layout import MemoryLayout
from ion_flux.compiler._3_backend.sparsity_tracer import SparsityAnalyzer
from ion_flux.compiler._3_backend.cpr_coloring import HybridGraphColorer

def compute_cpr(eq_stmts: List[Any], layout: MemoryLayout, jacobian_bandwidth: int) -> Tuple[List, List, List, List, List]:
    """
    Orchestrates static Compute IR tracing and Graph Coloring to produce 
    Vector-Jacobian Product (VJP) and Jacobian-Vector Product (JVP) execution schedules.
    """
    c_seeds, c_ptrs, c_rows, c_cols, c_dense = [], [], [], [], []
    
    if jacobian_bandwidth != -1:
        try:
            analyzer = SparsityAnalyzer(eq_stmts, layout)
            colorer = HybridGraphColorer(layout.n_states, analyzer.sparse_triplets, dense_threshold=20)
            
            c_seeds = colorer.color_seeds
            c_ptrs = [0]
            for c_idx in range(colorer.n_colors):
                count = 0
                for r, c in colorer.sparse_triplets:
                    if colorer.color_map[c] == c_idx:
                        c_rows.append(r)
                        c_cols.append(c)
                        count += 1
                c_ptrs.append(c_ptrs[-1] + count)
            c_dense = colorer.dense_rows
            
        except Exception as e:
            logging.warning(f"CPR Graph Coloring failed: {e}. Falling back to Dense Forward-Mode AD sweeps.")
            N = layout.n_states
            c_seeds = [[0.0] * N for _ in range(N)]
            for i in range(N): 
                c_seeds[i][i] = 1.0
                
            c_ptrs = list(range(0, N * N + 1, N))
            
            c_rows, c_cols = [], []
            for c in range(N):
                c_rows.extend(range(N))
                c_cols.extend([c] * N)
                
            c_dense = []
            
    return (c_seeds, c_ptrs, c_rows, c_cols, c_dense)