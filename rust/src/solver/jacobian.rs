use super::NativeJvpFn;
use faer::sparse::SparseColMat;
use std::time::Instant;

/// Structural sparsity and coloring mapped offline by the Python AST frontend.
pub struct GraphColoring {
    pub num_colors: usize,
    pub color_vectors: Vec<Vec<f64>>,   // num_colors x N
    pub row_indices: Vec<usize>,        // CSC row indices
    pub col_ptrs: Vec<usize>,           // CSC column pointers
}

/// Computes the exact CSC Jacobian using Forward-Mode AD sweeps.
pub fn compute_colored_jacobian(
    n: usize,
    y: &[f64], ydot: &[f64], p: &[f64], c_j: f64,
    coloring: &GraphColoring,
    jvp_fn: NativeJvpFn,
    diag: &mut super::Diagnostics,
) -> Result<SparseColMat<usize, f64>, String> {
    
    let start_time = Instant::now();
    let mut nz_values = vec![0.0; coloring.row_indices.len()];
    let mut jvp_out = vec![0.0; n];

    // 1. Execute K forward AD sweeps
    for color in 0..coloring.num_colors {
        let seed_vector = &coloring.color_vectors[color];
        
        unsafe {
            jvp_fn(
                y.as_ptr(), ydot.as_ptr(), p.as_ptr(), c_j,
                seed_vector.as_ptr(), jvp_out.as_mut_ptr(),
            );
        }

        // 2. Scatter projection into CSC non-zero locations
        for col in 0..n {
            if seed_vector[col] == 1.0 {
                let start = coloring.col_ptrs[col];
                let end = coloring.col_ptrs[col + 1];
                for nz_idx in start..end {
                    let row = coloring.row_indices[nz_idx];
                    nz_values[nz_idx] = jvp_out[row];
                }
            }
        }
    }

    diag.jacobian_assembly_time_us += start_time.elapsed().as_micros();
    diag.jacobian_evaluations += 1;
    diag.max_chromatic_number = diag.max_chromatic_number.max(coloring.num_colors);

    // 3. Assemble Zero-Copy Faer Matrix
    SparseColMat::try_new_from_nonnegative_indices(
        n, n,
        &coloring.col_ptrs,
        &coloring.row_indices,
        &nz_values,
    ).map_err(|_| "Failed to assemble CSC matrix.".to_string())
}