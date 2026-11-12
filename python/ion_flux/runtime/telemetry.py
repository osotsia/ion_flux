from typing import Any, Dict


class TelemetryReport:
    """Static compile-time model topology, sparsity, and memory footprint profile."""
    __slots__ = [
        "model_len", "n_states", "n_params", "n_observables",
        "jacobian_bandwidth", "solver_type", "nnz", "sparsity",
        "cpr_colors", "dense_rows", "memory_footprint_bytes"
    ]

    def __init__(self, manifest: Any):
        layout = manifest.layout
        self.n_states = layout.n_states
        self.model_len = self.n_states
        self.n_params = layout.n_params
        self.n_observables = layout.n_obs
        self.jacobian_bandwidth = manifest.jacobian_bandwidth

        # Solver classification
        if self.jacobian_bandwidth == -1:
            self.solver_type = "Matrix-Free GMRES"
        elif self.jacobian_bandwidth == 0:
            self.solver_type = "Sparse LU (Dense / Arrowhead Coupling)"
        else:
            self.solver_type = f"Banded LU (width={self.jacobian_bandwidth})"

        # Structural non-zero (NNZ) and sparsity analysis from CPR cache
        c_seeds, _, c_rows, c_cols, c_dense = manifest.cpr_cache if manifest.cpr_cache else ([], [], [], [], [])
        self.cpr_colors = len(c_seeds)
        self.dense_rows = len(c_dense)

        entries = set(zip(c_rows, c_cols))
        for r in c_dense:
            for c in range(self.n_states):
                entries.add((r, c))

        total_elements = self.n_states ** 2
        if entries:
            self.nnz = len(entries)
        elif self.n_states <= 1:
            self.nnz = self.n_states
        elif self.jacobian_bandwidth == -1:
            self.nnz = min(total_elements, self.n_states * 7)
        elif self.jacobian_bandwidth > 0:
            self.nnz = min(total_elements, self.n_states * (2 * self.jacobian_bandwidth + 1))
        else:
            self.nnz = total_elements

        self.sparsity = 1.0 - (self.nnz / total_elements) if total_elements > 0 else 0.0

        # Memory footprint calculation (BDF history [6] + Scratchpads [7] + Mesh + Parameters)
        state_bytes = self.n_states * 13 * 8
        mesh_bytes = layout.m_length * 8
        param_bytes = layout.p_length * 8
        self.memory_footprint_bytes = state_bytes + mesh_bytes + param_bytes

    def summary(self) -> Dict[str, Any]:
        """Returns structured dictionary of static model properties."""
        return {
            "n_states": self.n_states,
            "n_params": self.n_params,
            "n_observables": self.n_observables,
            "jacobian_bandwidth": self.jacobian_bandwidth,
            "solver_type": self.solver_type,
            "nnz": self.nnz,
            "sparsity": self.sparsity,
            "cpr_jvp_sweeps": self.cpr_colors,
            "cpr_vjp_sweeps": self.dense_rows,
            "memory_footprint_bytes": self.memory_footprint_bytes,
        }

    def __repr__(self) -> str:
        mem_kb = self.memory_footprint_bytes / 1024.0
        return (
            f"TelemetryReport(states={self.n_states}, params={self.n_params}, "
            f"nnz={self.nnz}, sparsity={self.sparsity:.1%}, "
            f"jvp_sweeps={self.cpr_colors}, vjp_sweeps={self.dense_rows}, "
            f"solver='{self.solver_type}', mem_est={mem_kb:.1f}KB)"
        )



