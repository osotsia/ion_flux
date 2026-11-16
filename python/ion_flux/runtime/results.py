import numpy as np
from typing import Dict, Any, List, Optional

class Variable:
    """Wrapper mapping flat FFI arrays back into intuitive multidimensional structures."""
    __slots__ = ["data", "result", "name"]
    def __init__(self, data: np.ndarray, result: Optional[Any] = None, name: str = ""): 
        self.data = data
        self.result = result
        self.name = name
    def __repr__(self) -> str: 
        return f"<Variable: {self.name} shape={self.data.shape}>"

class SimulationResult:
    """Data object returning exact multi-dimensional trajectories outputted by the Engine."""
    __slots__ = ["_data", "parameters", "status", "engine", "trajectory", "diagnostics"]
    
    def __init__(self, data: Dict[str, np.ndarray], parameters: Dict[str, float], status: str = "completed", engine: Optional[Any] = None, trajectory: Optional[Dict] = None, diagnostics: Optional[Dict[str, Any]] = None):
        self._data = data
        self.parameters = parameters
        self.status = status
        self.engine = engine
        self.trajectory = trajectory
        self.diagnostics = diagnostics or {}

    def __getitem__(self, key: str) -> Variable:
        if key not in self._data: raise KeyError(f"Variable '{key}' not found.")
        return Variable(self._data[key], result=self, name=key)
        
    def to_dict(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Provides native JSON-serialization compatibility for Cloud workflows."""
        keys = variables or self._data.keys()
        return {k: self._data[k].tolist() for k in keys if k in self._data}

    def print_summary(self) -> None:
        """Prints a structured execution and diagnostic summary."""
        print("=" * 64)
        print("ION_FLUX EXECUTION & DIAGNOSTIC SUMMARY")
        print("=" * 64)
        if self.engine and hasattr(self.engine, "telemetry") and self.engine.telemetry:
            t = self.engine.telemetry
            print("Model Profile:")
            print(f"  States / Params / Obs:   {t.n_states} / {t.n_params} / {t.n_observables}")
            print(f"  Jacobian NNZ:            {t.nnz} ({t.sparsity:.1%} sparse)")
            print(f"  Solver Configuration:    {t.solver_type}")
            print(f"  CPR AD Sweeps:           {t.cpr_colors} JVP (Fwd), {t.dense_rows} VJP (Rev)")
            print(f"  Static Memory Est:       {t.memory_footprint_bytes / 1024.0:.1f} KB")
            print("-" * 64)
        if self.diagnostics:
            d = self.diagnostics
            print("Solver Dynamics:")
            total = d.get('total_steps', 0)
            accepted = d.get('accepted_steps', 0)
            rejected = d.get('rejected_steps', 0)
            print(f"  Steps:                   {total} total ({accepted} accepted, {rejected} rejected)")
            print(f"  Newton Iterations:       {d.get('newton_iterations', 0)} ({d.get('newton_iterations_per_step', 0.0):.2f} / step)")
            print(f"  LU Factorizations:       {d.get('numeric_factorizations', 0)} numeric, {d.get('symbolic_factorizations', 0)} symbolic")
            print("Time Breakdown (us):")
            print(f"  Residual Evaluations:    {d.get('residual_time_us', 0):,} us")
            print(f"  Jacobian Assembly:       {d.get('jacobian_assembly_time_us', 0):,} us")
            print(f"  Linear Solve:            {d.get('linear_solve_time_us', 0):,} us")
            rss_mb = d.get('peak_rss_bytes', 0) / (1024.0 * 1024.0)
            if rss_mb > 0:
                print(f"Memory & Host Metrics:")
                print(f"  Peak Resident Memory:    {rss_mb:.2f} MB")
            cycles = d.get('cpu_cycles', 0)
            instr = d.get('instructions', 0)
            if cycles > 0 and instr > 0:
                ipc = instr / cycles if cycles > 0 else 0.0
                print(f"  Hardware Counters (PMU):")
                print(f"    CPU Cycles:            {cycles:,}")
                print(f"    Instructions:          {instr:,} (IPC: {ipc:.2f})")
        print("=" * 64)


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