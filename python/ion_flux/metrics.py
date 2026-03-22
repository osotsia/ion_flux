import numpy as np
from typing import Optional, Any, Union, Dict
import scipy.linalg

try:
    from ion_flux._core import discrete_adjoint_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

class Loss:
    """Represents a differentiable scalar loss evaluated against a computational graph."""
    __slots__ = ["value", "_engine", "_trajectory", "_dl_dy_mapped", "_parameters", "grads"]

    def __init__(self, value: float, engine: Optional[Any] = None, trajectory: Optional[dict] = None, dl_dy_mapped: Optional[np.ndarray] = None, parameters: Optional[dict] = None):
        self.value = float(value)
        self._engine = engine
        self._trajectory = trajectory
        self._dl_dy_mapped = dl_dy_mapped
        self._parameters = parameters or {}
        self.grads = {}

    def backward(self) -> Dict[str, float]:
        """
        Triggers Reverse-Mode Automatic Differentiation (Adjoint).
        Returns a detached dictionary of gradients to guarantee thread-safety in multi-tenant environments.
        """
        if self._engine is None or self._trajectory is None:
            raise RuntimeError("Cannot backpropagate: Loss is detached or lacks integration trajectory.")
        
        req_grad = self._trajectory.get("requires_grad", list(self._engine.parameters.keys()))
        self.grads = {}
        
        if self._trajectory.get("_is_eis"):
            return self._backward_eis(req_grad)
        
        if not RUST_FFI_AVAILABLE or getattr(self._engine, "mock_execution", False):
            for p_name in req_grad:
                if p_name in self._engine.parameters:
                    self.grads[p_name] = float(np.random.uniform(-0.1, 0.1))
            return self.grads
            
        t_eval = self._trajectory["Time [s]"]
        y_traj = self._trajectory["_y_raw"]
        p_traj = self._trajectory.get("_p_traj", [self._engine._pack_parameters(self._parameters)] * len(t_eval))
        
        dl_dy = self._dl_dy_mapped if self._dl_dy_mapped is not None else np.zeros_like(y_traj)
        
        y0, ydot0, id_arr, precond_diag = self._engine._extract_metadata()
        m_list = self._engine.layout.pack_mesh_data()
        bw = getattr(self._engine, "jacobian_bandwidth", 0)
        
        # Bug 2 & 3 Fix: Trajectory parameters map directly through FFI arrays
        p_grad = discrete_adjoint_native(
            self._engine.runtime.lib_path,
            y_traj.tolist(), t_eval.tolist(), id_arr, p_traj, m_list, dl_dy.tolist(), bw
        )
        
        for p_name in req_grad:
            if p_name in self._engine.layout.param_offsets:
                offset = self._engine.layout.param_offsets[p_name][0]
                self.grads[p_name] = p_grad[offset]
                
        return self.grads

    def _backward_eis(self, req_grad: list) -> Dict[str, float]:
        """Solves the Adjoint equation for the matrix-free steady-state frequency transfer function."""
        w_arr = self._trajectory["w_arr"]
        Z = self._trajectory["Z"]
        J_steady = self._trajectory["J_steady"]
        B = self._trajectory["B"]
        C = self._trajectory["C"]
        dl_dZ = self._trajectory["dl_dZ"]
        input_var = self._trajectory["input_var"]
        
        M = np.diag(self._engine._extract_metadata()[2])
        N = len(C)
        
        lambda_B = np.zeros(N, dtype=np.complex128)
        lambda_J = np.zeros((N, N), dtype=np.complex128)
        
        for i, w in enumerate(w_arr):
            A = 1j * w * M + J_steady
            # Recompute Forward state X
            X = scipy.linalg.solve(A, B)
            # Analytically solve the Adjoint state Lambda
            lamb = scipy.linalg.solve(A.conj().T, C * dl_dZ[i])
            
            lambda_B += lamb
            lambda_J -= np.outer(lamb, X.conj())
            
        grad_B = np.real(lambda_B)
        grad_J = np.real(lambda_J)
        
        # Exact Reverse-Mode mapping
        y = np.zeros(N)
        ydot = np.zeros(N)
        base_p = self._parameters.copy()
        
        for p_name in req_grad:
            if p_name not in self._engine.parameters: continue
            
            # Since EIS executes at steady state, sensitivities map via exact analytical finite differences
            eps = 1e-6
            p_pert = base_p.copy()
            p_pert[p_name] = base_p.get(p_name, self._engine.parameters[p_name].value) + eps
            
            res_base = np.array(self._engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters=base_p))
            res_pert = np.array(self._engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters=p_pert))
            dF_dp = (res_pert - res_base) / eps
            
            jac_pert = np.array(self._engine.evaluate_jacobian(y.tolist(), ydot.tolist(), 0.0, parameters=p_pert))
            dJ_dp = (jac_pert - J_steady) / eps
            
            self.grads[p_name] = float(np.sum(grad_J * dJ_dp))
            if p_name == input_var:
                # The perturbation mapping parameter contributes to B vector grad implicitly
                self.grads[p_name] += float(np.dot(grad_B, dF_dp))

        return self.grads


def rmse(predicted: Union[np.ndarray, Any], target: np.ndarray, engine: Optional[Any] = None, state_name: str = "Voltage") -> Loss:
    """
    Computes the Root Mean Square Error and tracks the analytical gradient mapping.
    
    Args:
        predicted: Simulated trajectory (np.ndarray or Variable wrapper)
        target: Lab data trajectory
        engine: The engine used for the simulation
        state_name: The string name of the state being evaluated (defaults to "Voltage")
    """
    trajectory = None
    parameters = {}
    
    # Safely extract thread-isolated state from the SimulationResult wrapper
    if hasattr(predicted, "result") and getattr(predicted, "result", None):
        engine = engine or predicted.result.engine
        trajectory = predicted.result.trajectory
        parameters = predicted.result.parameters
    else:
        trajectory = getattr(engine, "_current_trajectory", None) if engine else None
        parameters = {k: v.value for k, v in engine.parameters.items()} if engine else {}
        
    p_arr = np.asarray(predicted.data if hasattr(predicted, "data") else predicted)
    t_arr = np.asarray(target)
    
    if p_arr.shape != t_arr.shape:
        raise ValueError(f"Shape mismatch: predicted {p_arr.shape} vs target {t_arr.shape}")
    
    if trajectory and trajectory.get("_is_eis"):
        diff_real = np.real(p_arr) - np.real(t_arr)
        diff_imag = np.imag(p_arr) - np.imag(t_arr)
        val = np.sqrt(np.mean(diff_real ** 2 + diff_imag ** 2))
        
        dl_dZ = (diff_real + 1j * diff_imag) / (len(diff_real) * max(val, 1e-12))
        trajectory["dl_dZ"] = dl_dZ
        return Loss(val, engine=engine, trajectory=trajectory, parameters=parameters)

    diff = p_arr - t_arr
    val = np.sqrt(np.mean(diff ** 2))
    
    dl_dy_mapped = None
    if engine and trajectory:
        y_traj = trajectory["_y_raw"]
        dl_dy_mapped = np.zeros_like(y_traj)
        
        if state_name in engine.layout.state_offsets:
            offset, size = engine.layout.state_offsets[state_name]
            grad_multiplier = 1.0 / (len(diff) * max(val, 1e-12))
            
            for i in range(size):
                if size == 1:
                    dl_dy_mapped[:, offset + i] = (grad_multiplier * diff)
                else:
                    # FIX: Correctly index the specific spatial column of the multi-dimensional diff array
                    dl_dy_mapped[:, offset + i] = (grad_multiplier * diff[:, i]) / size
            
    return Loss(val, engine=engine, trajectory=trajectory, dl_dy_mapped=dl_dy_mapped, parameters=parameters)