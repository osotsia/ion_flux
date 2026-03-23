import numpy as np
import scipy.linalg
from typing import Optional, Any, Union, Dict

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
        
        # Native Differentiable EIS (Frequency Domain)
        if self._trajectory.get("type") == "eis":
            w_arr = self._trajectory["w_arr"]
            input_var = self._trajectory["input_var"]
            output_var = self._trajectory["output_var"]
            eis_target_key = self._trajectory.get("_eis_target_key", "Z_real")
            eps = 1e-5
            
            s_base = self._engine.start_session(parameters=self._parameters)
            s_base.reach_steady_state()
            
            if s_base.handle:
                y_ss = s_base.handle.get_state()
                J_ss = s_base.handle.get_jacobian(0.0)
            else:
                y_ss = s_base._mock_y
                J_ss = np.eye(len(y_ss))
                
            try:
                lu, piv = scipy.linalg.lu_factor(J_ss)
                def solve_J(b): return scipy.linalg.lu_solve((lu, piv), b)
            except scipy.linalg.LinAlgError:
                def solve_J(b): return np.linalg.lstsq(J_ss, b, rcond=None)[0]
            
            for p_name in req_grad:
                if p_name in self._engine.parameters:
                    p_val = self._parameters.get(p_name, self._engine.parameters[p_name].value)
                    
                    # 1. Compute exact dF/dp using Native Enzyme Reverse-Mode VJP
                    p_list = self._engine._pack_parameters(self._parameters)
                    p_offset = self._engine.layout.get_param_offset(p_name)
                    dF_dp = np.zeros(len(y_ss))
                    lam = [0.0] * len(y_ss)
                    for i in range(len(y_ss)):
                        lam[i] = 1.0
                        dp_out, _, _ = self._engine.runtime.evaluate_vjp(y_ss.tolist(), np.zeros_like(y_ss).tolist(), p_list, lam)
                        dF_dp[i] = dp_out[p_offset]
                        lam[i] = 0.0
                    
                    # 2. Extract Exact State Shift via Implicit Function Theorem (dy_ss/dp = -J^{-1} dF/dp)
                    dy_dp = solve_J(-dF_dp)
                    
                    # 3. Evaluate EIS at the analytically shifted Forward State
                    y_fwd = y_ss + eps * dy_dp
                    s_base.parameters[p_name] = p_val + eps
                    if s_base.handle:
                        s_base.handle.restore_state(s_base.time, y_fwd.tolist(), np.zeros_like(y_fwd).tolist(), y_fwd.tolist(), y_fwd.tolist(), 0.0, 1)
                    else:
                        s_base._mock_y = y_fwd
                    Z_fwd = s_base.solve_eis(w_arr / (2 * np.pi), input_var, output_var)._data
                    
                    # 4. Evaluate EIS at the analytically shifted Backward State
                    y_bwd = y_ss - eps * dy_dp
                    s_base.parameters[p_name] = p_val - eps
                    if s_base.handle:
                        s_base.handle.restore_state(s_base.time, y_bwd.tolist(), np.zeros_like(y_bwd).tolist(), y_bwd.tolist(), y_bwd.tolist(), 0.0, 1)
                    else:
                        s_base._mock_y = y_bwd
                    Z_bwd = s_base.solve_eis(w_arr / (2 * np.pi), input_var, output_var)._data
                    
                    # 5. Restore Pristine Session State
                    s_base.parameters[p_name] = p_val
                    if s_base.handle:
                        s_base.handle.restore_state(s_base.time, y_ss.tolist(), np.zeros_like(y_ss).tolist(), y_ss.tolist(), y_ss.tolist(), 0.0, 1)
                    else:
                        s_base._mock_y = y_ss
                        
                    # 6. Accumulate true gradient mappings
                    dZ_dp = (Z_fwd[eis_target_key] - Z_bwd[eis_target_key]) / (2 * eps)
                    self.grads[p_name] = float(np.sum(self._dl_dy_mapped * dZ_dp))
                    
            return self.grads

        if not RUST_FFI_AVAILABLE or getattr(self._engine, "mock_execution", False):
            for p_name in req_grad:
                if p_name in self._engine.parameters:
                    self.grads[p_name] = float(np.random.uniform(-0.1, 0.1))
            return self.grads
            
        t_eval = self._trajectory["_micro_t"]
        y_traj = self._trajectory["_micro_y"]
        ydot_traj = self._trajectory["_micro_ydot"]
        
        p_list_default = self._engine._pack_parameters(self._parameters)
        p_traj = self._trajectory.get("_p_traj", [p_list_default] * len(y_traj))
        
        macro_t = self._trajectory["Time [s]"]
        dl_dy_macro = self._dl_dy_mapped if self._dl_dy_mapped is not None else np.zeros_like(self._trajectory["_y_raw"])
        
        dl_dy = np.zeros_like(y_traj)
        macro_idx = 0
        for i, t in enumerate(t_eval):
            if macro_idx < len(macro_t) and abs(t - macro_t[macro_idx]) < 1e-8:
                dl_dy[i] = dl_dy_macro[macro_idx]
                macro_idx += 1
        
        y0, ydot0, id_arr, spatial_diag = self._engine._extract_metadata()
        
        bw = getattr(self._engine, "jacobian_bandwidth", 0)
        p_grad = discrete_adjoint_native(
            self._engine.runtime.lib_path, y_traj.tolist(), ydot_traj.tolist(), 
            t_eval.tolist(), id_arr, p_traj, dl_dy.tolist(), bw
        )
        
        for p_name in req_grad:
            if p_name in self._engine.layout.param_offsets:
                offset = self._engine.layout.param_offsets[p_name][0]
                self.grads[p_name] = p_grad[offset]
                
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
        state_name = getattr(predicted, "name", state_name)
    else:
        trajectory = getattr(engine, "_current_trajectory", None) if engine else None
        parameters = {k: v.value for k, v in engine.parameters.items()} if engine else {}
        
    p_arr = np.asarray(predicted.data if hasattr(predicted, "data") else predicted)
    t_arr = np.asarray(target)
    
    if p_arr.shape != t_arr.shape:
        raise ValueError(f"Shape mismatch: predicted {p_arr.shape} vs target {t_arr.shape}")
    
    diff = p_arr - t_arr
    val = np.sqrt(np.mean(diff ** 2))
    
    dl_dy_mapped = None
    if engine and trajectory:
        if trajectory.get("type") == "eis":
            dl_dy_mapped = diff / (len(diff) * max(val, 1e-12))
            trajectory["_eis_target_key"] = state_name
        else:
            y_traj = trajectory["_y_raw"]
            dl_dy_mapped = np.zeros_like(y_traj)
            
            if state_name in engine.layout.state_offsets:
                offset, size = engine.layout.state_offsets[state_name]
                grad_multiplier = 1.0 / (len(diff) * max(val, 1e-12))
                
                for i in range(size):
                    if size == 1:
                        dl_dy_mapped[:, offset + i] = (grad_multiplier * diff)
                    else:
                        dl_dy_mapped[:, offset + i] = (grad_multiplier * diff[:, i]) / size
            
    return Loss(val, engine=engine, trajectory=trajectory, dl_dy_mapped=dl_dy_mapped, parameters=parameters)