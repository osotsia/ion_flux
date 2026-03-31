import numpy as np
import scipy.linalg
from typing import Any

def solve_eis(session, frequencies: np.ndarray, input_var: str, output_var: str) -> Any:
    """Extracts the analytical Jacobian directly from Enzyme and algebraically solves the transfer function."""
    from ion_flux.runtime.engine import SimulationResult
    
    w_arr = np.asarray(frequencies) * 2 * np.pi
    
    if not session.handle:
        # Mock execution fallback
        Z = 0.05 + (0.1 / (1 + 1j * w_arr * 0.1)) + (0.01 / np.sqrt(1j * w_arr))
    else:
        N = session.engine.layout.n_states
        J_steady = session.handle.get_jacobian(0.0)
        
        y = session.handle.get_state()
        ydot = np.zeros_like(y)
        
        # Pack both standard mathematical parameters and structural mesh constants
        p_list = session.engine._pack_parameters(session.parameters)
        m_list = session.engine.layout.get_mesh_data()
        
        offset = session.engine.layout.get_param_offset(input_var)
        
        dF_dp_input = np.zeros(N)
        lam = [0.0] * N
        for i in range(N):
            lam[i] = 1.0
            # Evaluate Vector-Jacobian Product (VJP) passing the explicit mesh (m_list) safely
            dp_out, _, _ = session.engine.runtime.evaluate_vjp(
                y.tolist(), ydot.tolist(), p_list, m_list, lam
            )
            dF_dp_input[i] = dp_out[offset]
            lam[i] = 0.0
        
        B = -dF_dp_input
        out_offset = session.engine.layout.get_state_offset(output_var)
        C = np.zeros(N)
        C[out_offset] = 1.0 
        
        Z = np.zeros_like(w_arr, dtype=np.complex128)
        M = np.diag(session.id_arr)
        
        for i, w in enumerate(w_arr):
            A = 1j * w * M + J_steady
            try:
                # Solve the complex linear system (j*w*M + J) * X = B
                X = scipy.linalg.solve(A, B)
                Z[i] = np.dot(C, X)
            except scipy.linalg.LinAlgError:
                Z[i] = np.nan
                
    data = {"Z_real": Z.real, "Z_imag": Z.imag, "Frequencies": frequencies}
    trajectory = {
        "type": "eis", "w_arr": w_arr, 
        "input_var": input_var, "output_var": output_var
    }
    
    return SimulationResult(data, session.parameters, engine=session.engine, trajectory=trajectory)