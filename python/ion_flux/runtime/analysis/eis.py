import numpy as np
import scipy.linalg
from typing import Any

def solve_eis(session, frequencies: np.ndarray, input_var: str, output_var: str) -> Any:
    """
    Extracts the analytical Jacobian directly from the Ahead-of-Time Enzyme module 
    and algebraically solves the transfer function for an Electrochemical Impedance Spectrum.
    """
    from ion_flux.runtime.results import SimulationResult
    
    w_arr = np.asarray(frequencies) * 2 * np.pi
    
    if not session.handle:
        from ion_flux.runtime.backends.mock import mock_eis_impedance
        Z = mock_eis_impedance(w_arr)
    else:
        N = session.manifest.layout.n_states
        
        y = session.handle.get_state()
        ydot = np.zeros_like(y)
        
        M, J_steady, dF_dp_input = session.manifest.extract_eis_system(
            y.tolist(), ydot.tolist(), session.parameters, input_var
        )
        B = -dF_dp_input
        out_offset = session.manifest.layout.get_state_offset(output_var)
        C = np.zeros(N)
        C[out_offset] = 1.0 
        
        Z = np.zeros_like(w_arr, dtype=np.complex128)
        
        for i, w in enumerate(w_arr):
            A = 1j * w * M + J_steady
            try:
                # Solve the complex linear system algebraically: (j*w*M + J) * X = B
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