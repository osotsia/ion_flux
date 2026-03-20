import pytest
import numpy as np
import ion_flux as fx
from ion_flux.compiler.codegen import generate_cpp


class SimpleDAE(fx.PDE):
    """
    A coupled system containing an ODE and an algebraic constraint.
    y0_dot = y1
    0 = y1 - p0 * y0
    """
    y0 = fx.State()
    y1 = fx.State()
    p0 = fx.Parameter(default=2.0)
    
    def math(self):
        return {
            fx.dt(self.y0): self.y1,
            self.y1: self.p0 * self.y0
        }


def _evaluate_cpp_math(cpp_source: str, y: list, ydot: list, p: list) -> list:
    """
    Parses the emitted C++ strings and evaluates them natively in Python.
    Proves that the textual representation constructs the right mathematical graph.
    """
    lines = cpp_source.split("\n")
    res = [0.0] * len(y)
    
    # Very rudimentary parsing to extract equations
    for line in lines:
        line = line.strip()
        if line.startswith("res["):
            # e.g., "res[0] = (ydot[0]) - (y[1]);"
            idx_str = line[4:line.index("]")]
            idx = int(idx_str)
            
            eq_str = line[line.index("=") + 1 : -1].strip()
            
            # Python 'eval' sandbox to compute the actual value
            # Replace C++ syntax with Python syntax where necessary
            eq_str = eq_str.replace("std::pow", "pow")
            
            res[idx] = eval(eq_str, {"y": y, "ydot": ydot, "p": p, "pow": pow})
            
    return res


def test_codegen_emits_valid_dae_residual():
    model = SimpleDAE()
    
    # Alphabetical sorting determines the exact array indices
    states = ["y0", "y1"]
    params = ["p0"]
    
    cpp = generate_cpp(model.ast(), states, params)
    
    # Ensure Enzyme hooks and correct assignments are generated
    assert "extern void __enzyme_autodiff" in cpp
    assert "res[0] = (ydot[0]) - (y[1]);" in cpp
    assert "res[1] = (y[1]) - ((p[0] * y[0]));" in cpp


def test_numerical_jacobian_of_emitted_cpp():
    """
    Ensures that the emitted C++ equations map mathematically to the precise 
    Analytical Jacobian SUNDIALS expects: J = dF/dy + c_j * dF/dydot
    """
    model = SimpleDAE()
    states = ["y0", "y1"]
    params = ["p0"]
    cpp = generate_cpp(model.ast(), states, params)
    
    # State vectors
    y = [1.0, 2.0]
    ydot = [0.0, 0.0]
    p = [2.0]
    c_j = 0.5  # Arbitrary scaling factor applied to ydot by the solver
    
    eps = 1e-8
    N = len(y)
    J = np.zeros((N, N))
    
    # Base residual
    res_base = _evaluate_cpp_math(cpp, y, ydot, p)
    
    # Compute Finite Difference Jacobian using the generated C++ math logic
    for col in range(N):
        y_pert = list(y)
        ydot_pert = list(ydot)
        
        # Perturb y_k
        y_pert[col] += eps
        # Perturb ydot_k by c_j
        ydot_pert[col] += eps * c_j
        
        res_pert = _evaluate_cpp_math(cpp, y_pert, ydot_pert, p)
        
        for row in range(N):
            J[row, col] = (res_pert[row] - res_base[row]) / eps

    # Analytical Expectation for the Jacobian
    # res_0 = ydot_0 - y_1
    # res_1 = y_1 - p_0 * y_0
    
    # d(res_0)/d(y_0) = 0,   d(res_0)/d(ydot_0) = 1    => J[0,0] = 0 + c_j(1) = c_j
    # d(res_0)/d(y_1) = -1,  d(res_0)/d(ydot_1) = 0    => J[0,1] = -1 + c_j(0) = -1
    # d(res_1)/d(y_0) = -p0, d(res_1)/d(ydot_0) = 0    => J[1,0] = -2.0 + c_j(0) = -2.0
    # d(res_1)/d(y_1) = 1,   d(res_1)/d(ydot_1) = 0    => J[1,1] = 1 + c_j(0) = 1
    
    expected_J = np.array([
        [c_j,  -1.0],
        [-2.0,  1.0]
    ])
    
    np.testing.assert_allclose(J, expected_J, rtol=1e-5)
