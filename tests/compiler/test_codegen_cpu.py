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
    lines = cpp_source.split("\n")
    res = [0.0] * len(y)
    
    for line in lines:
        line = line.strip()
        if line.startswith("res["):
            idx_str = line[4:line.index("]")]
            idx = int(idx_str)
            eq_str = line[line.index("=") + 1 : -1].strip()
            eq_str = eq_str.replace("std::pow", "pow")
            res[idx] = eval(eq_str, {"y": y, "ydot": ydot, "p": p, "pow": pow})
            
    return res


def test_codegen_emits_valid_dae_residual():
    model = SimpleDAE()
    states = ["y0", "y1"]
    params = ["p0"]
    
    cpp = generate_cpp(model.ast(), states, params)
    
    # Assert Forward-Mode Enzyme hook and correct algebraic structure
    assert "extern void __enzyme_fwddiff" in cpp
    assert "res[0] = (ydot[0]) - (y[1]);" in cpp
    assert "res[1] = (y[1]) - ((p[0] * y[0]));" in cpp


def test_numerical_jacobian_of_emitted_cpp():
    model = SimpleDAE()
    states = ["y0", "y1"]
    params = ["p0"]
    cpp = generate_cpp(model.ast(), states, params)
    
    y = [1.0, 2.0]
    ydot = [0.0, 0.0]
    p = [2.0]
    c_j = 0.5  
    
    eps = 1e-8
    N = len(y)
    J = np.zeros((N, N))
    
    res_base = _evaluate_cpp_math(cpp, y, ydot, p)
    
    for col in range(N):
        y_pert = list(y)
        ydot_pert = list(ydot)
        
        y_pert[col] += eps
        ydot_pert[col] += eps * c_j
        
        res_pert = _evaluate_cpp_math(cpp, y_pert, ydot_pert, p)
        
        for row in range(N):
            J[row, col] = (res_pert[row] - res_base[row]) / eps

    expected_J = np.array([
        [c_j,  -1.0],
        [-2.0,  1.0]
    ])
    
    np.testing.assert_allclose(J, expected_J, rtol=1e-5)
