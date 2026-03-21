import pytest
import numpy as np
import ion_flux as fx
from ion_flux.compiler.memory import MemoryLayout
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
        return {fx.dt(self.y0): self.y1, self.y1: self.p0 * self.y0}


class SpatialPDE(fx.PDE):
    """1D System: dt(c) = grad(c). Tests loop unrolling and boundary overrides."""
    rod = fx.Domain(bounds=(0.0, 1.0), resolution=5)
    c = fx.State(domain=rod)
    def math(self):
        return {
            fx.dt(self.c): fx.grad(self.c),
            self.c.left: 0.0,
            self.c.right: 1.0
        }


def _evaluate_cpp_math(cpp_source: str, y: list, ydot: list, p: list) -> list:
    """A brittle but effective regex-like parser to test emitted scalar logic."""
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
    layout = MemoryLayout([model.y0, model.y1], [model.p0])
    cpp = generate_cpp(model.ast(), layout)
    
    assert "res[0] = (ydot[0 + CLAMP(0, 1)]) - (y[1 + CLAMP(0, 1)]);" in cpp
    assert "res[1] = (y[1 + CLAMP(0, 1)]) - ((p[0] * y[0 + CLAMP(0, 1)]));" in cpp

def test_codegen_emits_valid_spatial_loops_and_boundaries():
    model = SpatialPDE()
    layout = MemoryLayout([model.c], [])
    cpp = generate_cpp(model.ast(), layout)
    
    # Check that it generated a spatial loop for the bulk elements
    assert "for (int i = 0; i < 5; ++i) {" in cpp
    # Check that central difference macro-expansion worked exactly right
    assert "ydot[0 + CLAMP(i, 5)]) - (((y[0 + CLAMP((i)+1, 5)]) - (y[0 + CLAMP((i)-1, 5)])) / (2.0 * dx));" in cpp
    
    # Check that the boundary condition overrides were appended at the end
    assert "Boundary Condition Overrides" in cpp
    assert "res[0 + 0] = (y[0 + CLAMP(0, 5)]) - (0.0);" in cpp
    assert "res[0 + 4] = (y[0 + CLAMP(4, 5)]) - (1.0);" in cpp


def test_numerical_jacobian_of_emitted_dae_cpp():
    model = SimpleDAE()
    layout = MemoryLayout([model.y0, model.y1], [model.p0])
    cpp = generate_cpp(model.ast(), layout)
    
    y = [1.0, 2.0]
    ydot = [0.0, 0.0]
    p = [2.0]
    c_j = 0.5  
    eps = 1e-8
    N = len(y)
    
    # Inject CLAMP macro into the eval namespace
    eval_namespace = {"y": y, "ydot": ydot, "p": p, "CLAMP": lambda i, m: max(0, min(i, m-1))}
    
    res_base = [0.0] * N
    lines = [line.strip() for line in cpp.split("\n") if line.strip().startswith("res[")]
    for line in lines:
        idx = int(line[4:line.index("]")])
        res_base[idx] = eval(line[line.index("=") + 1 : -1], eval_namespace)
        
    J = np.zeros((N, N))
    for col in range(N):
        y_pert = list(y)
        ydot_pert = list(ydot)
        y_pert[col] += eps
        ydot_pert[col] += eps * c_j
        
        eval_namespace["y"] = y_pert
        eval_namespace["ydot"] = ydot_pert
        
        res_pert = [0.0] * N
        for line in lines:
            idx = int(line[4:line.index("]")])
            res_pert[idx] = eval(line[line.index("=") + 1 : -1], eval_namespace)
            
        for row in range(N):
            J[row, col] = (res_pert[row] - res_base[row]) / eps

    expected_J = np.array([
        [c_j,  -1.0],
        [-2.0,  1.0]
    ])
    np.testing.assert_allclose(J, expected_J, rtol=1e-5)