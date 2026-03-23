import pytest
import numpy as np
import ion_flux as fx
from ion_flux.compiler.memory import MemoryLayout
from ion_flux.compiler.codegen import generate_cpp

class SimpleDAE(fx.PDE):
    """0D System: y0_dot = y1, y1 = p0 * y0"""
    y0 = fx.State()
    y1 = fx.State()
    p0 = fx.Parameter(default=2.0)
    
    def math(self):
        return {
            "global": [ 
                fx.dt(self.y0) == self.y1,
                self.y1 == self.p0 * self.y0 
            ]
        }

class SpatialPDE(fx.PDE):
    """1D System: dt(c) = grad(c). Tests loop unrolling and boundary overrides."""
    rod = fx.Domain(bounds=(0.0, 1.0), resolution=5, name="rod")
    c = fx.State(domain=rod, name="c")
    
    def math(self):
        return {
            "regions": {
                self.rod: [ fx.dt(self.c) == fx.grad(self.c) ]
            },
            "boundaries": [
                self.c.left == 0.0,
                self.c.right == 1.0
            ]
        }

def test_codegen_emits_valid_dae_residual():
    model = SimpleDAE()
    states, params = [model.y0, model.y1], [model.p0]
    layout = MemoryLayout(states, params)
    cpp = generate_cpp(model.ast(), layout, states)
    
    assert "res[0 + 0] = (ydot[0 + CLAMP(0, 1)]) - (y[1 + CLAMP(0, 1)]);" in cpp
    assert "res[1 + 0] = (y[1 + CLAMP(0, 1)]) - ((p[0] * y[0 + CLAMP(0, 1)]));" in cpp

def test_codegen_emits_valid_spatial_loops_and_boundaries():
    model = SpatialPDE()
    states = [model.c]
    layout = MemoryLayout(states, [])
    cpp = generate_cpp(model.ast(), layout, states)
    
    # Verify spatial resolution parsed correctly from domain bounds (dx = 1.0 / 4)
    assert "double dx_rod = 0.25;" in cpp
    
    # Check spatial loops and dynamic DX injection 
    # Match the exact line emitted by the string builder in codegen.py
    assert "for (int i = 0; i < 5; ++i) {" in cpp
    assert "res[0 + i] = (ydot[0 + CLAMP(i, 5)]) - (((y[0 + CLAMP((i) + 1, 5)]) - (y[0 + CLAMP((i) - 1, 5)])) / (2.0 * dx_rod));" in cpp
    
    # Check that the boundary condition overrides were appended at the end
    assert "Boundary Condition Overrides" in cpp
    assert "res[0 + 0] = (y[0 + CLAMP(0, 5)]) - (0.0);" in cpp
    assert "res[0 + 4] = (y[0 + CLAMP(4, 5)]) - (1.0);" in cpp

def test_numerical_jacobian_of_emitted_dae_cpp():
    model = SimpleDAE()
    states, params = [model.y0, model.y1], [model.p0]
    layout = MemoryLayout(states, params)
    cpp = generate_cpp(model.ast(), layout, states)
    
    y = [1.0, 2.0]
    ydot = [0.0, 0.0]
    p = [2.0]
    c_j = 0.5  
    eps = 1e-8
    N = len(y)
    
    # Inject CLAMP macro into the eval namespace to mimic C++ macro expansion
    eval_namespace = {
        "y": y, 
        "ydot": ydot, 
        "p": p, 
        "CLAMP": lambda i, m: max(0, min(i, m-1))
    }
    
    res_base = [0.0] * N
    # Extract only the explicit assignment lines (e.g. `res[0] = ...`)
    lines = [line.strip() for line in cpp.split("\n") if line.strip().startswith("res[")]
    
    for line in lines:
        idx_str = line[4:line.index("]")]
        idx = eval(idx_str)
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
            idx_str = line[4:line.index("]")]
            idx = eval(idx_str)
            res_pert[idx] = eval(line[line.index("=") + 1 : -1], eval_namespace)
            
        for row in range(N):
            J[row, col] = (res_pert[row] - res_base[row]) / eps

    # The expected analytical Jacobian for:
    # res[0] = ydot0 - y1
    # res[1] = y1 - p0*y0
    expected_J = np.array([
        [c_j,  -1.0],
        [-2.0,  1.0]
    ])
    
    np.testing.assert_allclose(J, expected_J, rtol=1e-5)