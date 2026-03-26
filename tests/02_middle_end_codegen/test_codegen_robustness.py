"""
test_codegen_robustness.py

Comprehensive test suite for the Middle-End Codegen pipeline.
Validates the translation of Python ASTs to native C++ residual loops using
AST Validation and Numerical Oracles (Method of Manufactured Solutions),
ensuring mathematical equivalence without brittle string matching.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

# --- Environment Checks ---
def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

# ==============================================================================
# Category 1: The Happy Path (Standard 1D & 0D Emission)
# ==============================================================================

class StandardModel(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    V = fx.State(domain=None, name="V") # 0D DAE
    
    def math(self):
        return {
            "regions": {
                self.x: [ fx.dt(self.c) == fx.grad(self.c) ]
            },
            "boundaries": [
                self.c.left == 1.0, 
                self.c.right == 0.0
            ],
            "global": [
                self.V == 4.2 - self.c.right
            ]
        }

def test_happy_path_ast_structure():
    """Validates the intermediate representation (AST) captures the correct intent."""
    model = StandardModel()
    ast = model.ast()
    
    # 1. Verify semantic bucketing
    assert "x" in ast["regions"]
    assert len(ast["boundaries"]) == 2
    assert len(ast["global"]) == 1
    
    # 2. Verify algebraic DAE structure (no spatial loops)
    v_eq = ast["global"][0]
    assert v_eq["lhs"]["name"] == "V"
    assert v_eq["rhs"]["type"] == "BinaryOp"
    assert v_eq["rhs"]["op"] == "sub"
    assert v_eq["rhs"]["right"]["type"] == "Boundary"

@REQUIRES_COMPILER
def test_happy_path_numerical_oracle():
    """Validates the compiled C++ evaluates the exact expected mathematical residual."""
    engine = Engine(model=StandardModel(), target="cpu", mock_execution=False)
    
    # Initialize blank arrays based on the engine's exact layout size
    N = engine.layout.n_states
    y = [0.0] * N
    ydot = [0.0] * N
    expected_res = [0.0] * N
    
    # Dynamically fetch the memory offsets
    off_c, size_c = engine.layout.state_offsets["c"]
    off_V, _ = engine.layout.state_offsets["V"]
    
    # 1. Inject known states dynamically
    # Resolution is 5, bounds 0 to 1 -> dx = 0.25. Linear profile c(x) = 1 - x
    y[off_c : off_c + size_c] = [1.0, 0.75, 0.5, 0.25, 0.0]
    
    # Expected V = 4.2 - c(right) = 4.2 - 0.0 = 4.2
    y[off_V] = 4.2 
    
    # 2. Evaluate the compiled C++
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # 3. Build the Math Oracle Expectations
    # Bulk PDE: res = ydot - grad(c). Since c is linear with slope -1, grad(c) = -1.0.
    # Therefore, res = 0.0 - (-1.0) = 1.0 for the internal nodes.
    expected_res[off_c] = 0.0      # c.left BC: 1.0 - 1.0 = 0.0
    expected_res[off_c + 1] = 1.0  # Bulk PDE node 1
    expected_res[off_c + 2] = 1.0  # Bulk PDE node 2
    expected_res[off_c + 3] = 1.0  # Bulk PDE node 3
    expected_res[off_c + 4] = 0.0  # c.right BC: 0.0 - 0.0 = 0.0
    expected_res[off_V] = 0.0      # V DAE: 4.2 - (4.2 - 0.0) = 0.0
    
    np.testing.assert_allclose(res, expected_res, err_msg="Mathematical mismatch in standard 1D emission.")


# ==============================================================================
# Category 2: Topological Edge Cases (Spherical Singularity & Nested Grids)
# ==============================================================================

class SphericalModel(fx.PDE):
    r = fx.Domain(bounds=(0, 5e-6), resolution=5, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    def math(self):
        return { "regions": { self.r: [ fx.dt(self.c) == fx.div(fx.grad(self.c)) ] } }

class MacroMicroModel(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=2, name="x")
    r = fx.Domain(bounds=(0, 1), resolution=3, name="r")
    macro_micro = x * r
    c = fx.State(domain=macro_micro, name="c")
    def math(self):
        return { "regions": { self.macro_micro: [ fx.dt(self.c) == fx.grad(self.c, axis=self.r) ] } }

@REQUIRES_COMPILER
def test_spherical_origin_lhopital_limit_numerical():
    """Validates the compiler safely handles the 0/0 singularity at the spherical origin."""
    engine = Engine(model=SphericalModel(), target="cpu", mock_execution=False)
    
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    ydot = [0.0] * 5
    
    # If L'Hopital's limit is missing, the divergence at r=0 evaluates to NaN.
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    assert not np.isnan(res[0]), "Spherical singularity at r=0 produced NaN. L'Hopital's limit failed."
    assert np.isfinite(res).all(), "Non-finite values detected in spherical evaluation."

@REQUIRES_COMPILER
def test_macro_micro_hierarchical_unrolling_numerical():
    """Validates 2D cross-product domains calculate flat array strides correctly."""
    engine = Engine(model=MacroMicroModel(), target="cpu", mock_execution=False)
    
    # 2 macro nodes * 3 micro nodes = 6 total states
    # Inject an identifiable state mapping: y[macro][micro]
    # Macro 0: [10, 20, 30]  (gradient is constant 10)
    # Macro 1: [100, 200, 300] (gradient is constant 100)
    y = [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    ydot = [0.0] * 6
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # Equation: dt(c) == grad(c_micro)
    # Expected Residual: ydot - grad(c_micro)
    # Note: Micro domain is bounds(0,1) with resolution 3 -> dx = 0.5
    # Macro 0 gradient: (20-10)/0.5 = 20.0
    # Macro 1 gradient: (200-100)/0.5 = 200.0
    # Centered diffs vary slightly at boundaries, but we focus on the inner nodes.
    
    inner_macro0 = res[1] 
    inner_macro1 = res[4]
    
    assert inner_macro0 == pytest.approx(-20.0), "Striding mismatch in Macro node 0."
    assert inner_macro1 == pytest.approx(-200.0), "Striding mismatch in Macro node 1."


# ==============================================================================
# Category 3: Moving Boundaries (ALE Advection Stability)
# ==============================================================================

class MovingMeshModel(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    L = fx.State(domain=None, name="L")
    
    def math(self):
        return {
            "regions": { self.x: [ fx.dt(self.c) == fx.grad(self.c) ] },
            "boundaries": [ self.x.right == self.L ], # Triggers ALE
            "global": [ fx.dt(self.L) == 1.0 ]
        }

@REQUIRES_COMPILER
def test_ale_advection_upwind_stability_numerical():
    """Validates ALE moving meshes apply stable directional upwinding rather than centered differences."""
    engine = Engine(model=MovingMeshModel(), target="cpu", mock_execution=False)
    
    # Evaluate with the domain EXPANDING (v > 0)
    # When expanding, upwinding should pull from the left node (backward difference)
    y_expand = [0.0, 10.0, 20.0, 30.0, 40.0, 1.0] # c[0..4], L
    ydot_expand = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # L_dot = 1.0 (expanding)
    
    res_expand = engine.evaluate_residual(y_expand, ydot_expand, parameters={})
    
    # Evaluate with the domain CONTRACTING (v < 0)
    # When contracting, upwinding should pull from the right node (forward difference)
    y_contract = [0.0, 10.0, 20.0, 30.0, 40.0, 1.0]
    ydot_contract = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0] # L_dot = -1.0 (contracting)
    
    res_contract = engine.evaluate_residual(y_contract, ydot_contract, parameters={})
    
    # Because our state has a constant gradient (10.0) but non-zero values, 
    # the advection velocity calculation (which scales by x_coord) will differ
    # at node 2 based on whether it pulls from node 1 (value 10) or node 3 (value 30).
    assert res_expand[2] != res_contract[2], "ALE Advection failed to shift stencil based on velocity direction."


# ==============================================================================
# Category 4: Unstructured CSR Traversal Emission
# ==============================================================================

def test_unstructured_csr_lambda_emission_numerical():
    """Validates the emitted C++ perfectly executes matrix-free CSR graph traversals."""
    # Mock a simple 3-node graph representing a triangle
    # Node 0 connects to Node 1 and 2
    mesh_data = {
        "row_ptr": [0, 2, 4, 6], 
        "col_ind": [1, 2, 0, 2, 0, 1], 
        "weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    }
    mesh = fx.Domain(bounds=(0,1), resolution=3, coord_sys="unstructured", name="mesh", csr_data=mesh_data)
    
    class UnstructuredModel(fx.PDE):
        c = fx.State(domain=mesh, name="c")
        def math(self):
            return { "regions": { mesh: [ fx.dt(self.c) == fx.div(fx.grad(self.c)) ] } }

    model = UnstructuredModel()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    if getattr(engine, "mock_execution", False):
        pytest.skip("Compilation environment absent.")
        
    # Inject known states
    y = [10.0, 0.0, 0.0]
    ydot = [0.0, 0.0, 0.0]
    p = engine._pack_parameters({})
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # Math Oracle Check for Node 0:
    # Divergence = sum( weight * (neighbor_val - center_val) )
    # Div(Node 0) = 0.5 * (y[1] - y[0]) + 0.5 * (y[2] - y[0])
    # Div(Node 0) = 0.5 * (0 - 10) + 0.5 * (0 - 10) = -10.0
    # res[0] = ydot[0] - Div(Node 0) = 0.0 - (-10.0) = 10.0
    
    assert res[0] == pytest.approx(10.0), "CSR Graph traversal calculated incorrect flux accumulation for Node 0."
    
    # Div(Node 1) = 0.5 * (y[0] - y[1]) + 0.5 * (y[2] - y[1])
    # Div(Node 1) = 0.5 * (10 - 0) + 0.5 * (0 - 0) = 5.0
    # res[1] = 0.0 - 5.0 = -5.0
    assert res[1] == pytest.approx(-5.0), "CSR Graph traversal calculated incorrect flux accumulation for Node 1."