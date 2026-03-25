"""
Middle-End Codegen: Pipeline Diagnostics

Targeted diagnostics for the Codegen pipeline, isolating finite difference
stencil deformations, boundary condition erasures, and DAE initialization.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE

# ==============================================================================
# Diagnostic Models (Minimal Reprex for DFN Complexities)
# ==============================================================================

class StencilDiagnosticPDE(fx.PDE):
    """
    Isolates Problem 1: Finite Difference Stencil Deformation.
    A simple 1D domain where c(x) = x. The gradient must be exactly 1.0 everywhere.
    """
    x = fx.Domain(bounds=(0.0, 4.0), resolution=5, name="x") # dx = 1.0
    c = fx.State(domain=x, name="c")
    
    def math(self):
        return {
            "regions": {
                # dt(c) = grad(c) -> If c=x, then grad(c)=1.0, so dt(c) should evaluate to 1.0
                self.x: [ fx.dt(self.c) == fx.grad(self.c) ]
            },
            "boundaries": [
                # No boundaries applied to keep the pure PDE stencil exposed
            ]
        }

class InterfaceDiagnosticPDE(fx.PDE):
    """
    Isolates Problem 2: Boundary Condition Erasure (Dirichlet vs Neumann Collision).
    Mimics the DFN Negative-Electrode to Separator boundary.
    """
    x_n = fx.Domain(bounds=(0, 1), resolution=3, name="x_n")
    x_s = fx.Domain(bounds=(1, 2), resolution=3, name="x_s")
    
    c_n = fx.State(domain=x_n, name="c_n")
    c_s = fx.State(domain=x_s, name="c_s")
    
    def math(self):
        flux_n = -fx.grad(self.c_n)
        flux_s = -fx.grad(self.c_s)
        
        return {
            "regions": {
                self.x_n: [ fx.dt(self.c_n) == -fx.div(flux_n) ],
                self.x_s: [ fx.dt(self.c_s) == -fx.div(flux_s) ]
            },
            "boundaries": [
                # The critical 1D-1D interface definition:
                self.c_n.right == self.c_s.left,   # 1. State Continuity (Dirichlet)
                flux_n.right == flux_s.left,       # 2. Flux Continuity (Neumann)
                
                self.c_n.left == 0.0,
                self.c_s.right == 0.0
            ]
        }

class AlgebraicMaskingPDE(fx.PDE):
    """
    Isolates Problem 3 & 4: Over-Constrained DAEs & Initialization.
    Mimics DFN charge conservation: 0 == div(i_e) - j
    """
    x = fx.Domain(bounds=(0, 1), resolution=4, name="x")
    phi = fx.State(domain=x, name="phi")
    j_flux = fx.Parameter(default=1.0)
    
    def math(self):
        i_e = -fx.grad(self.phi)
        return {
            "regions": {
                # Purely algebraic bulk equation (no fx.dt)
                self.x: [ 0 == fx.div(i_e) - self.j_flux ]
            },
            "boundaries": [
                self.phi.left == 0.0,
                i_e.right == 0.0
            ],
            "global": [
                # Intentional bad initial guess to test initialization handling
                self.phi.t0 == 100.0 
            ]
        }

# ==============================================================================
# Diagnostic Test Suite
# ==============================================================================

def test_diagnose_boundary_stencil_deformation():
    """
    X-Ray for the `CLAMP` macro issue. 
    If a linear profile c(x) = x is applied, the numerical gradient must be 
    exactly constant (1.0) across all nodes, including the boundaries.
    """
    model = StencilDiagnosticPDE()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    # 1. AST string inspection
    cpp = engine.cpp_source
    # We expect boundary nodes to use forward/backward differences (dx), not centered (2*dx)
    # The current codegen strictly emits `2.0 * dx_x` everywhere.
    assert "2.0 * dx_x" in cpp, "Verify baseline: builder currently emits centered differences."
    
    if getattr(engine, "mock_execution", False):
        pytest.skip("Compilation environment absent.")
        
    # 2. Numerical Residual Inspection
    # State length = 5. Coordinates = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_linear = [0.0, 1.0, 2.0, 3.0, 4.0] 
    ydot_zero = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Evaluate: res = ydot - grad(c) -> res = 0.0 - 1.0 = -1.0 everywhere
    res = engine.evaluate_residual(y_linear, ydot_zero, parameters={})
    
    # DIAGNOSTIC ASSERTION:
    # If the CLAMP macro is active, res[0] will be (y[1]-y[0]) / (2*dx) = 1.0 / 2.0 = 0.5.
    # We assert it should be 1.0. This test will FAIL until the stencil logic is fixed.
    expected_res = [-1.0, -1.0, -1.0, -1.0, -1.0]
    
    np.testing.assert_allclose(
        res, expected_res, 
        err_msg="Boundary stencils are deformed. The CLAMP macro halves the true gradient at edges."
    )


def test_diagnose_interface_boundary_erasure():
    """
    X-Ray for Dirichlet/Neumann AST collision.
    Verifies that state continuity does not mathematically overwrite flux continuity.
    """
    model = InterfaceDiagnosticPDE()
    engine = Engine(model=model, target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    # 1. String inspection: Check if the codegen actually emitted both constraints
    cpp = engine.cpp_source
    # State continuity: c_n.right == c_s.left (offset_cn + 2 == offset_cs + 0)
    has_dirichlet = "y[0 + 2]) - (y[3 + 0]" in cpp or "y[3 + 0]) - (y[0 + 2]" in cpp
    # Flux continuity: grad(c_n).right == grad(c_s).left
    # This involves a complex expression with dx. We just look for the Neumann injection.
    # Currently, `builder.py` overwrites the residual index directly.
    
    if getattr(engine, "mock_execution", False):
        pytest.skip("Compilation environment absent.")
        
    # 2. Jacobian Rank Inspection
    N = engine.layout.n_states # 6 states (3 for c_n, 3 for c_s)
    y_rand = np.random.uniform(0.1, 1.0, size=N).tolist()
    ydot_zero = np.zeros(N).tolist()
    
    # Evaluate Jacobian
    J = np.array(engine.evaluate_jacobian(y_rand, ydot_zero, c_j=1.0))
    
    # DIAGNOSTIC ASSERTION:
    # If the boundary Dirichlet equation overwrote the Neumann equation, we lost a physical constraint.
    # This will result in an undefined node, causing the Jacobian to be rank-deficient.
    rank = np.linalg.matrix_rank(J)
    
    assert rank == N, (
        f"Singular Jacobian (Rank {rank} < {N}). "
        f"The compiler erased a boundary condition at the domain interface."
    )


def test_diagnose_algebraic_dae_masking_and_initialization():
    """
    X-Ray for `id_arr` overlaps and uninitialized algebraic roots.
    """
    model = AlgebraicMaskingPDE()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    # 1. Verify differential vs algebraic masking
    y0, ydot0, id_arr, _ = engine._extract_metadata()
    
    # phi is purely algebraic, so id_arr MUST be all zeros.
    assert sum(id_arr) == 0.0, "Algebraic variable incorrectly flagged as differential."
    
    if getattr(engine, "mock_execution", False):
        pytest.skip("Compilation environment absent.")
        
    # 2. Check for initial residual explosion
    # The user provided a bad guess (phi = 100.0 everywhere).
    # If we pass this directly to the solver without a root-finding initialization,
    # it might crash or cause massive initial Newton steps.
    res_initial = engine.evaluate_residual(y0, ydot0, parameters={"j_flux": 1.0})
    
    # Evaluate rank to ensure Dirichlet boundary and algebraic bulk didn't over-constrain the same node
    J = np.array(engine.evaluate_jacobian(y0, ydot0, c_j=1.0))
    rank = np.linalg.matrix_rank(J)
    
    assert rank == engine.layout.n_states, (
        "Algebraic masking overlapping with Dirichlet boundaries caused a singular matrix."
    )