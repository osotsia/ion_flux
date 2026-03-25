"""
Middle-End Codegen: Spatial Discretization

Isolates and validates bandwidth truncation handling in hierarchical (macro-micro)
meshes to prevent silent Jacobian truncation in the native solver.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE

class HierarchicalCouplingPDE(fx.PDE):
    """
    Isolates Bandwidth Truncation in Hierarchical (Macro-Micro) meshes.
    No 0D scalars or integrals are used, which tricks the Engine into 
    assuming a standard 1D Tridiagonal Bandwidth (bw=2) is sufficient.
    """
    x = fx.Domain(bounds=(0, 1), resolution=3, name="x")
    r = fx.Domain(bounds=(0, 1), resolution=10, name="r") # 10 micro nodes per macro node
    macro_micro = x * r
    
    # Alphabetical sorting puts c_e at offset 0, and c_s at offset 3
    c_e = fx.State(domain=x, name="c_e")
    c_s = fx.State(domain=macro_micro, name="c_s") 
    
    def math(self):
        # A mock reaction coupling the macro electrolyte to the micro particle surface
        # For macro node `i`, it reads c_s at micro node 9.
        # Flat memory distance: c_s[i*10 + 9] is deeply separated from c_e[i].
        c_surf = self.c_s.boundary("right", domain=self.r)
        reaction = c_surf - self.c_e
        
        return {
            "regions": {
                self.x: [ fx.dt(self.c_e) == fx.grad(self.c_e) + reaction ],
                self.macro_micro: [ fx.dt(self.c_s) == fx.grad(self.c_s, axis=self.r) ]
            },
            "boundaries": [
                self.c_e.left == 0.0, self.c_e.right == 0.0,
                self.c_s.boundary("left", domain=self.r) == 0.0,
                self.c_s.boundary("right", domain=self.r) == 0.0
            ]
        }

@pytest.mark.xfail(reason="Truncation issue.")
def test_diagnose_nonlinear_bandwidth_truncation():
    """
    X-Ray for silent Jacobian truncation. 
    Proves that hardcoding `jacobian_bandwidth = 2` for spatial systems 
    destroys the physical coupling between macro and micro states.
    """
    model = HierarchicalCouplingPDE()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    # 1. Verify the Engine's naive heuristic failed to allocate enough bandwidth
    # The true bandwidth required to link c_e[i] with c_s[i*10 + 9] is at least 10 + offsets.
    assert engine.jacobian_bandwidth == 2, "Baseline check: Engine incorrectly assumed a narrow tridiagonal bandwidth."
    
    if getattr(engine, "mock_execution", False):
        pytest.skip("Compilation environment absent.")
        
    N = engine.layout.n_states
    y = np.random.uniform(0.1, 1.0, size=N).tolist()
    ydot = np.zeros(N).tolist()
    
    # 2. Extract the Banded Jacobian (what the solver actually sees)
    J_banded = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0))
    
    # 3. Extract the True Dense Jacobian by forcing bandwidth=0 via a temporary engine
    dense_engine = Engine(model=model, target="cpu", mock_execution=False, jacobian_bandwidth=0)
    J_dense = np.array(dense_engine.evaluate_jacobian(y, ydot, c_j=1.0))
    
    # DIAGNOSTIC ASSERTION:
    # If the banded matrix drops non-zero elements present in the dense matrix, 
    # the Newton solver will diverge because it is blind to the inter-domain physics.
    # This test will FAIL until the Engine dynamically calculates bandwidth by traversing AST node distances.
    np.testing.assert_allclose(
        J_banded, J_dense, 
        err_msg="Jacobian Bandwidth Truncation detected! The solver is silently dropping macro-micro coupling terms."
    )