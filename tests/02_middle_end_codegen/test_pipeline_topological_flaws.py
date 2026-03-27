"""
Middle-End Codegen: Topological Pipeline Flaws

Isolates critical indexing and boundary projection bugs in the AST-to-C++ compiler.
These tests use the Foreign Function Interface (FFI) as a mathematical oracle to 
prove the generated C++ residuals and Jacobians map multi-scale topologies correctly.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

# ==============================================================================
# Bug 1: Cross-Domain State Projection
# ==============================================================================

class MacroMicroStateProjectionPDE(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=3, name="x")
    r = fx.Domain(bounds=(0, 1), resolution=4, name="r")
    macro_micro = x * r
    
    c_mac = fx.State(domain=x, name="c_mac")
    c_mic = fx.State(domain=macro_micro, name="c_mic")
    
    def math(self):
        return {
            "regions": {
                # Intent: Evaluate the macro state uniformly across the micro loop.
                # Bug: CLAMP(idx) evaluates c_mac at the micro flat index, exceeding 
                # c_mac's bounds and snapping everything to c_mac's last node.
                self.macro_micro: [ fx.dt(self.c_mic) == self.c_mac ]
            }
        }

@REQUIRES_COMPILER
def test_cross_domain_state_projection():
    """
    Proves that evaluating a Macro state inside a Macro-Micro loop correctly 
    projects the flat index back to the macro domain (idx / micro_resolution).
    """
    engine = Engine(model=MacroMicroStateProjectionPDE(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y = np.zeros(N)
    ydot = np.zeros(N)
    
    off_mac, size_mac = engine.layout.state_offsets["c_mac"]
    off_mic, size_mic = engine.layout.state_offsets["c_mic"]
    
    # Inject linearly increasing macro states: [10.0, 20.0, 30.0]
    y[off_mac : off_mac + size_mac] = [10.0, 20.0, 30.0]
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist())
    
    # The mathematical oracle: res = ydot - c_mac = 0.0 - c_mac
    # Micro states 0-3 (Macro node 0) should equal -10.0
    # Micro states 4-7 (Macro node 1) should equal -20.0
    # Micro states 8-11 (Macro node 2) should equal -30.0
    expected_mic_res = [-10.0]*4 + [-20.0]*4 + [-30.0]*4
    
    np.testing.assert_allclose(
        res[off_mic : off_mic + size_mic], 
        expected_mic_res,
        err_msg="Cross-Domain State Projection failed! The macro state was incorrectly indexed from the micro loop."
    )


# ==============================================================================
# Bug 2: Cross-Domain Boundary Evaluation
# ==============================================================================

class CrossDomainBoundaryEvaluationPDE(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=3, name="x")
    r = fx.Domain(bounds=(0, 1), resolution=4, name="r")
    macro_micro = x * r
    
    c_mac = fx.State(domain=x, name="c_mac")
    c_mic = fx.State(domain=macro_micro, name="c_mic")
    
    def math(self):
        return {
            "regions": {
                # Intent: Extract the micro surface concentration for each macro node.
                # Bug: Evaluated in the macro loop, `idx` ranges 0-2. The flat index 
                # equation ((idx / 4) * 4) rounds to 0. Every macro node pulls from particle 0.
                self.x: [ fx.dt(self.c_mac) == self.c_mic.boundary("right", domain=self.r) ]
            }
        }

@REQUIRES_COMPILER
def test_cross_domain_boundary_evaluation():
    """
    Proves that extracting a Micro boundary from within a Macro loop generates 
    the correct flat memory index (macro_idx * micro_resolution + boundary_offset).
    """
    engine = Engine(model=CrossDomainBoundaryEvaluationPDE(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y = np.zeros(N)
    ydot = np.zeros(N)
    
    off_mac, size_mac = engine.layout.state_offsets["c_mac"]
    off_mic, size_mic = engine.layout.state_offsets["c_mic"]
    
    # Inject distinct surface concentrations for each particle
    # Particle 0 (indices 0-3): Right boundary is index 3
    # Particle 1 (indices 4-7): Right boundary is index 7
    # Particle 2 (indices 8-11): Right boundary is index 11
    y[off_mic + 3] = 100.0
    y[off_mic + 7] = 200.0
    y[off_mic + 11] = 300.0
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist())
    
    # Oracle: res = ydot - c_mic_surf = 0.0 - c_mic_surf
    expected_mac_res = [-100.0, -200.0, -300.0]
    
    np.testing.assert_allclose(
        res[off_mac : off_mac + size_mac], 
        expected_mac_res,
        err_msg="Cross-Domain Boundary Evaluation failed! The micro surface was incorrectly indexed from the macro loop."
    )


# ==============================================================================
# Bug 3: Interface Boundary Condition Loss
# ==============================================================================

class InterfaceLossPDE(fx.PDE):
    x_L = fx.Domain(bounds=(0, 1), resolution=3, name="x_L")
    x_R = fx.Domain(bounds=(1, 2), resolution=3, name="x_R")
    
    c_L = fx.State(domain=x_L, name="c_L")
    c_R = fx.State(domain=x_R, name="c_R")
    
    def math(self):
        flux_L = -fx.grad(self.c_L)
        flux_R = -fx.grad(self.c_R)
        return {
            "regions": {
                self.x_L: [ fx.dt(self.c_L) == -fx.div(flux_L) ],
                self.x_R: [ fx.dt(self.c_R) == -fx.div(flux_R) ]
            },
            "boundaries": [
                self.c_L.left == 0.0,
                self.c_R.right == 0.0,
                
                # The Critical Interface
                # Bug: Because both equations target the same LHS domain implicitly, 
                # Dirichlet completely overwrites Neumann. `c_R` gets no boundary condition,
                # destroying interface flux coupling and leaving the Jacobian rank-deficient.
                flux_L.right == flux_R.left,       # Neumann Continuity
                self.c_L.right == self.c_R.left    # Dirichlet Continuity
            ]
        }

@REQUIRES_COMPILER
def test_interface_boundary_condition_loss():
    """
    Proves that adjacent spatial regions explicitly enforce *both* Neumann and 
    Dirichlet boundary conditions across their shared interface natively in the Jacobian.
    """
    engine = Engine(model=InterfaceLossPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    y = np.random.uniform(0.1, 1.0, size=N)
    ydot = np.zeros(N)
    
    J = np.array(engine.evaluate_jacobian(y.tolist(), ydot.tolist(), c_j=1.0))
    
    off_L, size_L = engine.layout.state_offsets["c_L"]
    off_R, _ = engine.layout.state_offsets["c_R"]
    
    # Oracle: To prove flux coupling exists natively, the differential equation 
    # for `c_R` at its left boundary (off_R) MUST analytically depend on the state 
    # of `c_L` at its right boundary (off_L + size_L - 1).
    coupling_derivative = J[off_R, off_L + size_L - 1]
    
    assert abs(coupling_derivative) > 1e-10, (
        "Interface Boundary Condition Loss detected! "
        "The Neumann flux coupling was overwritten by the Dirichlet state coupling, "
        "breaking mathematical conservation across the separator interface."
    )