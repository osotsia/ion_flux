"""
Middle-End Codegen: Solver Instability Diagnostics

Isolates and proves the existence of severe numerical and topological flaws 
in the AST-to-C++ compiler causing the native implicit solver to diverge.
These tests demand correct mathematical behavior and will FAIL until the pipeline is fixed.
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

RUST_FFI_AVAILABLE = _has_compiler()


# ==============================================================================
# Suspect 1: Wide Stencil Decoupling (Checkerboarding)
# ==============================================================================

class WideStencilDiagnosticPDE(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    phi = fx.State(domain=x, name="phi")
    
    def math(self):
        return {
            "regions": {
                self.x: [ 0 == fx.div(fx.grad(self.phi)) + self.phi ]
            },
            "boundaries": [
                self.phi.left == 1.0,
                self.phi.right == 0.0
            ]
        }

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native compiler.")
def test_compact_stencil_adjacency_coupling():
    """
    DIAGNOSTIC: Proves whether the AST compiler emits a compact 3-point stencil (correct)
    or a wide 5-point stencil (flawed) for div(grad(phi)).
    """
    engine = Engine(model=WideStencilDiagnosticPDE(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    J = np.array(engine.evaluate_jacobian(np.zeros(N).tolist(), np.zeros(N).tolist(), c_j=1.0))
    off_phi, _ = engine.layout.state_offsets["phi"]
    
    # Extract the Jacobian row for the exact center node (index 2)
    coupling_to_left_neighbor = J[off_phi + 2, off_phi + 1]
    coupling_to_right_neighbor = J[off_phi + 2, off_phi + 3]
    
    # If the DSL emits a compact staggered grid, it MUST depend on adjacent nodes.
    assert abs(coupling_to_left_neighbor) > 1e-10, (
        "Checkerboard Instability Detected! The divergence operator skipped the left adjacent node."
    )
    assert abs(coupling_to_right_neighbor) > 1e-10, (
        "Checkerboard Instability Detected! The divergence operator skipped the right adjacent node."
    )


# ==============================================================================
# Suspect 2: Discrete Flux Non-Conservation
# ==============================================================================

class DiscreteConservationDiagnosticPDE(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=10, name="x")
    c = fx.State(domain=x, name="c")
    
    def math(self):
        flux = -fx.grad(self.c)
        return {
            "regions": { self.x: [ fx.dt(self.c) == -fx.div(flux) ] },
            "boundaries": [ flux.left == 10.0, flux.right == 0.0 ]
        }

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native compiler.")
def test_discrete_flux_conservation():
    """
    DIAGNOSTIC: Proves whether summing the divergence over all nodes exactly matches
    the net boundary flux, ensuring strict mass/charge conservation.
    """
    engine = Engine(model=DiscreteConservationDiagnosticPDE(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    res = engine.evaluate_residual(np.zeros(N).tolist(), np.zeros(N).tolist(), parameters={})
    
    # The sum of divergence over the domain * dx MUST exactly equal (Flux_out - Flux_in)
    dx = 1.0 / 9.0
    discrete_integral = np.sum(res) * dx
    
    # Net flux expected = Flux_right - Flux_left = 0.0 - 10.0 = -10.0
    expected_integral = -10.0
    
    np.testing.assert_allclose(
        discrete_integral, expected_integral, atol=1e-5,
        err_msg="Conservation Violated! The divergence operator does not perfectly telescope."
    )


# ==============================================================================
# Suspect 3: Scale-Induced Ill-Conditioning
# ==============================================================================

class HighConditionNumberDiagnosticPDE(fx.PDE):
    x = fx.Domain(bounds=(0, 40e-6), resolution=10, name="x")
    phi = fx.State(domain=x, name="phi")
    
    def math(self):
        i_s = -100.0 * fx.grad(self.phi)
        return {
            "regions": { self.x: [ 0 == fx.div(i_s) + self.phi ] },
            "boundaries": [ i_s.left == -30.0, i_s.right == 0.0 ]
        }

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native compiler.")
def test_high_condition_number_dae():
    """
    DIAGNOSTIC: Evaluates the raw mathematical condition number of battery-scale parameters.
    """
    engine = Engine(model=HighConditionNumberDiagnosticPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    J = np.array(engine.evaluate_jacobian(np.zeros(N).tolist(), np.zeros(N).tolist(), c_j=1.0))
    cond_number = np.linalg.cond(J)
    
    # A numerically healthy matrix requires a condition number low enough to survive double precision math
    assert cond_number < 1e8, f"Severe Ill-Conditioning Detected: Condition Number is {cond_number:.2e}."


# ==============================================================================
# Suspect 4: Interface Boundary Condition Loss (DFN specific)
# ==============================================================================

class InterfaceLossDiagnosticPDE(fx.PDE):
    x_L = fx.Domain(bounds=(0, 1), resolution=4, name="x_L")
    x_R = fx.Domain(bounds=(1, 2), resolution=4, name="x_R")
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
                # The Critical Interface: Both Neumann & Dirichlet constraints
                flux_L.right == flux_R.left,
                self.c_L.right == self.c_R.left
            ]
        }

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native compiler.")
def test_interface_boundary_condition_preservation():
    """
    DIAGNOSTIC: Proves whether the compiler successfully maps both Neumann and Dirichlet 
    conditions across a shared interface without overwriting one of them.
    """
    engine = Engine(model=InterfaceLossDiagnosticPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    np.random.seed(42)
    y = np.random.uniform(0.1, 1.0, size=N).tolist()
    
    J = np.array(engine.evaluate_jacobian(y, np.zeros(N).tolist(), c_j=1.0))
    rank = np.linalg.matrix_rank(J)
    
    assert rank == N, (
        f"Interface Loss Detected! The Jacobian is rank-deficient (Rank {rank} < {N}). "
        "The Dirichlet state boundary completely overwrote the Neumann flux boundary."
    )