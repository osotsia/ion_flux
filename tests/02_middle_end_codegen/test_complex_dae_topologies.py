"""
test_complex_dae_topologies.py

Validates the DSL-to-Binary pipeline against complex topological architectures
(e.g., highly coupled PDE-ODE-DAE systems, 1D-1D macro-micro cross-products, 
and explicitly stitched regional interfaces).

Focuses on diagnosing the root causes of "Algebraic initialization failed" 
by isolating Jacobian rank-deficiency, DAE masking, and bandwidth truncation.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

# ==============================================================================
# Environment Configuration
# ==============================================================================

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

# ==============================================================================
# Mock Models Isolating Complex Architectural Features
# ==============================================================================

class MinimalDFN(fx.PDE):
    """
    Happy Path Model: A structurally complete but minimal 1D-1D DFN equivalent.
    Couples a macro domain (x) to a micro domain (r) with spatial algebraic DAEs.
    """
    x = fx.Domain(bounds=(0, 10e-6), resolution=5, name="x")
    r = fx.Domain(bounds=(0, 5e-6), resolution=4, coord_sys="spherical", name="r")
    macro_micro = x * r
    
    c_e = fx.State(domain=x, name="c_e")              # Macro PDE
    phi_e = fx.State(domain=x, name="phi_e")          # Macro Spatial DAE
    c_s = fx.State(domain=macro_micro, name="c_s")    # Macro-Micro PDE
    V_cell = fx.State(domain=None, name="V_cell")     # 0D Algebraic DAE
    
    def math(self):
        j_flux = self.c_s.boundary("right", domain=self.r) - self.phi_e
        return {
            "regions": {
                self.x: [
                    fx.dt(self.c_e) == fx.grad(self.c_e) + j_flux,
                    0 == fx.div(fx.grad(self.phi_e)) - j_flux  # Pure Spatial DAE
                ],
                self.macro_micro: [
                    fx.dt(self.c_s) == fx.grad(self.c_s, axis=self.r)
                ]
            },
            "boundaries": [
                self.c_e.left == 1000.0, self.c_e.right == 1000.0,
                self.phi_e.left == 0.0, self.phi_e.right == 0.0,
                self.c_s.boundary("left", domain=self.r) == 0.0,
                self.c_s.boundary("right", domain=self.r) == 0.0
            ],
            "global": [
                self.V_cell == 4.2 - self.phi_e.right,
                self.c_e.t0 == 1000.0, self.phi_e.t0 == 0.0,
                self.c_s.t0 == 500.0, self.V_cell.t0 == 4.2
            ]
        }

class RegionalInterfaceCoupling(fx.PDE):
    """
    Category 1 (Interface Continuity): Isolates adjacent regions linked by 
    explicit state and flux equality constraints.
    """
    reg_A = fx.Domain(bounds=(0, 1), resolution=4, name="reg_A")
    reg_B = fx.Domain(bounds=(1, 2), resolution=4, name="reg_B")
    
    c_A = fx.State(domain=reg_A, name="c_A")
    c_B = fx.State(domain=reg_B, name="c_B")
    
    def math(self):
        flux_A = -fx.grad(self.c_A)
        flux_B = -fx.grad(self.c_B)
        return {
            "regions": {
                self.reg_A: [ fx.dt(self.c_A) == -fx.div(flux_A) ],
                self.reg_B: [ fx.dt(self.c_B) == -fx.div(flux_B) ]
            },
            "boundaries": [
                self.c_A.left == 1.0,
                self.c_B.right == 0.0,
                # The Critical Interface: Overlapping constraints
                self.c_A.right == self.c_B.left,
                flux_A.right == flux_B.left
            ]
        }

class SpatialAlgebraicDAE(fx.PDE):
    """
    Category 2 & 3 (DAE Masking & Initialization): Isolates a spatial grid 
    governed entirely by algebraic constraints without time derivatives.
    """
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")       # ODE/PDE
    phi = fx.State(domain=x, name="phi")   # DAE
    
    def math(self):
        return {
            "regions": {
                self.x: [
                    fx.dt(self.c) == fx.grad(self.c),
                    0 == fx.grad(self.phi) + self.c   # Pure Spatial DAE
                ]
            },
            "boundaries": [
                self.c.left == 1.0, self.c.right == 0.0,
                self.phi.left == 0.0, self.phi.right == 0.0
            ],
            "global": [
                self.c.t0 == 0.5,
                self.phi.t0 == 0.0  # Initial guess for Newton solver
            ]
        }

# ==============================================================================
# The Happy Path
# ==============================================================================

@REQUIRES_COMPILER
def test_happy_path_complex_dae_compilation():
    """
    Validates that a complex 1D-1D macro-micro model compiles, identifies its
    bandwidth requirements, and evaluates to a finite initial residual.
    """
    engine = Engine(model=MinimalDFN(), target="cpu", mock_execution=False)
    
    # Validation 1: Bandwidth Truncation Prevention
    # The heuristic MUST detect the hierarchical cross-product (x * r) and
    # assign a dense or GMRES bandwidth (<= 0) to avoid dropping coupling terms.
    assert engine.jacobian_bandwidth <= 0, "Failed to assign dense/GMRES bandwidth to a composite topology."
    
    # Validation 2: Initial Residual Finiteness
    # Ensures the AST-parsed initial conditions do not result in NaN/Inf math,
    # which is the #1 cause of immediate line search rejection.
    y0, ydot0, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0, parameters={})
    
    assert np.isfinite(res).all(), "Initial residual evaluation produced non-finite values."

# ==============================================================================
# Failure Mode Category 1: Interface Continuity & Rank Deficiency
# ==============================================================================

@REQUIRES_COMPILER
def test_category1_interface_continuity_rank():
    """
    Failure Mode: If the C++ codegen naively emits both `c_A.right = c_B.left` 
    and `flux_A.right = flux_B.left` onto the same physical node without cleanly 
    shifting one to the adjacent domain, the Jacobian becomes rank-deficient (singular).
    """
    engine = Engine(model=RegionalInterfaceCoupling(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    # Random uniform state to avoid trivial zero-cancellations in the Jacobian
    np.random.seed(42)
    y_rand = np.random.uniform(0.1, 1.0, size=N).tolist()
    ydot_zero = np.zeros(N).tolist()
    
    J = np.array(engine.evaluate_jacobian(y_rand, ydot_zero, c_j=1.0, parameters={}))
    rank = np.linalg.matrix_rank(J)
    
    # A full-rank Jacobian guarantees that the interface equations are linearly 
    # independent and safely stitched across the two domains.
    assert rank == N, f"Jacobian is singular (Rank {rank} < N={N})! Interface boundary conditions collided."

# ==============================================================================
# Failure Mode Category 2: Spatial DAE Masking
# ==============================================================================

@REQUIRES_COMPILER
def test_category2_algebraic_dae_masking():
    """
    Failure Mode: If a spatial equation declared as `0 == RHS` is not flagged as 
    algebraic (id=0.0) across its entire spatial array, the solver will treat it 
    as an ODE. This corrupts the Newton step scaling (c_j).
    """
    engine = Engine(model=SpatialAlgebraicDAE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    # 1. Verify AST-extracted ID Mask Intent
    _, _, id_arr, _ = engine._extract_metadata()
    id_arr = np.array(id_arr)
    
    off_phi, size_phi = engine.layout.state_offsets["phi"]
    off_c, size_c = engine.layout.state_offsets["c"]
    
    assert np.all(id_arr[off_phi : off_phi + size_phi] == 0.0), "Spatial DAE 'phi' was not fully masked as algebraic."
    
    # PDE 'c' has Dirichlet boundaries. Edges must be algebraic (0.0), bulk must be differential (1.0).
    id_arr_c = id_arr[off_c : off_c + size_c]
    assert id_arr_c[0] == 0.0, "Left boundary of PDE 'c' not masked as algebraic."
    assert id_arr_c[-1] == 0.0, "Right boundary of PDE 'c' not masked as algebraic."
    assert np.all(id_arr_c[1:-1] == 1.0), "Bulk nodes of PDE 'c' incorrectly masked."

    # 2. Verify Executed Mathematical Independence from Time (ydot)
    N = engine.layout.n_states
    y = np.ones(N).tolist()
    ydot = np.zeros(N).tolist()
    
    J_1 = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    J_100 = np.array(engine.evaluate_jacobian(y, ydot, c_j=100.0, parameters={}))
    
    delta_J = J_100 - J_1
    
    # The derivative of the residual with respect to ydot for 'phi' MUST be strictly zero.
    delta_phi_rows = delta_J[off_phi : off_phi + size_phi, :]
    np.testing.assert_allclose(delta_phi_rows, 0.0, atol=1e-12, err_msg="Algebraic DAE rows were corrupted by time-step scaling.")

# ==============================================================================
# Failure Mode Category 3: Algebraic Initialization Domain Validations
# ==============================================================================

class SingularInitDAE(fx.PDE):
    """Isolates algebraic states initializing outside valid mathematical domains."""
    V = fx.State(domain=None)
    
    def math(self):
        return {
            "global": [
                # At t0, V=0.0 -> log(0.0) yields -Inf, triggering immediate line search rejection
                self.V == fx.log(self.V) + 1.0, 
                self.V.t0 == 0.0
            ]
        }

@REQUIRES_COMPILER
def test_category3_initial_condition_domain_crash():
    """
    Failure Mode: Validates that invalid initial guesses for algebraic roots 
    yield non-finite residuals. A well-architected engine surfaces this quickly 
    rather than hanging the numerical solver.
    """
    engine = Engine(model=SingularInitDAE(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0, parameters={})
    
    # Expected behavior: The pipeline successfully evaluates the AST, but the 
    # mathematical consequence of the initial condition yields a NaN or Inf.
    assert not np.isfinite(res).all(), "Expected singular math evaluation, but got finite numbers."

# ==============================================================================
# Category 4: Commutative Interface Routing
# ==============================================================================

class UnsortedInterfacePoisson(fx.PDE):
    """
    Two regions governed by a spatial DAE (Poisson equation).
    Written intuitively: Both interface equations put Region A on the LHS.
    """
    reg_A = fx.Domain(bounds=(0, 1), resolution=4, name="reg_A")
    reg_B = fx.Domain(bounds=(1, 2), resolution=4, name="reg_B")
    
    phi_A = fx.State(domain=reg_A, name="phi_A")
    phi_B = fx.State(domain=reg_B, name="phi_B")
    
    def math(self):
        flux_A = -fx.grad(self.phi_A)
        flux_B = -fx.grad(self.phi_B)
        return {
            "regions": {
                self.reg_A: [ 0 == -fx.div(flux_A) ],
                self.reg_B: [ 0 == -fx.div(flux_B) ]
            },
            "boundaries": [
                self.phi_A.left == 0.0,           # Anchor A
                
                # Written intuitively. The compiler must auto-flip one of these.
                self.phi_A.right == self.phi_B.left,
                flux_A.right == flux_B.left,
                
                flux_B.right == 0.0               # Neumann B
            ],
            "global": [
                self.phi_A.t0 == 0.0, self.phi_B.t0 == 0.0
            ]
        }

@REQUIRES_COMPILER
def test_commutative_dsl_resolves_interface_collisions():
    """
    Validates that the compiler automatically detects target collisions on the LHS 
    and flips the equation to constrain the unassigned RHS node, preserving matrix rank.
    """
    engine = Engine(model=UnsortedInterfacePoisson(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    y = np.random.uniform(0.1, 1.0, size=N).tolist()
    ydot = np.zeros(N).tolist()
    
    J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    rank = np.linalg.matrix_rank(J)
    
    assert rank == N, f"Jacobian is singular (Rank {rank} < N={N})! Commutative DSL auto-flipper failed."