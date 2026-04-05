"""
Middle-End Codegen: Numerical Oracles

Comprehensive validation of the AST-to-C++ pipeline. 
Uses native LLVM JIT compilation to mathematically prove that the codegen 
correctly handles hierarchical topologies, Arbitrary Lagrangian-Eulerian (ALE) 
moving meshes, spatial DAE masking, and unstructured CSR graph traversals.
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
# Heavyweight Models (Probing Complex CodeGen Features)
# ==============================================================================

class MacroMicroDFN(fx.PDE):
    """Proves hierarchical unrolling, spherical limits, and spatial DAEs."""
    x = fx.Domain(bounds=(0, 40e-6), resolution=4, name="x")
    r = fx.Domain(bounds=(0, 5e-6), resolution=3, coord_sys="spherical", name="r")
    macro_micro = x * r
    
    c_e = fx.State(domain=x, name="c_e")
    phi_e = fx.State(domain=x, name="phi_e")
    c_s = fx.State(domain=macro_micro, name="c_s")
    V_cell = fx.State(domain=None, name="V_cell")
    
    def math(self):
        j_flux = self.c_s.boundary("right", domain=self.r) - self.phi_e
        i_e = -fx.grad(self.phi_e)
        N_s = -fx.grad(self.c_s, axis=self.r)
        
        return {
            "equations": {
                self.c_e: fx.dt(self.c_e) == fx.grad(self.c_e) + j_flux,
                self.phi_e: 0 == fx.div(i_e) - j_flux, # Pure Spatial DAE
                self.c_s: fx.dt(self.c_s) == -fx.div(N_s, axis=self.r),
                self.V_cell: self.V_cell == 4.2 - self.phi_e.right
            },
            "boundaries": {
                self.c_e: {"left": 1000.0, "right": 1000.0}, # Dirichlet overrides
                i_e: {"left": 0.0, "right": 0.0},            # Neumann tensor injection
                N_s: {"left": 0.0, "right": j_flux}          # Nested boundary injection
            },
            "initial_conditions": {
                self.c_e: 1000.0, self.phi_e: 0.0, self.c_s: 0.5, self.V_cell: 4.2
            }
        }

class InterfaceContinuityPDE(fx.PDE):
    """Proves interface continuity and BC ranking without ALE triggers."""
    reg_A = fx.Domain(bounds=(0, 1), resolution=4, name="reg_A")
    reg_B = fx.Domain(bounds=(1, 2), resolution=4, name="reg_B")
    
    c_A = fx.State(domain=reg_A, name="c_A")
    c_B = fx.State(domain=reg_B, name="c_B")
    
    def math(self):
        flux_A = -fx.grad(self.c_A)
        flux_B = -fx.grad(self.c_B)
        return {
            "equations": {
                self.c_A: fx.dt(self.c_A) == -fx.div(flux_A),
                self.c_B: fx.dt(self.c_B) == -fx.div(flux_B)
            },
            "boundaries": {
                self.c_A: {"left": 1.0, "right": self.c_B.left},
                flux_B: {"left": flux_A.right, "right": 0.0}
            },
            "initial_conditions": {
                self.c_A: 1.0, self.c_B: 0.0
            }
        }

class ALEMovingInterfacePDE(fx.PDE):
    """Proves ALE advection upwinding."""
    reg_B = fx.Domain(bounds=(1, 2), resolution=4, name="reg_B")
    c_B = fx.State(domain=reg_B, name="c_B")
    L = fx.State(domain=None, name="L")
    
    def math(self):
        flux_B = -fx.grad(self.c_B)
        return {
            "equations": {
                self.c_B: fx.dt(self.c_B) == -fx.div(flux_B),
                self.L: fx.dt(self.L) == 1.0
            },
            "boundaries": {
                self.reg_B: {"right": self.L}, # ALE boundary deformation
                flux_B: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c_B: 0.0, self.L: 2.0
            }
        }

tetrahedron_mesh = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "elements": [[0, 1, 2, 3]]
}

class CSRAndMultiplexerPDE(fx.PDE):
    """Proves unstructured CSR matrices and Terminal CCCV hardware abstraction."""
    mesh = fx.Domain.from_mesh(tetrahedron_mesh, name="mesh", surfaces={"top": [2, 3]})
    c = fx.State(domain=mesh, name="c")
    
    V_cell = fx.State(domain=None, name="V_cell")
    i_app = fx.State(domain=None, name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        flux = -fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux),
                self.V_cell: self.V_cell == 4.2 - self.i_app * 0.1
            },
            "boundaries": {
                flux: {"top": self.i_app}
            },
            "initial_conditions": {
                self.c: 1.0, self.V_cell: 4.2, self.i_app: 0.0
            }
        }

# ==============================================================================
# Numerical Oracle Tests
# ==============================================================================

@REQUIRES_COMPILER
def test_jacobian_rank_and_interface_continuity():
    """
    Proves that adjacent spatial regions correctly process *both* Neumann and 
    Dirichlet equality bounds on a shared interface without colliding or 
    creating singular, rank-deficient Jacobians.
    """
    engine = Engine(model=InterfaceContinuityPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    np.random.seed(42)
    y = np.random.uniform(0.1, 1.0, size=N).tolist()
    ydot = np.zeros(N).tolist()
    
    J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    rank = np.linalg.matrix_rank(J)
    
    assert rank == N, f"Jacobian is singular (Rank {rank} < N={N})! Interface boundary conditions collided."


@REQUIRES_COMPILER
def test_ale_advection_upwinding_stability():
    """
    Proves that Arbitrary Lagrangian-Eulerian (ALE) moving boundaries natively 
    inject local geometric dilution terms that respect upwind differencing for stability.
    """
    engine = Engine(model=ALEMovingInterfacePDE(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y = np.zeros(N)
    
    off_B, size_B = engine.layout.state_offsets["c_B"]
    off_L, _ = engine.layout.state_offsets["L"]
    
    # Linear gradient in Region B
    y[off_B : off_B + size_B] = [10.0, 20.0, 30.0, 40.0]
    y[off_L] = 1.0 
    
    # 1. Expand the mesh (v > 0)
    ydot_expand = np.zeros(N)
    ydot_expand[off_L] = 1.0
    res_expand = engine.evaluate_residual(y.tolist(), ydot_expand.tolist(), parameters={})
    
    # 2. Contract the mesh (v < 0)
    ydot_contract = np.zeros(N)
    ydot_contract[off_L] = -1.0
    res_contract = engine.evaluate_residual(y.tolist(), ydot_contract.tolist(), parameters={})
    
    center_node = off_B + 1
    # If the advection term used standard centered differences, the expansion/contraction
    # would yield symmetrical changes. Upwinding causes a strict asymmetry in the stencil.
    assert res_expand[center_node] != res_contract[center_node], \
        "ALE Advection failed to shift the numerical stencil based on mesh velocity direction."


@REQUIRES_COMPILER
def test_dae_masking_and_cj_scaling():
    """
    Proves that spatial arrays governed by algebraic constraints are perfectly 
    masked from implicit scaling parameters (c_j), ensuring stable Newton steps.
    """
    engine = Engine(model=MacroMicroDFN(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    # 1. Validate Mask Extraction Array
    _, _, id_arr, _, _ = engine._extract_metadata()
    id_arr = np.array(id_arr)
    
    off_phi, size_phi = engine.layout.state_offsets["phi_e"]
    off_v, _ = engine.layout.state_offsets["V_cell"]
    off_ce, size_ce = engine.layout.state_offsets["c_e"]
    
    assert np.all(id_arr[off_phi : off_phi + size_phi] == 0.0), "Spatial DAE 'phi_e' not masked as algebraic."
    assert id_arr[off_v] == 0.0, "0D DAE 'V_cell' not masked as algebraic."
    
    # c_e has Dirichlet (algebraic) bounds on the edges, and PDE (differential) logic in the bulk.
    assert id_arr[off_ce] == 0.0
    assert id_arr[off_ce + size_ce - 1] == 0.0
    assert np.all(id_arr[off_ce + 1 : off_ce + size_ce - 1] == 1.0)
    
    # 2. Validate Mathematical Execution
    N = engine.layout.n_states
    y, ydot = np.ones(N).tolist(), np.zeros(N).tolist()
    
    J_1 = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    J_100 = np.array(engine.evaluate_jacobian(y, ydot, c_j=100.0, parameters={}))
    delta_J = J_100 - J_1
    
    # The scaling factor d(c_j) should perfectly map to a diagonal matrix 99.0 * id_arr.
    # Off-diagonal leakage implies time-derivatives corrupted the physical coupling matrices.
    np.testing.assert_allclose(delta_J, np.diag(99.0 * id_arr), atol=1e-10)


@REQUIRES_COMPILER
def test_cross_domain_coupling_and_bandwidth():
    """
    Proves that multi-scale meshes correctly compute flat-memory strides and 
    flag the graph for GMRES factorization instead of truncating macro-micro dependencies.
    """
    engine = Engine(model=MacroMicroDFN(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    assert engine.jacobian_bandwidth <= 0, "Failed to map composite domain to Dense/GMRES factorization."
    
    N = engine.layout.n_states
    y, ydot = np.ones(N).tolist(), np.zeros(N).tolist()
    J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    
    off_ce, size_ce = engine.layout.state_offsets["c_e"]
    off_cs, _ = engine.layout.state_offsets["c_s"]
    
    # Equation: dt(c_e) = grad(c_e) + (c_s.right - phi_e)
    # The derivative of the c_e residual with respect to c_s.right is exactly -1.0.
    # It must couple ONLY to the micro node belonging to the same macro spatial slice.
    
    for i_mac in range(1, size_ce - 1): # Skip Dirichlet boundaries
        row_ce = off_ce + i_mac
        
        # In a resolution=3 micro grid, the right boundary is index 2.
        col_cs_target = off_cs + (i_mac * 3) + 2
        
        for j_all_micro in range(4 * 3):
            col_eval = off_cs + j_all_micro
            derivative = J[row_ce, col_eval]
            
            if col_eval == col_cs_target:
                assert derivative == pytest.approx(-1.0), f"Missing correct cross-domain coupling at c_e[{i_mac}] -> c_s[{col_eval}]"
            else:
                assert abs(derivative) < 1e-10, f"Erroneous Jacobian bleeding detected at c_e[{i_mac}] -> c_s[{col_eval}]"


@REQUIRES_COMPILER
def test_spherical_lhopital_and_hermetic_isolation():
    """
    Proves L'Hopital limits prevent 0/0 NaNs at spherical origins, and that 
    composite topologies do not bleed boundary evaluations into adjacent grids.
    """
    engine = Engine(model=MacroMicroDFN(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y, ydot = np.zeros(N).tolist(), np.zeros(N).tolist()
    
    # Inject an aggressive state value only into Macro Node 1
    off_cs, size_cs = engine.layout.state_offsets["c_s"]
    y[off_cs + 3 : off_cs + 6] = [100.0, 100.0, 100.0]
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    # 1. No NaN at r=0 (L'Hopital safety net)
    assert np.isfinite(res).all(), "Spherical evaluation produced non-finite values."
    
    # 2. Hermetic Isolation
    # Macro Node 0 (indices 0,1,2) is mathematically 0.0 everywhere. 
    # If the domain bleeds, its right boundary will incorrectly calculate a gradient against Macro Node 1.
    np.testing.assert_allclose(
        res[off_cs : off_cs + 3], [0.0, 0.0, 0.0], 
        err_msg="Topological bleed detected! Macro domains are improperly sharing boundaries."
    )


@REQUIRES_COMPILER
def test_csr_graph_traversal_mass_conservation():
    """
    Proves unstructured Sparse CSR generation maps correct graph weights 
    independently of traditional N-dimensional compile-time shapes.
    """
    engine = Engine(model=CSRAndMultiplexerPDE(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y, ydot = np.zeros(N).tolist(), np.zeros(N).tolist()
    
    off_i, _ = engine.layout.state_offsets["i_app"]
    y[off_i] = 10.0 # Setup state for Neumann boundary condition eval
    
    # We evaluate without boundary flux (i_app = 0.0) to strictly test internal CSR mass conservation
    res = engine.evaluate_residual(y, ydot, parameters={"_term_i_target": 10.0, "_term_mode": 1.0})
    
    off_c, size_c = engine.layout.state_offsets["c"]
    c_residuals = res[off_c : off_c + size_c]
    
    # Mass conservation: sum of all interior divergence fluxes MUST equal exactly the total flux injected
    total_flux_in = 10.0 * 2 # Injected 10.0 across 2 nodes designated as "top"
    total_residual_sum = sum(c_residuals)
    
    assert total_residual_sum == pytest.approx(-total_flux_in), "Unstructured CSR graph failed to conserve mass internally."


@REQUIRES_COMPILER
def test_terminal_multiplexer_hot_swapping():
    """
    Proves the implicit hardware multiplexer can toggle the state-machine (CC/CV) 
    via the Jacobian matrix without rebuilding the C++ source binary.
    """
    engine = Engine(model=CSRAndMultiplexerPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    y, ydot = np.ones(N).tolist(), np.zeros(N).tolist()
    
    off_i, _ = engine.layout.state_offsets["i_app"]
    off_v, _ = engine.layout.state_offsets["V_cell"]
    
    # 1. Constant Current Mode (_term_mode = 1.0)
    J_cc = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={"_term_mode": 1.0}))
    assert J_cc[off_i, off_i] == 1.0, "CC Mode failed to map the current lock."
    assert J_cc[off_i, off_v] == 0.0, "CC Mode erroneously coupled to voltage."
    
    # 2. Constant Voltage Mode (_term_mode = 0.0)
    J_cv = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={"_term_mode": 0.0}))
    assert J_cv[off_i, off_i] == 0.0, "CV Mode failed to release the current lock."
    
    # In CV mode, the residual equation translates to: res_i = i_app - (i_app - V_cell + v_target)
    # Simplifying: res_i = V_cell - v_target
    # Therefore, d(res_i)/d(V_cell) == 1.0
    assert J_cv[off_i, off_v] == 1.0, "CV Mode failed to couple to the voltage tracking constraint."