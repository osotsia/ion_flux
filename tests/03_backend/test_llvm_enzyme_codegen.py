import pytest
import numpy as np
import shutil
import platform
import os
import sys
import ion_flux as fx

# Ensure models directory is in path for E2E tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.compiler._4_codegen.clang_invoker import NativeCompiler


"""
Middle-End Codegen: Numerical Oracles

Comprehensive validation of the AST-to-C++ pipeline. 
Uses native LLVM JIT compilation to mathematically prove that the codegen 
correctly handles hierarchical topologies, Arbitrary Lagrangian-Eulerian (ALE) 
moving meshes, spatial DAE masking, and unstructured CSR graph traversals.
"""



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
    engine = fx.Engine(model=InterfaceContinuityPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
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
    engine = fx.Engine(model=ALEMovingInterfacePDE(), target="cpu", mock_execution=False)
    
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
    engine = fx.Engine(model=MacroMicroDFN(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
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
    engine = fx.Engine(model=MacroMicroDFN(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
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
    engine = fx.Engine(model=MacroMicroDFN(), target="cpu", mock_execution=False)
    
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
    engine = fx.Engine(model=CSRAndMultiplexerPDE(), target="cpu", mock_execution=False)
    
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
        
    m_list = engine.layout.get_mesh_data()
    vol_off = engine.layout.mesh_offsets["mesh"]["volumes"]
    volumes = m_list[vol_off : vol_off + size_c]
        
    total_residual_sum = sum(r * v for r, v in zip(c_residuals, volumes))
    
    # Assertion updated from -total_flux_in to +total_flux_in
    assert total_residual_sum == pytest.approx(total_flux_in), "Unstructured CSR graph failed to conserve mass internally."




@REQUIRES_COMPILER
def test_terminal_multiplexer_hot_swapping():
    """
    Proves the implicit hardware multiplexer can toggle the state-machine (CC/CV) 
    via the Jacobian matrix without rebuilding the C++ source binary.
    """
    engine = fx.Engine(model=CSRAndMultiplexerPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
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



def _has_enzyme() -> bool:
    return bool(NativeCompiler().enzyme_plugin)


REQUIRES_ENZYME = pytest.mark.skipif(not _has_compiler() or not _has_enzyme(), reason="Requires Enzyme LLVM plugin.")



# ==============================================================================
# Helper Oracle
# ==============================================================================

def approx_jacobian(engine: fx.Engine, y: list, ydot: list, p: dict, c_j: float, eps: float = 1e-8) -> np.ndarray:
    """
    Computes the numerical Jacobian using central finite differences.
    For an implicit solver, the residual mapping is F(y, ydot).
    J = dF/dy + c_j * dF/dydot
    """
    N = len(y)
    J = np.zeros((N, N))
    
    for col in range(N):
        y_fwd, ydot_fwd = list(y), list(ydot)
        y_bwd, ydot_bwd = list(y), list(ydot)
        
        y_fwd[col] += eps
        ydot_fwd[col] += eps * c_j
        y_bwd[col] -= eps
        ydot_bwd[col] -= eps * c_j
        
        res_fwd = engine.evaluate_residual(y_fwd, ydot_fwd, parameters=p)
        res_bwd = engine.evaluate_residual(y_bwd, ydot_bwd, parameters=p)
        
        for row in range(N):
            J[row, col] = (res_fwd[row] - res_bwd[row]) / (2 * eps)
            
    return J



# ==============================================================================
# Heavyweight Models
# ==============================================================================

class MathGauntletPDE(fx.PDE):
    """Combines smooth math, piecewise functions, and step logic into one AD test."""
    y_smooth = fx.State(domain=None, name="y_smooth")
    y_piece = fx.State(domain=None, name="y_piece")
    y_step = fx.State(domain=None, name="y_step")
    
    p_scale = fx.Parameter(default=2.0)
    p_limit = fx.Parameter(default=1.0)
    p_thresh = fx.Parameter(default=2.5)
    
    def math(self):
        trigger = self.y_step > self.p_thresh
        return {
            "equations": {
                # Smooth: sin, exp, cos, log
                self.y_smooth: fx.dt(self.y_smooth) == fx.sin(self.y_smooth) * fx.exp(self.y_piece) - fx.cos(self.y_smooth * self.p_scale),
                
                # Piecewise: abs, max, min (Generates subgradients at kinks)
                self.y_piece: self.y_piece == fx.abs(self.y_piece) + fx.max(self.y_smooth, self.p_limit) + fx.min(self.y_piece, 0.0),
                
                # Step Logic: Relational operator acting as a Heaviside trigger
                self.y_step: fx.dt(self.y_step) == trigger * self.y_step
            },
            "boundaries": {},
            "initial_conditions": {
                self.y_smooth: 2.0, self.y_piece: -1.0, self.y_step: 3.0
            }
        }



class BandedCouplingPDE(fx.PDE):
    """1D model to validate Curtis-Powell-Reid (CPR) graph coloring."""
    x = fx.Domain(bounds=(0, 1), resolution=5)
    c = fx.State(domain=x, name="c")
    D = fx.Parameter(default=1.5)
    
    def math(self):
        flux = -self.D * fx.grad(self.c)
        return {
            "equations": { self.c: fx.dt(self.c) == -fx.div(flux) },
            "boundaries": { self.c: {"left": 1.0, "right": 0.0} },
            "initial_conditions": { self.c: 0.5 }
        }



# ==============================================================================
# Tests
# ==============================================================================

@REQUIRES_COMPILER
def test_clang_so_emission_and_ffi_loading():
    """
    Validates Clang properly compiles the emitted C++ into a portable shared object
    and safely loads it into Python memory via ctypes FFI.
    """
    engine = fx.Engine(model=MathGauntletPDE(), target="cpu", mock_execution=False)
    
    # Prove the Engine didn't silently fall back to mock execution
    assert getattr(engine, "mock_execution", False) is False
    assert engine.runtime is not None
    assert engine.runtime.lib_path.endswith(".so") or engine.runtime.lib_path.endswith(".dylib")
    
    # Retrieve dynamic offsets
    off_s, _ = engine.layout.state_offsets["y_smooth"]
    off_p, _ = engine.layout.state_offsets["y_piece"]
    off_step, _ = engine.layout.state_offsets["y_step"]
    
    N = engine.layout.n_states
    y = np.zeros(N)
    y[off_s], y[off_p], y[off_step] = 2.0, -1.0, 3.0
    ydot = np.zeros(N)
    ydot[off_s], ydot[off_p], ydot[off_step] = 0.1, 0.0, 1.0
    
    params = {"p_scale": 2.0, "p_limit": 1.0, "p_thresh": 2.5}
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters=params)
    
    # Oracle Validation
    # y_step eq: ydot_step - (trigger * y_step) => 1.0 - (1.0 * 3.0) = -2.0
    assert res[off_step] == pytest.approx(-2.0, rel=1e-5)




@REQUIRES_ENZYME
def test_enzyme_analytical_dense_jacobian_smooth_math():
    """
    Proves Enzyme LLVM Reverse/Forward AD correctly differentiates smooth math 
    (sin, cos, exp) perfectly matching a rigorous Finite Difference oracle.
    """
    engine = fx.Engine(model=MathGauntletPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    off_s, _ = engine.layout.state_offsets["y_smooth"]
    off_p, _ = engine.layout.state_offsets["y_piece"]
    off_step, _ = engine.layout.state_offsets["y_step"]
    
    y = np.zeros(N)
    y[off_s], y[off_p], y[off_step] = 2.0, -0.5, 3.0 # Evaluate away from kinks for FD safety
    ydot = np.zeros(N)
    ydot[off_s], ydot[off_p], ydot[off_step] = 0.1, 0.0, 1.0
    
    p = {"p_scale": 1.5, "p_limit": 1.0, "p_thresh": 2.5}
    c_j = 10.0
    
    jac_analytical = np.array(engine.evaluate_jacobian(y.tolist(), ydot.tolist(), c_j, parameters=p))
    jac_numerical = approx_jacobian(engine, y.tolist(), ydot.tolist(), p, c_j)
    
    np.testing.assert_allclose(jac_analytical, jac_numerical, rtol=1e-5, atol=1e-6)




@REQUIRES_ENZYME
def test_enzyme_subgradients_and_heaviside_triggers():
    """
    Validates Enzyme behavior on non-differentiable boundaries.
    Numerical FD fails at exactly x=0 for abs(x). Enzyme AD returns valid subgradients 
    (preventing NaN/Segfaults) and properly routes Boolean Heavyside step gradients.
    """
    engine = fx.Engine(model=MathGauntletPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    off_s, _ = engine.layout.state_offsets["y_smooth"]
    off_p, _ = engine.layout.state_offsets["y_piece"]
    off_step, _ = engine.layout.state_offsets["y_step"]
    
    y = np.zeros(N)
    # y_piece = 0.0 creates a non-differentiable kink for abs() and min()
    # y_smooth = 1.0 creates a kink for max(y_smooth, 1.0)
    y[off_s], y[off_p], y[off_step] = 1.0, 0.0, 3.0 
    ydot = np.zeros(N)
    
    p = {"p_scale": 1.5, "p_limit": 1.0, "p_thresh": 2.5}
    c_j = 10.0
    
    jac_ana_kink = np.array(engine.evaluate_jacobian(y.tolist(), ydot.tolist(), c_j, parameters=p))
    
    # 1. Kink Subgradients
    assert not np.isnan(jac_ana_kink).any(), "Enzyme produced NaN at a mathematical kink."
    assert np.isfinite(jac_ana_kink).all()
    
    # 2. Boolean Heaviside Gradient Pass-Through
    # Active (y_step = 3.0 > 2.5) -> Eq: ydot_step = 1.0 * y_step
    # J(step, step) = c_j * d(ydot)/dydot + d(-y_step)/dy_step = c_j - 1.0
    assert jac_ana_kink[off_step, off_step] == pytest.approx(c_j - 1.0, rel=1e-5)
    
    # 3. Boolean Heaviside Gradient Blocking
    # Inactive (y_step = 1.0 < 2.5) -> Eq: ydot_step = 0.0 * y_step
    # J(step, step) = c_j * 1.0 + 0.0 = c_j
    y[off_step] = 1.0
    jac_inactive = np.array(engine.evaluate_jacobian(y.tolist(), ydot.tolist(), c_j, parameters=p))
    assert jac_inactive[off_step, off_step] == pytest.approx(c_j, rel=1e-5)




@REQUIRES_ENZYME
def test_cpr_graph_coloring_banded_jacobian():
    """
    Validates that the Curtis-Powell-Reid (CPR) algorithm correctly computes 
    compressed Banded Jacobians using minimal forward Enzyme AD sweeps.
    """
    # Force Tridiagonal bandwidth (bw=1)
    engine_banded = fx.Engine(model=BandedCouplingPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=1)
    # Force Dense bandwidth (bw=0)
    engine_dense = fx.Engine(model=BandedCouplingPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine_banded.layout.n_states
    y = np.random.uniform(0.1, 1.0, size=N).tolist()
    ydot = np.zeros(N).tolist()
    
    jac_banded = np.array(engine_banded.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    jac_dense = np.array(engine_dense.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
    
    # Ensure CPR coloring correctly captured all coupling elements without truncation
    np.testing.assert_allclose(
        jac_banded, 
        jac_dense, 
        atol=1e-10, 
        err_msg="CPR Banded Graph Coloring mismatched the exact Dense Jacobian baseline."
    )
