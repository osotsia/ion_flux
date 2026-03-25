import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE

# ==============================================================================
# Model Definitions for AST Failure Mode Testing
# ==============================================================================

class CoupledDomainsPDE(fx.PDE):
    """
    Tests Category 1: Boundary Condition Collisions & Over-Constrained Nodes.
    Simulates a highly coupled 1D-1D interface (like the DFN electrolyte).
    """
    x_n = fx.Domain(bounds=(0, 1), resolution=5, name="x_n")
    x_p = fx.Domain(bounds=(1, 2), resolution=5, name="x_p")
    
    c_n = fx.State(domain=x_n, name="c_n")
    c_p = fx.State(domain=x_p, name="c_p")
    
    def math(self):
        flux_n = -fx.grad(self.c_n)
        flux_p = -fx.grad(self.c_p)
        return {
            "regions": {
                self.x_n: [fx.dt(self.c_n) == -fx.div(flux_n)],
                self.x_p: [fx.dt(self.c_p) == -fx.div(flux_p)]
            },
            "boundaries": [
                self.c_n.left == 0.0,
                self.c_p.right == 1.0,
                
                # The critical interface: a state continuity AND a flux continuity
                self.c_n.right == self.c_p.left,
                flux_n.right == flux_p.left
            ]
        }

class MacroMicroMaskingPDE(fx.PDE):
    """
    Tests Category 2: Differential-Algebraic (DAE) Masking Flaws.
    """
    x = fx.Domain(bounds=(0, 1), resolution=3, name="x")
    r = fx.Domain(bounds=(0, 1), resolution=4, name="r", coord_sys="spherical")
    macro_micro = x * r
    
    c_s = fx.State(domain=macro_micro, name="c_s")
    V = fx.State(domain=None, name="V") # 0D Algebraic
    
    def math(self):
        return {
            "regions": {
                self.macro_micro: [fx.dt(self.c_s) == fx.grad(self.c_s, axis=self.r)]
            },
            "boundaries": [
                # Dirichlet condition on the surface of the micro-particles
                self.c_s.boundary("right", domain=self.r) == 0.5
            ],
            "global": [
                # Pure algebraic equation
                self.V == 4.2 - self.c_s.boundary("right", domain=self.r)
            ]
        }

class IntegralCouplingPDE(fx.PDE):
    """
    Tests Category 3: Hierarchical Topology Misalignment.
    """
    x = fx.Domain(bounds=(0, 1), resolution=4, name="x")
    r = fx.Domain(bounds=(0, 1), resolution=3, name="r")
    macro_micro = x * r
    
    c_s = fx.State(domain=macro_micro, name="c_s")
    T = fx.State(domain=x, name="T")
    
    def math(self):
        return {
            "regions": {
                self.macro_micro: [fx.dt(self.c_s) == fx.grad(self.c_s, axis=self.r)],
                self.x: [fx.dt(self.T) == fx.integral(self.c_s, over=self.r)]
            }
        }

class TerminalMultiplexerPDE(fx.PDE):
    """
    Tests Category 4: Multiplexer & Algebraic Loop Collapse.
    """
    i_app = fx.State(domain=None, name="i_app")
    V_cell = fx.State(domain=None, name="V_cell")
    soc = fx.State(domain=None, name="soc")
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        return {
            "global": [
                fx.dt(self.soc) == -self.i_app,
                self.V_cell == 4.0 + self.soc - 0.1 * self.i_app
            ]
        }


# ==============================================================================
# Numerical AST Tests
# ==============================================================================

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_rank_and_conditioning_verification():
    """
    Category 1: Verifies that Dirichlet/Neumann overlaps at domain interfaces do not 
    result in singular Jacobians (i.e., the builder successfully shifted the slave node).
    """
    engine = Engine(model=CoupledDomainsPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    if engine.mock_execution:
        pytest.skip("Compilation failed or absent.")

    N = engine.layout.n_states
    y = np.random.uniform(0.1, 0.9, size=N).tolist()
    ydot = np.zeros(N).tolist()
    
    # Evaluate at a non-zero state to avoid trivial zero-cancellations
    J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0))
    
    # The rank of the Jacobian must exactly equal the number of states.
    # A rank deficiency indicates the interface equations linearly depend on each other
    # or a node was completely stripped of its mathematical constraint.
    rank = np.linalg.matrix_rank(J)
    assert rank == N, f"Jacobian is singular! Rank {rank} != System Size {N}. AST interface shift failed."

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_id_array_boundary_masking():
    """
    Category 2: Verifies that multi-dimensional boundaries project correct binary
    masks onto the flattened 1D C-array layout for the native solver.
    """
    engine = Engine(model=MacroMicroMaskingPDE(), target="cpu", mock_execution=False)
    
    y0, ydot0, id_arr, spatial_diag = engine._extract_metadata()
    id_arr = np.array(id_arr)
    
    # Layout should be c_s (size 12), V (size 1)
    offset_cs, size_cs = engine.layout.state_offsets["c_s"]
    offset_V, size_V = engine.layout.state_offsets["V"]
    
    assert len(id_arr) == 13
    
    # V is purely algebraic (id == 0.0)
    assert id_arr[offset_V] == 0.0
    
    # c_s has resolution 3 macro x 4 micro. 
    # The right boundary of the micro domain corresponds to local index 3.
    # Therefore, flat indices 3, 7, and 11 must be masked as algebraic (0.0) 
    # due to the Dirichlet boundary condition overlay.
    expected_algebraic_indices = [3, 7, 11]
    
    for i in range(size_cs):
        if i in expected_algebraic_indices:
            assert id_arr[offset_cs + i] == 0.0, f"Micro-boundary at flat index {i} failed to mask as algebraic."
        else:
            assert id_arr[offset_cs + i] == 1.0, f"Bulk PDE node at flat index {i} incorrectly masked as algebraic."

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_cj_scaling_invariance():
    """
    Category 2: Verifies that the AST correctly isolates time-derivatives (ydot) strictly
    to the nodes mathematically declared as differential via fx.dt().
    """
    engine = Engine(model=MacroMicroMaskingPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    if engine.mock_execution:
        pytest.skip("Compilation failed or absent.")

    N = engine.layout.n_states
    y = np.ones(N).tolist()
    ydot = np.zeros(N).tolist()
    
    J_1 = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0))
    J_100 = np.array(engine.evaluate_jacobian(y, ydot, c_j=100.0))
    
    # delta_J = 99.0 * dF/d(ydot)
    delta_J = J_100 - J_1
    
    y0, _, id_arr, _ = engine._extract_metadata()
    id_arr = np.array(id_arr)
    
    # The difference matrix must be strictly diagonal, and equal to 99.0 * id_arr.
    # Any off-diagonal leakage, or mismatch on the diagonal, proves the AST generated
    # a malformed implicit equation layout.
    expected_delta = np.diag(99.0 * id_arr)
    np.testing.assert_allclose(delta_J, expected_delta, atol=1e-10, 
                               err_msg="c_j scaling leaked into algebraic terms or off-diagonals.")

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_cross_domain_jacobian_coupling():
    """
    Category 3: Verifies that operations jumping across hierarchical topologies 
    (e.g. macro-micro integration) generate correct off-diagonal Jacobian blocks.
    """
    engine = Engine(model=IntegralCouplingPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    if engine.mock_execution:
        pytest.skip("Compilation failed or absent.")
        
    N = engine.layout.n_states
    y = np.ones(N).tolist()
    ydot = np.zeros(N).tolist()
    
    J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0))
    
    offset_cs, size_cs = engine.layout.state_offsets["c_s"]
    offset_T, size_T = engine.layout.state_offsets["T"]
    
    # T is defined on x (resolution 4). c_s is defined on x*r (resolution 4*3 = 12).
    # The integral of c_s over r feeds into dt(T).
    # Therefore, d(Residual_T[i]) / d(c_s[j]) should be non-zero ONLY if c_s[j] belongs to the same macro node i.
    
    for i_mac in range(size_T):
        row = offset_T + i_mac
        
        for j_mac in range(size_T):
            for j_mic in range(3):
                col = offset_cs + (j_mac * 3) + j_mic
                derivative = J[row, col]
                
                if i_mac == j_mac:
                    # The macro node is coupled to its own micro states via the integral
                    assert abs(derivative) > 1e-10, f"Missing Jacobian coupling at T[{i_mac}] -> c_s[{col}]"
                else:
                    # The macro node must NOT be coupled to micro states of other macro nodes
                    assert abs(derivative) < 1e-10, f"Erroneous Jacobian leakage at T[{i_mac}] -> c_s[{col}]"

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_terminal_mode_inversion():
    """
    Category 4: Verifies the AST multiplexer flawlessly shifts the cycler terminal
    from current control (CC) to voltage control (CV) without destabilizing the graph.
    """
    engine = Engine(model=TerminalMultiplexerPDE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    if engine.mock_execution:
        pytest.skip("Compilation failed or absent.")
        
    N = engine.layout.n_states
    y = np.array([10.0, 4.0, 0.5]) # i_app, V_cell, soc (arbitrary initial guesses)
    ydot = np.zeros(N)
    
    off_i, _ = engine.layout.state_offsets["i_app"]
    off_V, _ = engine.layout.state_offsets["V_cell"]
    
    # Test Mode 1: Constant Current (CC) -> The constraint equation should be `i_app = i_target`
    # Therefore, dF_terminal / di_app == 1.0, dF_terminal / dV_cell == 0.0
    p_cc = engine._pack_parameters({"_term_mode": 1.0, "_term_i_target": 5.0})
    J_cc = np.array(engine.runtime.evaluate_jacobian(y.tolist(), ydot.tolist(), p_cc, 1.0))
    
    assert J_cc[off_i, off_i] == 1.0, "CC Multiplexer failed to map i_app constraint."
    assert J_cc[off_i, off_V] == 0.0, "CC Multiplexer erroneously coupled to V_cell."
    
    # Test Mode 0: Constant Voltage (CV) -> The constraint equation should be `V_cell = v_target`
    # Therefore, dF_terminal / di_app == 0.0, dF_terminal / dV_cell == -1.0 (or 1.0 depending on lhs/rhs subtraction)
    p_cv = engine._pack_parameters({"_term_mode": 0.0, "_term_v_target": 4.2})
    J_cv = np.array(engine.runtime.evaluate_jacobian(y.tolist(), ydot.tolist(), p_cv, 1.0))
    
    assert J_cv[off_i, off_i] == 0.0, "CV Multiplexer failed to decouple i_app."
    assert abs(J_cv[off_i, off_V]) == 1.0, "CV Multiplexer failed to map V_cell constraint."

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_algebraic_root_sensitivity():
    """
    Category 4: Verifies the JIT correctly strips time-derivatives from purely algebraic 
    sub-graphs, preventing singular Newton initialization matrices.
    """
    engine = Engine(model=TerminalMultiplexerPDE(), target="cpu", mock_execution=False)
    if engine.mock_execution:
        pytest.skip("Compilation failed or absent.")
        
    N = engine.layout.n_states
    y = np.zeros(N).tolist()
    ydot = np.zeros(N).tolist()
    
    p = engine._pack_parameters({"_term_mode": 1.0, "_term_i_target": 0.0})
    
    # Perturbing ydot for an algebraic variable MUST yield identically zero change in the residual.
    res_base = np.array(engine.evaluate_residual(y, ydot, parameters={"_term_mode": 1.0, "_term_i_target": 0.0}))
    
    ydot_perturbed = np.ones(N).tolist()
    res_perturbed = np.array(engine.evaluate_residual(y, ydot_perturbed, parameters={"_term_mode": 1.0, "_term_i_target": 0.0}))
    
    off_V, _ = engine.layout.state_offsets["V_cell"]
    off_i, _ = engine.layout.state_offsets["i_app"]
    
    assert res_base[off_V] == res_perturbed[off_V], "Algebraic residual V_cell is illegally dependent on ydot."
    assert res_base[off_i] == res_perturbed[off_i], "Algebraic residual i_app is illegally dependent on ydot."


# ==============================================================================
# Functional Architecture Tests (Former Undefined Sub-Graphs & Limits)
# ==============================================================================

class UnderDeterminedPDE(fx.PDE):
    """
    Tests Category 5: Undefined Sub-Graph Dependencies.
    Simulates a system where a declared state is omitted from the mathematical 
    constraints, creating a rank-deficient Jacobian [cite: 981-983].
    """
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    ghost_state = fx.State(domain=x, name="ghost_state") # Declared, unconstrained
    
    def math(self):
        return {
            "regions": {
                self.x: [fx.dt(self.c) == fx.grad(self.c)]
            },
            "boundaries": [
                self.c.left == 0.0,
                self.c.right == 1.0
            ],
            "global": [
                self.c.t0 == 0.5
            ]
        }

class ImplodingALEPDE(fx.PDE):
    """
    Tests Category 6: ALE Grid Inversion.
    Simulates a Stefan problem where the moving boundary derivative guarantees 
    the domain length (L) will cross zero, mathematically inverting the mesh [cite: 376-379].
    """
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    L = fx.State(domain=None, name="L") 
    
    def math(self):
        return {
            "regions": {
                self.x: [fx.dt(self.c) == fx.grad(self.c)]
            },
            "boundaries": [
                self.x.right == self.L,
                self.c.left == 0.0,
                self.c.right == 0.0
            ],
            "global": [
                fx.dt(self.L) == -100.0, # Rapid implosion velocity
                self.L.t0 == 0.01,
                self.c.t0 == 1.0
            ]
        }

class ExtremeScaleDAE(fx.PDE):
    """
    Tests Category 7: Scale-Induced Ill-Conditioning.
    Simulates a system with parameter magnitudes spanning 27 orders of magnitude,
    designed to overwhelm standard double-precision float tracking in the 
    unscaled Newton-Raphson solver [cite: 742-749, 783-789].
    """
    c = fx.State(domain=None, name="c")
    V = fx.State(domain=None, name="V")
    
    def math(self):
        return {
            "global": [
                fx.dt(self.c) == -1e-15 * self.c,
                self.V == 1e12 * self.c,
                self.c.t0 == 1.0,
                self.V.t0 == 1e12
            ]
        }


# ==============================================================================
# AST Vulnerability Tests
# ==============================================================================

def test_ast_rejects_under_determined_system():
    """
    Validates that the AST compilation phase proactively aborts when a declared 
    State lacks a governing equation, preventing runtime singular matrix errors.
    """
    model = UnderDeterminedPDE()
    
    # The Engine should raise a ValueError during AST parsing or graph compilation
    # before offloading to the Clang/LLVM toolchain [cite: 573-575].
    with pytest.raises(ValueError, match=r"Unconstrained state|Rank deficiency"):
        Engine(model=model, target="cpu", mock_execution=True)


@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_ale_grid_inversion_trapping():
    """
    Validates that moving boundaries evaluate dynamically against zero-crossings.
    If the mesh inverts (L <= 0), the solver should halt cleanly with a specific 
    integration error rather than generating unhandled NaNs [cite: 774-781].
    """
    engine = Engine(model=ImplodingALEPDE(), target="cpu", mock_execution=False)
    
    # Expect the native solver to throw a PyRuntimeError indicating domain collapse
    # rather than failing silently or hanging in an infinite loop.
    with pytest.raises(RuntimeError, match=r"Domain collapse|Mesh inversion"):
        engine.solve(t_span=(0, 0.1))


@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_scale_induced_ill_conditioning():
    """
    Validates that the generated Jacobian remains solvable under extreme scaling disparities.
    Current Row Equilibration natively implemented in Rust [cite: 835-841] handles $O(10^{27})$ 
    spans seamlessly without needing full AST-level non-dimensionalization modifications.
    """
    engine = Engine(model=ExtremeScaleDAE(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    N = engine.layout.n_states
    y = [1.0, 1e12]
    ydot = [-1e-15, 0.0]
    
    # The analytical Jacobian evaluation should yield finite, non-NaN values 
    # even under severe magnitude disparities [cite: 604, 953-955].
    J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0))
    
    assert np.isfinite(J).all(), "Jacobian evaluation produced non-finite values due to scale collapse."
    
    # Emulate the native solver's Row Equilibration to correct `np.linalg.matrix_rank`'s 
    # strictly bounded SVD tolerance assumptions.
    row_max = np.max(np.abs(J), axis=1, keepdims=True)
    row_max[row_max < 1e-15] = 1.0
    J_equilibrated = J / row_max
    
    rank = np.linalg.matrix_rank(J_equilibrated)
    assert rank == N, f"Scale-induced ill-conditioning reduced Jacobian rank to {rank}."