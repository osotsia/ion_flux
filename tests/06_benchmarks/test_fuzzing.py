"""
Property-Based Fuzzing Suite

This suite aggressively probes the boundaries of the ion_flux AST compiler, 
topology analyzer, graph coloring algorithms, and native execution engine using 
thousands of randomized, adversarial inputs.

It enforces strict mathematical and structural invariants:
1. Compiler Resilience: Whitelists specific domain/validation rejections but traps 
   internal compiler exceptions (IndexError, KeyError, etc.) caused by deeply 
   nested spatial mathematics (integrals, piecewise, boundaries).
2. Topological Verification: Constructs full AST payloads to trigger manifold closure 
   checks, aggressively injecting micro-epsilon floating point errors to test clipping limits.
3. CPR Graph Coloring: Simulates exact Reverse-Mode Vector-Jacobian Products (VJP) 
   to mathematically prove collision-free isolation of dense global matrices.
4. Native Session Integrity: Interleaves BDF checkpoint/restores with extreme time 
   steps, ensuring memory buffers never corrupt to NaN even upon severe non-linear divergence.
"""

import pytest
import numpy as np
import shutil
import platform
from hypothesis import given, settings, strategies as st

import ion_flux as fx
from ion_flux.stage1_dsl.nodes import Scalar, BinaryOp, UnaryOp
from ion_flux.stage2_compiler._1_analysis.memory_layout import MemoryLayout
from ion_flux.stage2_compiler._1_analysis.topology import TopologyAnalyzer
from ion_flux.stage2_compiler._1_analysis.semantics import SemanticContext
from ion_flux.stage2_compiler._2_lowering.normalization import NormalizationPass
from ion_flux.stage2_compiler._1_analysis.verification import verify_manifold, TopologicalError
from ion_flux.stage2_compiler._3_optimization.cpr_coloring import HybridGraphColorer
from ion_flux.stage2_compiler._4_codegen.builder import generate_cpp

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

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

# ==============================================================================
# SECTION 1: AST Structural & Spatial Lowering Resilience
# ==============================================================================

# Anchors for spatial AST node generation
D_MACRO = fx.Domain(bounds=(0, 1), resolution=5, name="d_macro")
D_MICRO = fx.Domain(bounds=(0, 1), resolution=4, coord_sys="spherical", name="d_micro")
C_STATE = fx.State(domain=D_MACRO * D_MICRO, name="c_fuzz")
T_PARAM = fx.Parameter(default=1.0, name="t_param")

def ast_expression_strategy():
    """
    Builds deeply nested, randomized mathematical expression trees.
    Incorporates topology-aware spatial operators (grad, div, integral, boundary)
    to comprehensively stress the C++ code generator's context routing.
    """
    base_nodes = st.one_of(
        st.builds(Scalar, st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        st.just(C_STATE),
        st.just(T_PARAM),
        st.just(D_MACRO.coords)
    )
    
    return st.recursive(
        base_nodes,
        lambda children: st.one_of(
            st.builds(lambda op, l, r: BinaryOp(op, l, r), 
                      st.sampled_from(["add", "sub", "mul", "div", "max", "min"]), 
                      children, children),
            st.builds(lambda op, c: UnaryOp(op, c), 
                      st.sampled_from(["exp", "sin", "cos", "abs", "neg"]), 
                      children),
            st.builds(lambda c: fx.grad(c, axis=D_MICRO), children),
            st.builds(lambda c: fx.div(c, axis=D_MICRO), children),
            st.builds(lambda c: fx.integral(c, over=D_MICRO), children),
            st.builds(lambda c: c.boundary("right", domain=D_MICRO), children),
        ),
        max_leaves=10
    )

class MockFuzzPDE(fx.PDE):
    """Wraps the randomized AST payload, explicitly injecting boundaries to trigger SemanticContext."""
    d_macro = D_MACRO
    d_micro = D_MICRO
    c_fuzz = C_STATE
    t_param = T_PARAM
    
    def __init__(self, random_ast):
        super().__init__()
        self.random_ast = random_ast
        
    def math(self):
        flux = fx.grad(self.c_fuzz, axis=self.d_micro)
        return {
            "equations": {
                # Wrap the fuzzed expression inside a Piecewise block to test regional context
                self.c_fuzz: fx.Piecewise({
                    self.d_macro: fx.dt(self.c_fuzz) == self.random_ast
                })
            },
            "boundaries": {
                # Triggers the SemanticContext Neumann parsing logic
                flux: {"left": 0.0, "right": 1.0}
            },
            "initial_conditions": {self.c_fuzz: 1.0}
        }

@settings(max_examples=100, deadline=None)
@given(random_ast=ast_expression_strategy())
def test_fuzz_ast_spatial_lowering_resilience(random_ast):
    """
    PROBE: Feeds chaotic, deeply nested spatial ASTs into the Middle-End Compiler.
    INVARIANT: The AST translator must cleanly lower the tree. We whitelist safe rejections 
    (Topological Errors, invalid user bounds), but explicitly catch and fail on internal 
    structural compiler crashes (KeyError, TypeError, IndexError) masking as logic flaws.
    """
    model = MockFuzzPDE(random_ast)
    ast_payload = model.ast()
    
    layout = MemoryLayout(states=[model.c_fuzz], parameters=[model.t_param], all_domains=[model.d_macro, model.d_micro])
    
    try:
        # Replicate the exact middleware pipeline from `_1_builder.py`
        topo = TopologyAnalyzer(ast_payload.get("domains", {}))
        semantic_ctx = SemanticContext(ast_payload)
        state_map = {model.c_fuzz.name: model.c_fuzz}
        
        ast_payload = NormalizationPass(ast_payload, topo, semantic_ctx, state_map).run()
        verify_manifold(ast_payload)
        
        cpp_str, _ = generate_cpp(ast_payload, layout, states=[model.c_fuzz], observables=[], target="cpu")
        
        assert isinstance(cpp_str, str)
        assert len(cpp_str) > 0
        
    except TopologicalError:
        pass  # Expected rejection of garbage physics topologies
    except ValueError as e:
        # Trap explicitly uncaught AST dialects and mathematical leaks
        err_msg = str(e)
        if "Unknown IR Node" in err_msg or "Math Leak" in err_msg:
            pytest.fail(f"Compiler AST generation bug detected: {e}\nAST: {random_ast}")
        # Other ValueErrors (e.g. "Unconstrained state detected") are valid user rejections
    except Exception as e:
        # ALL OTHER EXCEPTIONS ARE CRITICAL COMPILER BUGS
        pytest.fail(f"Unhandled structural compiler crash ({type(e).__name__}): {e}\nAST: {random_ast}")

# ==============================================================================
# SECTION 2: Topological Grid Slicing & Verification
# ==============================================================================

@st.composite
def topological_manifold_strategy(draw):
    """
    Generates random sub-regions, intentionally injecting both gross topological flaws 
    and micro-epsilon float discrepancies to push clipping boundaries.
    """
    parent_res = 100
    parent_bounds = (0.0, 10.0)
    coord_sys = draw(st.sampled_from(["cartesian", "spherical", "cylindrical"]))
    should_be_valid = draw(st.booleans())
    
    regions = []
    num_regions = draw(st.integers(1, 4))
    
    if should_be_valid:
        boundaries = sorted(draw(st.lists(
            st.integers(1, 99), min_size=num_regions-1, max_size=num_regions-1, unique=True
        ))) if num_regions > 1 else []
        indices = [0] + boundaries + [100]
        
        for i in range(len(indices)-1):
            res = indices[i+1] - indices[i]
            start_b = parent_bounds[0] + (indices[i] / parent_res) * (parent_bounds[1] - parent_bounds[0])
            end_b = parent_bounds[0] + (indices[i+1] / parent_res) * (parent_bounds[1] - parent_bounds[0])
            
            regions.append({
                "name": f"reg_{i}", "start_idx": indices[i], "resolution": res, 
                "bounds": (start_b, end_b), "type": "standard", "parent": "cell"
            })
    else:
        # Generate invalid boundaries: mix of gross structural gaps and machine-epsilon perturbations
        boundaries = sorted(draw(st.lists(
            st.integers(1, 99), min_size=num_regions-1, max_size=num_regions-1, unique=True
        ))) if num_regions > 1 else []
        indices = [0] + boundaries + [100]
        
        for i in range(len(indices)-1):
            res = indices[i+1] - indices[i]
            start_b = parent_bounds[0] + (indices[i] / parent_res) * (parent_bounds[1] - parent_bounds[0])
            end_b = parent_bounds[0] + (indices[i+1] / parent_res) * (parent_bounds[1] - parent_bounds[0])
            
            # Inject spatial tracking discrepancies
            if draw(st.booleans()):
                perturbation = draw(st.sampled_from([1e-16, -1e-16, 1e-10, -1e-10, 1.0, -1.0]))
                end_b += perturbation
                
            regions.append({
                "name": f"reg_{i}", "start_idx": indices[i], "resolution": res, 
                "bounds": (start_b, end_b), "type": "standard", "parent": "cell"
            })
            
    return should_be_valid, regions, coord_sys

def _is_valid_tiling(regions, p_bounds, p_res):
    """Ground truth oracle to determine if a set of regions perfectly tiles the parent."""
    if not regions: return False
    regions_sorted = sorted(regions, key=lambda r: r["start_idx"])
    
    current_idx = 0
    current_bound = p_bounds[0]
    
    for r in regions_sorted:
        if r["start_idx"] != current_idx: return False
        # The oracle MUST precisely match the compiler's strict 1e-12 geometry limit
        if abs(r["bounds"][0] - current_bound) > 1e-12: return False
        current_idx += r["resolution"]
        current_bound = r["bounds"][1]
        
    if current_idx != p_res: return False
    # The oracle MUST precisely match the compiler's strict 1e-12 geometry limit
    if abs(current_bound - p_bounds[1]) > 1e-12: return False
    return True

@settings(max_examples=100, deadline=None)
@given(manifold_data=topological_manifold_strategy())
def test_fuzz_verify_manifold_rejections(manifold_data):
    """
    PROBE: Feeds chaotic boundary geometries into the Manifold Verifier.
    INVARIANT: The Verifier must raise `TopologicalError` if and only if the 
    regions fail the perfect tiling test. Includes boundaries to ensure 
    `_verify_boundaries` mathematically closes the manifold securely.
    """
    should_be_valid_generated, regions, coord_sys = manifold_data
    
    p_bounds, p_res = (0.0, 10.0), 100
    is_mathematically_valid = _is_valid_tiling(regions, p_bounds, p_res)
    
    domains = {"cell": {"bounds": p_bounds, "resolution": p_res, "coord_sys": coord_sys, "type": "standard"}}
    for r in regions:
        domains[r["name"]] = r
        
    # Inject dummy states and boundaries to trigger the _verify_boundaries pass
    eq_payload = {"state": "c", "type": "standard", "eq": {"type": "UnaryOp", "op": "grad", "child": {"type": "State", "name": "c"}}}
    bc_payload = {"type": "dirichlet", "state": "c", "bcs": {"left": {"type": "Scalar", "value": 0.0}, "right": {"type": "Scalar", "value": 0.0}}}
    
    ast_payload = {
        "domains": domains, 
        "equations": [eq_payload], 
        "boundaries": [bc_payload]
    }
    
    if is_mathematically_valid:
        verify_manifold(ast_payload)
    else:
        with pytest.raises(TopologicalError):
            verify_manifold(ast_payload)

# ==============================================================================
# SECTION 3: CPR Graph Coloring (Hybrid Density Segregation)
# ==============================================================================

@st.composite
def structured_sparsity_strategy(draw):
    """
    Generates highly structured sparse Jacobian dependencies (e.g., Banded FVM elements) 
    interspersed with dense global constraint rows (Arrowheads).
    """
    N = draw(st.integers(20, 100))
    band = draw(st.integers(0, 4))
    
    triplets = set()
    J_true = np.zeros((N, N))
    
    # Generate banded bulk
    for i in range(N):
        for j in range(max(0, i - band), min(N, i + band + 1)):
            val = draw(st.floats(0.1, 10.0))
            triplets.add((i, j))
            J_true[i, j] = val
            
    # Inject dense arrowhead rows
    dense_rows = draw(st.lists(st.integers(0, N-1), min_size=0, max_size=3, unique=True))
    for r in dense_rows:
        for c in range(N):
            val = draw(st.floats(0.1, 10.0))
            triplets.add((r, c))
            J_true[r, c] = val
            
    return N, triplets, J_true

@settings(max_examples=50)
@given(graph_data=structured_sparsity_strategy())
def test_fuzz_cpr_jvp_reconstruction_exactness(graph_data):
    """
    PROBE: Validates CPR Welsh-Powell coloring by simulating an exact AD reconstruction.
    INVARIANT: Simulates Forward-Mode AD JVP sweeps for the sparse bulk, and Reverse-Mode 
    AD VJP passes for the isolated dense components. Perfect mathematical recovery of 
    the hybrid sparse matrix is required.
    """
    N, triplets, J_true = graph_data
    
    colorer = HybridGraphColorer(n_states=N, triplets=triplets, dense_threshold=15)
    J_reconstructed = np.zeros((N, N))
    
    # 1. Simulate Forward-Mode AD JVP Sweeps
    for c_idx, seed_vector in enumerate(colorer.color_seeds):
        v = np.array(seed_vector)
        jvp_out = J_true @ v
        
        for row, col in colorer.sparse_triplets:
            if colorer.color_map[col] == c_idx:
                J_reconstructed[row, col] = jvp_out[row]
                
    # 2. Simulate Reverse-Mode AD VJP passes for Arrowhead dense rows
    for r in colorer.dense_rows:
        # Construct the adjoint vector lambda mapping directly to the isolated row
        lam_vjp = np.zeros(N)
        lam_vjp[r] = 1.0
        
        # Simulated VJP: evaluate_vjp(..., lambda) yields lambda^T @ J
        dy_out = lam_vjp @ J_true
        
        # Re-scatter evaluating strictly non-zero tolerance
        for col in range(N):
            val = dy_out[col]
            if abs(val) > 1e-16:
                J_reconstructed[r, col] = val
                
    # 3. Assert Perfect Sparsity Recovery
    np.testing.assert_allclose(
        J_reconstructed, J_true, atol=1e-12,
        err_msg="CPR Reconstruction Failed! Color collision caused a JVP overlap, or "
                "the VJP failed to correctly amputate and map the dense Arrowhead row."
    )

# ==============================================================================
# SECTION 4: Native Session Memory Safety (Stiff Non-Linear Integration)
# ==============================================================================

class StiffNonLinearDAE(fx.PDE):
    """
    A stiff, highly non-linear model designed to fight the Newton-Raphson root finder. 
    Coupling rapid spatial diffusion with logarithmic algebraic constraints guarantees 
    extreme sensitivity to arbitrary parameter jumps.
    """
    x = fx.Domain(bounds=(0, 1), resolution=10, name="x")
    c = fx.State(domain=x, name="c")
    v = fx.State(domain=None, name="v")
    i_app = fx.Parameter(default=1.0, name="i_app")
    
    def math(self):
        flux = -fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux) - (self.c ** 3) + self.i_app,
                self.v: self.v == fx.log(fx.max(self.c.right, 1e-3))
            },
            "boundaries": {
                flux: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c: 1.0, self.v: 0.0
            }
        }

try:
    if _has_compiler() and RUST_FFI_AVAILABLE:
        _STIFF_ENGINE = fx.Engine(model=StiffNonLinearDAE(), target="cpu", mock_execution=False)
    else:
        _STIFF_ENGINE = None
except Exception:
    _STIFF_ENGINE = None

@st.composite
def session_action_strategy(draw):
    """
    Emits a sequence of aggressive session commands. 
    Time steps (dt) span 18 orders of magnitude (1e-12 to 1e6) to induce tolerance starvation.
    """
    action_type = draw(st.sampled_from(["STEP", "CHECKPOINT", "RESTORE"]))
    if action_type == "STEP":
        log_dt = draw(st.floats(min_value=-12.0, max_value=6.0))
        i_app = draw(st.floats(min_value=-100.0, max_value=100.0))
        return ("STEP", 10 ** log_dt, i_app)
    return (action_type, 0.0, 0.0)

@pytest.mark.skipif(_STIFF_ENGINE is None, reason="Requires Native Execution Environment.")
@settings(max_examples=50, deadline=None)
@given(actions=st.lists(session_action_strategy(), min_size=1, max_size=30))
def test_fuzz_ffi_stiff_nonlinear_stepping(actions):
    """
    PROBE: Rapidly steps the Native Rust BDF solver using randomized extreme time-steps, 
    violent parameter jumps, and constant history checkpointing/restorations.
    INVARIANT: We tolerate explicit Integration Rejections (divergence, thrashing). We 
    STRICTLY assert that failed or aborted integration sweeps DO NOT leak NaN garbage 
    into the observable user state arrays, maintaining hermetic memory integrity.
    """
    session = _STIFF_ENGINE.start_session()
    
    for action, val1, val2 in actions:
        if action == "CHECKPOINT":
            session.checkpoint()
            continue
        elif action == "RESTORE":
            session.restore()
            continue
            
        dt, i_app = val1, val2
        step_crashed = False
        
        try:
            session.step(dt, inputs={"i_app": i_app})
        except RuntimeError as e:
            # Tolerable rejection: Extreme parameter swings violate BDF tolerances or cause divergence.
            err_str = str(e).lower()
            assert "divergence" in err_str or "convergence" in err_str or "crash" in err_str, \
                f"Unexpected Native Engine exception: {e}"
            step_crashed = True
            
        # GUARANTEE STRICT ARRAY OBSERVABILITY (MASK LOCAL NANS).
        # By extracting the arrays *even when the solver crashes*, we prove that the
        # workspace rollback successfully cleared any NaNs injected by a speculative Newton step.
        c_arr = session.get_array("c")
        v_arr = session.get_array("v")
        
        assert np.all(np.isfinite(c_arr)), f"Solver leaked NaN into state array 'c' after step failure.\nActions: {actions}"
        assert np.all(np.isfinite(v_arr)), f"Solver leaked NaN into algebraic array 'v' after step failure.\nActions: {actions}"

        if step_crashed:
            break 

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])