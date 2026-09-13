"""
Property-Based Fuzzing Suite

Aggressively stresses the ion_flux compiler pipeline, topology analyzer,
CPR graph coloring, and native execution runtime using randomized, adversarial inputs.

Enforces four key invariants:
1. Spatial Lowering Resilience: ASTs with nested differentials, integrals, and boundary
   conditions must either lower cleanly to Math IR and C++ or raise expected validation errors.
2. Topological Verification: Manifold slicing must reject invalid boundary topologies
   matching the compiler's strict geometry tolerance (1e-12).
3. CPR Sparsity Recovery: Color-scheduled Forward JVPs and Reverse VJPs must exactly
   reconstruct hybrid sparse/dense Jacobian structures.
4. Memory Safety & Rollback: Extreme time steps and non-linear solver divergence must
   not leak NaNs or corrupt state arrays across checkpoint/restore boundaries.
"""

import os
import sys
import shutil
import platform
import pytest
import numpy as np
from hypothesis import given, settings, strategies as st

import ion_flux as fx
from ion_flux.compiler._1_frontend.nodes import Scalar, BinaryOp, UnaryOp
from ion_flux.compiler._2_middle_end.memory_layout import MemoryLayout
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._2_middle_end.verification import verify_manifold, TopologicalError
from ion_flux.compiler._3_backend.normalization import NormalizationPass
from ion_flux.compiler._3_backend.math_ir import MathSystem
from ion_flux.compiler._3_backend.cpr_coloring import HybridGraphColorer
from ion_flux.compiler._4_codegen.builder import generate_cpp

# Ensure models directory is in path for E2E tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))


# ==============================================================================
# Environment Capabilities
# ==============================================================================

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac


try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")
REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)


# ==============================================================================
# SECTION 1: AST Structural & Spatial Lowering Resilience
# ==============================================================================

D_MACRO = fx.Domain(bounds=(0, 1), resolution=5, name="d_macro")
D_MICRO = fx.Domain(bounds=(0, 1), resolution=4, coord_sys="spherical", name="d_micro")
C_STATE = fx.State(domain=D_MACRO * D_MICRO, name="c_fuzz")
T_PARAM = fx.Parameter(default=1.0, name="t_param")


def ast_expression_strategy():
    """Generates deeply nested, randomized AST expressions with spatial operators."""
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
    """Wraps randomized AST expressions in a PDE with spatial boundary definitions."""
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
                self.c_fuzz: fx.Piecewise({
                    self.d_macro: fx.dt(self.c_fuzz) == self.random_ast
                })
            },
            "boundaries": {
                flux: {"left": 0.0, "right": 1.0}
            },
            "initial_conditions": {self.c_fuzz: 1.0}
        }


@settings(max_examples=100, deadline=None)
@given(random_ast=ast_expression_strategy())
def test_fuzz_ast_spatial_lowering_resilience(random_ast):
    """
    PROBE: Drives randomized spatial ASTs through the lowering pipeline:
           AST -> Math IR (MathSystem) -> Compute IR -> C++ strings.
    INVARIANT: The compiler must either cleanly lower the AST or reject it with expected
               domain errors (TopologicalError, ValueError). Unhandled internal exceptions
               (KeyError, IndexError, TypeError) indicate structural defects.
    """
    model = MockFuzzPDE(random_ast)
    ast_payload = model.ast()
    
    layout = MemoryLayout(
        states=[model.c_fuzz], 
        parameters=[model.t_param], 
        all_domains=[model.d_macro, model.d_micro]
    )
    
    try:
        # Step 1: Middle-End Topology & Semantic Analysis
        topo = TopologyAnalyzer(ast_payload.get("domains", {}))
        verify_manifold(ast_payload)
        semantic_ctx = SemanticContext(ast_payload)
        state_map = {model.c_fuzz.name: model.c_fuzz}

        # Step 2: Normalization (AST -> Math IR)
        norm_pass = NormalizationPass(ast_payload, topo, semantic_ctx, state_map, layout)
        math_sys: MathSystem = norm_pass.lower_to_math_ir()

        # Step 3: FVM Discretization & Codegen (Math IR -> Compute IR -> C++)
        cpp_str, eq_stmts = generate_cpp(
            math_sys=math_sys,
            layout=layout,
            topo=topo,
            semantic_ctx=semantic_ctx,
            state_map=state_map,
            target="cpu"
        )
        
        assert isinstance(cpp_str, str)
        assert len(cpp_str) > 0
        assert isinstance(eq_stmts, list)
        
    except TopologicalError:
        pass  # Expected rejection of physically invalid domain configurations
    except ValueError as e:
        err_msg = str(e)
        if "Unknown IR Node" in err_msg or "Math Leak" in err_msg:
            pytest.fail(f"Compiler AST generation defect detected: {e}\nAST: {random_ast}")
        # Other ValueErrors (e.g. unconstrained states) represent valid frontend rejections
    except Exception as e:
        pytest.fail(f"Unhandled structural compiler crash ({type(e).__name__}): {e}\nAST: {random_ast}")


# ==============================================================================
# SECTION 2: Topological Grid Slicing & Verification
# ==============================================================================

@st.composite
def topological_manifold_strategy(draw):
    """Generates random sub-regions, intentionally injecting clipping and boundary gaps."""
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
    """Reference check verifying whether sub-regions partition the parent domain without gaps."""
    if not regions:
        return False
    regions_sorted = sorted(regions, key=lambda r: r["start_idx"])
    
    current_idx = 0
    current_bound = p_bounds[0]
    
    for r in regions_sorted:
        if r["start_idx"] != current_idx:
            return False
        if abs(r["bounds"][0] - current_bound) > 1e-12:
            return False
        current_idx += r["resolution"]
        current_bound = r["bounds"][1]
        
    if current_idx != p_res:
        return False
    if abs(current_bound - p_bounds[1]) > 1e-12:
        return False
    return True


@settings(max_examples=100, deadline=None)
@given(manifold_data=topological_manifold_strategy())
def test_fuzz_verify_manifold_rejections(manifold_data):
    """
    PROBE: Feeds randomized regional partitions into verify_manifold.
    INVARIANT: Must raise TopologicalError if and only if regions fail the exact tiling check.
    """
    _, regions, coord_sys = manifold_data
    
    p_bounds, p_res = (0.0, 10.0), 100
    is_mathematically_valid = _is_valid_tiling(regions, p_bounds, p_res)
    
    domains = {"cell": {"bounds": p_bounds, "resolution": p_res, "coord_sys": coord_sys, "type": "standard"}}
    for r in regions:
        domains[r["name"]] = r
        
    eq_payload = {
        "state": "c", 
        "type": "standard", 
        "eq": {"type": "UnaryOp", "op": "grad", "child": {"type": "State", "name": "c"}}
    }
    bc_payload = {
        "type": "dirichlet", 
        "state": "c", 
        "bcs": {"left": {"type": "Scalar", "value": 0.0}, "right": {"type": "Scalar", "value": 0.0}}
    }
    
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
    """Generates banded sparse Jacobian matrices interspersed with dense constraint rows."""
    N = draw(st.integers(20, 100))
    band = draw(st.integers(0, 4))
    
    triplets = set()
    J_true = np.zeros((N, N))
    
    for i in range(N):
        for j in range(max(0, i - band), min(N, i + band + 1)):
            val = draw(st.floats(0.1, 10.0))
            triplets.add((i, j))
            J_true[i, j] = val
            
    dense_rows = draw(st.lists(st.integers(0, N - 1), min_size=0, max_size=3, unique=True))
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
    PROBE: Validates CPR Welsh-Powell column-intersection graph coloring.
    INVARIANT: Sparse columns colored together must not collide in the same row. Dense rows
               must be isolated into the dedicated VJP sweep. Reconstructed J must equal J_true.
    """
    N, triplets, J_true = graph_data
    
    colorer = HybridGraphColorer(n_states=N, triplets=triplets, dense_threshold=15)
    J_reconstructed = np.zeros((N, N))
    
    # Forward-Mode AD JVP Sweeps (Sparse Bulk)
    for c_idx, seed_vector in enumerate(colorer.color_seeds):
        v = np.array(seed_vector)
        jvp_out = J_true @ v
        
        for row, col in colorer.sparse_triplets:
            if colorer.color_map[col] == c_idx:
                J_reconstructed[row, col] = jvp_out[row]
                
    # Reverse-Mode AD VJP Passes (Dense Arrowhead Rows)
    for r in colorer.dense_rows:
        lam_vjp = np.zeros(N)
        lam_vjp[r] = 1.0
        
        dy_out = lam_vjp @ J_true
        for col in range(N):
            val = dy_out[col]
            if abs(val) > 1e-16:
                J_reconstructed[r, col] = val
                
    np.testing.assert_allclose(
        J_reconstructed, J_true, atol=1e-12,
        err_msg="CPR Reconstruction Failed: Color collision or dense row amputation error."
    )


# ==============================================================================
# SECTION 4: Native Session Memory Safety (Stiff Non-Linear Integration)
# ==============================================================================

class StiffNonLinearDAE(fx.PDE):
    """Stiff nonlinear DAE combining rapid spatial diffusion with logarithmic algebraic constraints."""
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


# Lazily instantiate the test engine only when the native runtime is present
_STIFF_ENGINE = (
    fx.Engine(model=StiffNonLinearDAE(), target="cpu", mock_execution=False)
    if (_has_compiler() and RUST_FFI_AVAILABLE) else None
)


@st.composite
def session_action_strategy(draw):
    """Emits random session commands with time steps spanning 1e-12 to 1e6."""
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
    PROBE: Applies aggressive parameter steps and time increments to the native Rust solver.
    INVARIANT: Divergence or step rejections are acceptable, but solver workspace rollback
               must maintain finite, non-corrupted state arrays without NaN leakage.
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
            err_str = str(e).lower()
            assert "divergence" in err_str or "convergence" in err_str or "crash" in err_str, \
                f"Unexpected Native Engine exception: {e}"
            step_crashed = True
            
        # Verify workspace rollback cleared speculative NaNs from state buffers
        c_arr = session.get_array("c")
        v_arr = session.get_array("v")
        
        assert np.all(np.isfinite(c_arr)), f"State array 'c' contains non-finite values.\nActions: {actions}"
        assert np.all(np.isfinite(v_arr)), f"Algebraic array 'v' contains non-finite values.\nActions: {actions}"

        if step_crashed:
            break


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])