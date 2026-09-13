"""
Codegen Builder.

Coordinates FVM spatial discretization and mechanical C++ emission from Math IR.
Does not perform semantic normalization or AST manipulation.
"""

from typing import List, Dict, Any, Tuple
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._3_backend.math_ir import MathSystem
from ion_flux.compiler._3_backend.discretizer import FVMDiscretizer
from ion_flux.compiler._4_codegen.compute_ir import Stmt
from ion_flux.compiler._4_codegen.cpp_emitter import CppEmitter
from ion_flux.compiler._4_codegen.templates import generate_cpp_skeleton


def generate_cpp(
    math_sys: MathSystem,
    layout: Any,
    topo: TopologyAnalyzer,
    semantic_ctx: SemanticContext,
    state_map: Dict[str, Any],
    target: str = "cpu"
) -> Tuple[str, List[Stmt]]:
    """
    Lowers Math IR to Compute IR, stringifies statements to C++, and wraps them in
    the native execution and Enzyme AD template.

    Args:
        math_sys: Strongly typed continuum Math IR system.
        layout: Contiguous memory offset and geometry stride tables.
        topo: Coordinate systems, manifolds, and composite domain strides.
        semantic_ctx: Pre-processed boundary condition lookups.
        state_map: Mapping of state and observable names to frontend objects.
        target: Compilation target flag (e.g., 'cpu:serial', 'cpu:omp').

    Returns:
        Tuple of (cpp_source_code, compute_ir_equation_statements).
    """
    discretizer = FVMDiscretizer(layout, topo, semantic_ctx, state_map, target)
    l_phys_stmts, eq_stmts, obs_stmts = discretizer.discretize_system(math_sys)

    emitter = CppEmitter()
    body_str = "\n    ".join(emitter.emit(stmt) for stmt in (l_phys_stmts + eq_stmts))
    obs_body_str = "\n    ".join(emitter.emit(stmt) for stmt in (l_phys_stmts + obs_stmts))

    cpp_str = generate_cpp_skeleton(
        n_states=layout.n_states,
        n_params=layout.n_params,
        n_obs=layout.n_obs,
        body=body_str,
        obs_body=obs_body_str
    )

    return cpp_str, eq_stmts