"""
Codegen Builder.

Coordinates the FVM Discretizer and CppEmitter to generate C++ source code.
"""

from typing import List, Dict, Any, Tuple
from ion_flux.stage2_compiler._1_analysis.semantics import SemanticContext
from ion_flux.stage2_compiler._1_analysis.topology import TopologyAnalyzer
from ion_flux.stage2_compiler._2_lowering.normalization import NormalizationPass
from ion_flux.stage2_compiler._2_lowering.discretizer import FVMDiscretizer
from ion_flux.stage2_compiler._4_codegen.cpp_emitter import CppEmitter
from ion_flux.stage2_compiler._4_codegen.templates import generate_cpp_skeleton


def generate_cpp(ast_payload: Dict[str, Any], layout: Any, states: List[Any], 
                 observables: List[Any], target: str = "cpu") -> Tuple[str, List[Any]]:
    """Generates the residual and observable C++ functions from Math IR."""
    topo = TopologyAnalyzer(ast_payload.get("domains", {}))
    semantic_ctx = SemanticContext(ast_payload)
    emitter = CppEmitter()

    state_map = {s.name: s for s in states}
    state_map.update({o.name: o for o in observables})

    norm_pass = NormalizationPass(ast_payload, topo, semantic_ctx, state_map, layout)
    math_sys = norm_pass.lower_to_math_ir()

    discretizer = FVMDiscretizer(layout, topo, semantic_ctx, state_map, target)
    l_phys_stmts, eq_stmts, obs_stmts = discretizer.discretize_system(math_sys)

    body_str = "\n    ".join(emitter.emit(stmt) for stmt in (l_phys_stmts + eq_stmts))
    obs_body_str = "\n    ".join(emitter.emit(stmt) for stmt in (l_phys_stmts + obs_stmts))

    cpp_str = generate_cpp_skeleton(layout.n_states, layout.n_params, layout.n_obs, body_str, obs_body_str)
    return cpp_str, eq_stmts