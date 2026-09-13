"""
Central Compiler Orchestration Pipeline.

Directs the unidirectional staged compilation flow:
  Stage 1: Intent Capture (Python AST)
  Stage 2: Topology Verification & Semantic Parsing (Middle-End)
  Stage 3: Normalization & Lowering to Math IR (Backend)
  Stage 4: FVM Discretization & Mechanical C++ Codegen
  Stage 5: Static AD Analysis via CPR Graph Coloring
"""

from typing import Dict, Any, List, Optional
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._2_middle_end.verification import verify_manifold
from ion_flux.compiler._3_backend.normalization import NormalizationPass
from ion_flux.compiler._3_backend.cpr_orchestrator import compute_cpr
from ion_flux.compiler._3_backend.math_ir import MathSystem, extract_math_state_names
from ion_flux.compiler._4_codegen.builder import generate_cpp


class Compiler:
    """The central orchestrator for the Staged Lowering Pipeline."""

    @staticmethod
    def compile(
        ast_payload: Dict[str, Any],
        layout: Any,
        states: List[Any],
        observables: List[Any],
        target: str,
        jacobian_bandwidth: Optional[int] = None
    ) -> Dict[str, Any]:
        # --- Stage 2: Topological Validation & Semantic Resolution ---
        topo = TopologyAnalyzer(ast_payload.get("domains", {}))
        verify_manifold(ast_payload)
        semantic_ctx = SemanticContext(ast_payload)

        state_map = {s.name: s for s in states}
        state_map.update({o.name: o for o in observables})

        # --- Stage 3: Normalization (AST Dict -> Math IR) ---
        norm_pass = NormalizationPass(ast_payload, topo, semantic_ctx, state_map, layout)
        math_sys: MathSystem = norm_pass.lower_to_math_ir()

        Compiler._verify_system_rank(math_sys, layout)

        if jacobian_bandwidth is None:
            jacobian_bandwidth = Compiler._compute_symbolic_bandwidth(layout, states, math_sys)

        # --- Stage 4: Discretization & Codegen (Math IR -> Compute IR -> C++) ---
        cpp_source, eq_stmts = generate_cpp(
            math_sys=math_sys,
            layout=layout,
            topo=topo,
            semantic_ctx=semantic_ctx,
            state_map=state_map,
            target=target
        )

        # --- Stage 5: CPR Sparsity Analysis on Compute IR ---
        cpr_cache = compute_cpr(eq_stmts, layout, jacobian_bandwidth)

        return {
            "cpp_source": cpp_source,
            "cpr_cache": cpr_cache,
            "ast_payload": ast_payload,
            "topo": topo,
            "jacobian_bandwidth": jacobian_bandwidth,
            "math_sys": math_sys
        }

    @staticmethod
    def _verify_system_rank(math_sys: MathSystem, layout: Any) -> None:
        """Validates that all declared states have at least one governing equation."""
        targeted_states = {eq.state_name for eq in math_sys.equations}
        for state_name in layout.state_offsets.keys():
            if state_name not in targeted_states:
                raise ValueError(f"Unconstrained state detected: '{state_name}'. Rank deficiency in system.")

    @staticmethod
    def _compute_symbolic_bandwidth(layout: Any, states: List[Any], math_sys: MathSystem) -> int:
        """
        Determines structural Jacobian bandwidth from the Math IR.
        Returns:
            -1: Unstructured CSR mesh (Matrix-Free GMRES).
             0: Dense coupling or dynamic ALE grid kinematics.
            >0: Banded system width.
        """
        if any(getattr(s.domain, "coord_sys", "") == "unstructured" for s in states):
            return -1

        if math_sys.dynamic_domain_bindings:
            return 0

        max_bw = 0
        for eq in math_sys.equations:
            target_state = eq.state_name
            if target_state not in layout.state_offsets:
                continue

            off_t, size_t = layout.state_offsets[target_state]
            if size_t > 1:
                max_bw = max(max_bw, 2)

            deps = extract_math_state_names(eq.lhs) + extract_math_state_names(eq.rhs)
            for dep in deps:
                if dep not in layout.state_offsets:
                    continue
                off_d, _ = layout.state_offsets[dep]
                if abs(off_t - off_d) > 0:
                    return 0

        return max_bw if max_bw > 0 else 0