from typing import Dict, Any, List, Optional
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._3_backend.normalization import NormalizationPass
from ion_flux.compiler._2_middle_end.verification import verify_manifold
from ion_flux.compiler._4_codegen.builder import generate_cpp
from ion_flux.compiler._3_backend.cpr_orchestrator import compute_cpr

class Compiler:
    """
    The central orchestrator for the Staged Lowering Pipeline.
    Transforms ASTs to Math IR, discretizes to Compute IR, performs static CPR
    sparsity analysis, and emits C++ strings and CPR schedules.
    """
    @staticmethod
    def compile(ast_payload: Dict[str, Any], layout: Any, states: List[Any], 
                observables: List[Any], target: str, jacobian_bandwidth: Optional[int]) -> Dict[str, Any]:
        
        topo = TopologyAnalyzer(ast_payload.get("domains", {}))
        semantic_ctx = SemanticContext(ast_payload)
        state_map = {s.name: s for s in states}
        state_map.update({o.name: o for o in observables})
        
        ast_payload = NormalizationPass(ast_payload, topo, semantic_ctx, state_map, layout).run()
        verify_manifold(ast_payload)
        
        # Guard: Check rank deficiencies prior to static analysis
        targeted_states = {eq["state"] for eq in ast_payload.get("equations", [])}
        for state_name in layout.state_offsets.keys():
            if state_name not in targeted_states:
                raise ValueError(f"Unconstrained state detected: '{state_name}'. Rank deficiency in system.")

        if jacobian_bandwidth is None:
            jacobian_bandwidth = Compiler._compute_symbolic_bandwidth(layout, states, ast_payload)

        cpp_source, eq_stmts = generate_cpp(ast_payload, layout, states, observables, target)
        cpr_cache = compute_cpr(eq_stmts, layout, jacobian_bandwidth)
        
        return {
            "cpp_source": cpp_source,
            "cpr_cache": cpr_cache,
            "ast_payload": ast_payload,
            "topo": topo,
            "jacobian_bandwidth": jacobian_bandwidth
        }

    @staticmethod
    def _compute_symbolic_bandwidth(layout: Any, states: List[Any], ast_payload: Dict[str, Any]) -> int:
        from ion_flux.compiler._2_middle_end.ast_utils import extract_state_names
        
        if any(getattr(s.domain, "coord_sys", "") == "unstructured" for s in states): 
            return -1
        
        max_bw = 0
        def check_dependencies(target_state: str, node: Dict[str, Any]) -> int:
            nonlocal max_bw
            if target_state not in layout.state_offsets: return max_bw
            off_t, size_t = layout.state_offsets[target_state]
            if size_t > 1: max_bw = max(max_bw, 2)
            
            deps = extract_state_names(node)
            for d in deps:
                if d not in layout.state_offsets: continue
                off_d, _ = layout.state_offsets[d]
                if abs(off_t - off_d) > 0: return 0 
            return max_bw

        for bc_data in ast_payload.get("boundaries", []):
            if bc_data.get("type") == "moving_domain": return 0

        for eq_data in ast_payload.get("equations", []):
            target_state = eq_data["state"]
            if eq_data["type"] == "piecewise":
                for reg in eq_data["regions"]:
                    if check_dependencies(target_state, reg["eq"]) == 0: return 0
            else:
                if check_dependencies(target_state, eq_data["eq"]) == 0: return 0
                
        return max_bw if max_bw > 0 else 0