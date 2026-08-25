from typing import Dict, Any, List
from ion_flux.stage2_compiler._1_analysis.topology import TopologyAnalyzer
from ion_flux.stage2_compiler._1_analysis.semantics import SemanticContext
from ion_flux.stage2_compiler._1_analysis.ast_utils import extract_div_child

class NormalizationPass:
    """
    Structural AST-to-AST transformation.
    Unrolls syntactic sugar like Piecewise domains into explicit, independent
    equations locked to specific sub-domains, dramatically simplifying downstream C++ emission.
    """
    def __init__(self, ast_payload: Dict[str, Any], topo: TopologyAnalyzer, semantic_ctx: SemanticContext, state_map: Dict[str, Any]):
        self.ast_payload = ast_payload
        self.topo = topo
        self.semantic_ctx = semantic_ctx
        self.state_map = state_map

    def run(self) -> Dict[str, Any]:
        normalized = self.ast_payload.copy()
        normalized["equations"] = self._normalize_equations(self.ast_payload.get("equations", []))
        normalized["observables"] = self._normalize_equations(self.ast_payload.get("observables", []), is_obs=True)
        return normalized

    def _normalize_equations(self, equations: List[Dict[str, Any]], is_obs: bool = False) -> List[Dict[str, Any]]:
        flat_eqs = []
        
        for eq_data in equations:
            state_name = eq_data["state"]
            d_bcs = self.semantic_ctx.get_dirichlet_bc(state_name)
            d_name = getattr(self.state_map.get(state_name), "domain", None)
            last_axis = self.topo.get_axes(d_name.name)[-1] if d_name else None
            
            if eq_data["type"] == "piecewise":
                regions = eq_data["regions"]
                region_divs = {r["domain"]: extract_div_child(r["eq"]) for r in regions}
                
                for reg in regions:
                    b_axis = self.topo.get_base_axis(reg["domain"])
                    r_start = reg["start_idx"]
                    r_res = reg["end_idx"] - reg["start_idx"]
                    
                    # Shrink the bulk calculation bounds if a Dirichlet boundary overrides the edge node
                    if d_bcs and last_axis and self.topo.get_base_axis(last_axis) == b_axis:
                        domain_start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
                        domain_res = self.topo.domains.get(last_axis, {}).get("resolution", 1)
                        
                        if "left" in d_bcs and r_start == domain_start:
                            r_start += 1
                            r_res -= 1
                        if "right" in d_bcs and r_start + r_res == domain_start + domain_res:
                            r_res -= 1
                            
                    if r_res > 0:
                        flat_eqs.append({
                            "state": state_name,
                            "type": "standard",
                            "eq": reg["eq"],
                            "bounds_override": {b_axis: (r_start, r_res)},
                            "piecewise_info": {
                                "regions": regions,
                                "region_divs": region_divs,
                                "current_region": reg
                            }
                        })
            else:
                bounds_override = {}
                if d_bcs and last_axis:
                    start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
                    res = self.topo.domains.get(last_axis, {}).get("resolution", 1)
                    b_axis = self.topo.get_base_axis(last_axis)
                    
                    if "left" in d_bcs:
                        start += 1
                        res -= 1
                    if "right" in d_bcs:
                        res -= 1
                    if res > 0:
                        bounds_override[b_axis] = (start, res)
                        
                # Prevent emitting a bulk equation if Dirichlet boundary entirely consumes a 1-node domain
                is_valid = True
                if d_bcs and last_axis and bounds_override.get(self.topo.get_base_axis(last_axis), (0, 1))[1] <= 0:
                    is_valid = False
                    
                if is_valid:
                    eq_out = {
                        "state": state_name,
                        "type": "standard",
                        "eq": eq_data["eq"]
                    }
                    if bounds_override:
                        eq_out["bounds_override"] = bounds_override
                    flat_eqs.append(eq_out)
        
        return flat_eqs