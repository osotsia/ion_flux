import math
from typing import Dict, Any, Tuple, List
from ion_flux.runtime.manifest import ExecutableManifest
from ion_flux.stage2_compiler._1_analysis.topology import TopologyAnalyzer

def evaluate_ic(manifest: ExecutableManifest, current_parameters: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """
    Dynamically evaluates the Initial Conditions (y0) AST using current parameters.
    Since initial conditions depend on variables that map back to parameter inputs,
    this strictly happens dynamically at Runtime, right before FFI delegation.
    """
    layout = manifest.layout
    ast_payload = manifest.ast_payload
    
    y0 = [0.0] * layout.n_states
    ydot0 = [0.0] * layout.n_states
    
    if not ast_payload:
        return y0, ydot0
        
    topo = TopologyAnalyzer(ast_payload.get("domains", {}))
    
    def _eval_ic(node: Dict[str, Any], flat_idx: int, d_name: str) -> float:
        t = node.get("type")
        if t == "Scalar": return float(node["value"])
        if t == "Parameter":
            p_name = node.get("name")
            return current_parameters.get(p_name, manifest.default_parameters.get(p_name, 0.0))
        if t == "BinaryOp":
            l = _eval_ic(node["left"], flat_idx, d_name)
            r = _eval_ic(node["right"], flat_idx, d_name)
            op = node["op"]
            if op == "add": return l + r
            if op == "sub": return l - r
            if op == "mul": return l * r
            if op == "div": return l / r if r != 0 else 0.0
            if op == "pow": return l ** r
            if op == "max": return max(l, r)
            if op == "min": return min(l, r)
        if t == "UnaryOp":
            c = _eval_ic(node["child"], flat_idx, d_name)
            op = node["op"]
            if op == "neg": return -c
            if op == "coords":
                b_axis = node.get("axis")
                if b_axis and d_name:
                    axes = topo.get_axes(d_name)
                    strides = topo.get_strides(d_name)
                    if b_axis in axes:
                        stride = strides[b_axis]
                        res = topo.domains.get(b_axis, {}).get("resolution", 1)
                        start = topo.domains.get(b_axis, {}).get("start_idx", 0)
                        local_idx = (flat_idx // stride) % res
                        
                        # Translate FVM indices back into normalized Physical spaces dynamically
                        b_base_axis = topo.get_base_axis(b_axis)
                        if b_base_axis in layout.mesh_offsets and "w_centers" in layout.mesh_offsets[b_base_axis]:
                            centers_offset = layout.mesh_offsets[b_base_axis]["w_centers"]
                            norm_center = layout.mesh_cache.get(centers_offset + start + local_idx, 0.0)
                            bounds = topo.domains.get(b_base_axis, {}).get("bounds", (0.0, 1.0))
                            l_phys = float(bounds[1] - bounds[0])
                            return bounds[0] + norm_center * l_phys
                return 0.0
            if op == "sin": return math.sin(c)
            if op == "cos": return math.cos(c)
            if op == "exp": return math.exp(c)
            if op == "log": return math.log(c) if c > 0 else 0.0
            if op == "sqrt": return math.sqrt(c) if c > 0 else 0.0
            if op == "abs": return abs(c)
        return 0.0
        
    for ic_data in ast_payload.get("initial_conditions", []):
        state_name = ic_data["state"]
        offset, size = layout.state_offsets[state_name]
        d_name = manifest.state_domain_map.get(state_name, "")
            
        for i in range(size):
            y0[offset + i] = _eval_ic(ic_data["value"], i, d_name)
            
    return y0, ydot0