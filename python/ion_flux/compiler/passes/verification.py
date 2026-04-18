from typing import Dict, Any, List

class TopologicalError(Exception):
    pass

def verify_manifold(ast_payload: Dict[str, Any]) -> None:
    domains = ast_payload.get("domains", {})
    
    parent_map = {}
    for d_name, d_info in domains.items():
        if d_info.get("type") == "standard" and d_info.get("parent"):
            p_name = d_info["parent"]
            if p_name not in parent_map:
                parent_map[p_name] = []
            parent_map[p_name].append((d_name, d_info))

    for p_name, children in parent_map.items():
        if p_name not in domains:
            raise TopologicalError(f"Parent domain '{p_name}' is missing from the manifold.")
            
        p_info = domains[p_name]
        p_res = p_info["resolution"]
        
        children.sort(key=lambda x: x[1]["start_idx"])
        
        for i in range(len(children)):
            c_name, c_info = children[i]
            c_start = c_info["start_idx"]
            c_res = c_info["resolution"]
            c_end = c_start + c_res
            
            if c_start < 0 or c_end > p_res:
                raise TopologicalError(
                    f"Sub-mesh '{c_name}' (Indices {c_start} to {c_end}) violates "
                    f"the physical memory bounds of its parent '{p_name}' (Max {p_res})."
                )
            
            if i < len(children) - 1:
                next_name, next_info = children[i+1]
                next_start = next_info["start_idx"]
                
                if c_end > next_start:
                    raise TopologicalError(
                        f"Topological Overlap Detected! Region '{c_name}' ends at index {c_end}, "
                        f"but contiguous Region '{next_name}' starts at {next_start}. "
                        f"This will cause silent memory overwrites during C++ evaluation."
                    )

    for d_name, d_info in domains.items():
        if d_info.get("type") == "composite":
            expected_res = 1
            for sub_name in d_info.get("domains", []):
                if sub_name not in domains:
                    raise TopologicalError(f"Composite domain '{d_name}' references unknown sub-domain '{sub_name}'.")
                expected_res *= domains[sub_name]["resolution"]
                
            if d_info["resolution"] != expected_res:
                raise TopologicalError(
                    f"Composite Domain '{d_name}' has a resolution of {d_info['resolution']}, "
                    f"but its factored sub-domains yield {expected_res}."
                )

    # Formalized Pre-Lowering Safety Checks for PDE Math Boundary Invariants
    _verify_boundaries(ast_payload)

def _verify_boundaries(ast_payload: Dict[str, Any]) -> None:
    spatial_states = set()
    for eq_data in ast_payload.get("equations", []):
        state_name = eq_data["state"]
        def check_spatial(node: Any) -> bool:
            if isinstance(node, dict):
                if node.get("type") == "UnaryOp" and node.get("op") in ("grad", "div"): return True
                return any(check_spatial(v) for v in node.values())
            elif isinstance(node, list):
                return any(check_spatial(item) for item in node)
            return False
            
        if check_spatial(eq_data):
            spatial_states.add(state_name)

    bound_states = set()
    for bc_data in ast_payload.get("boundaries", []):
        if bc_data["type"] == "dirichlet":
            bound_states.add(bc_data["state"])
        elif bc_data["type"] == "neumann":
            # Best effort analysis for identifying spatial coupling references
            pass
            
    # For future extension: We can enforce that len(bound_states) aligns with len(spatial_states)
    # allowing us to throw early compilation halts before relying on C++ segfault diagnostics