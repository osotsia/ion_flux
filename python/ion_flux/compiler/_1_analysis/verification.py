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
        p_bounds = p_info.get("bounds", (0.0, 1.0))
        
        children.sort(key=lambda x: x[1]["start_idx"])
        
        current_bound = p_bounds[0]
        
        for i in range(len(children)):
            c_name, c_info = children[i]
            c_start = c_info["start_idx"]
            c_res = c_info["resolution"]
            c_end = c_start + c_res
            c_bounds = c_info.get("bounds", (0.0, 1.0))
            
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

            # Enforce Top-Down strict physical bounds mapping
            if abs(c_bounds[0] - current_bound) > 1e-12:
                raise TopologicalError(
                    f"Topological Gap/Overlap Detected! Region '{c_name}' starts at physical bound {c_bounds[0]}, "
                    f"but expected {current_bound} to perfectly tile parent '{p_name}'."
                )
            current_bound = c_bounds[1]

        if abs(current_bound - p_bounds[1]) > 1e-12:
            raise TopologicalError(
                f"Topological Gap Detected! Regions of parent '{p_name}' end at {current_bound}, "
                f"but parent extends to {p_bounds[1]}."
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
    from collections import defaultdict
    bc_id_to_states = defaultdict(set)
    
    # 1. Walk equations to map spatial states and track _bc_id contexts
    for eq_data in ast_payload.get("equations", []):
        state_name = eq_data["state"]
        
        def walk_and_map(node: Any, in_reduction: bool = False):
            if isinstance(node, dict):
                is_reduction = in_reduction or node.get("type") in ("Boundary", "DomainBoundary") or (node.get("type") == "UnaryOp" and node.get("op") == "integral")
                
                # Flag state as spatial if it uses grad or div and is not inside a reduction
                if not in_reduction and node.get("type") == "UnaryOp" and node.get("op") in ("grad", "div"):
                    spatial_states.add(state_name)
                
                # Map any node capable of receiving a boundary condition to its parent state
                if "_bc_id" in node:
                    bc_id_to_states[node["_bc_id"]].add(state_name)
                    
                for v in node.values():
                    walk_and_map(v, is_reduction)
            elif isinstance(node, list):
                for item in node:
                    walk_and_map(item, in_reduction)
                    
        walk_and_map(eq_data)

    # 2. Collect all states that have boundary conditions applied to them
    bound_states = set()
    for bc_data in ast_payload.get("boundaries", []):
        if bc_data["type"] == "dirichlet":
            bound_states.add(bc_data["state"])
        elif bc_data["type"] == "neumann":
            node_id = bc_data.get("node_id")
            if node_id in bc_id_to_states:
                bound_states.update(bc_id_to_states[node_id])

    # 3. Manifold Closure Check
    # We do not strictly enforce `spatial_states == bound_states` because 0D states 
    # might have algebraic Dirichlet overrides. However, EVERY spatial state MUST 
    # have at least one boundary condition to mathematically close the domain.
    unbound_spatial = spatial_states - bound_states
    
    if unbound_spatial:
        raise TopologicalError(
            f"Missing Boundary Conditions! The following states are governed by spatial "
            f"PDEs (grad/div) but have no boundaries defined: {', '.join(unbound_spatial)}. "
            f"This creates an open-ended mathematical manifold and will cause undefined physics."
        )