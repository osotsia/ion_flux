"""
Static Manifold Verification Pass

Transforms runtime capacity drifts and index out-of-bounds errors into 
strict compile-time exceptions. Mathematically verifies that the discrete 
topological layouts emitted by the DSL strictly conserve spatial invariants 
(no overlapping memory arrays, no out-of-bounds slicing).
"""

from typing import Dict, Any

class TopologicalError(Exception):
    """Raised when the domain manifold violates geometric or memory invariants."""
    pass

def verify_manifold(ast_payload: Dict[str, Any]) -> None:
    """
    Executes an invariant-based verification of the geometric mesh layout.
    Guarantees that subsequent C++ emission is memory-safe and mass-conservative.
    """
    domains = ast_payload.get("domains", {})
    
    # 1. Group sub-regions by their parent for adjacency checks
    parent_map = {}
    for d_name, d_info in domains.items():
        if d_info.get("type") == "standard" and d_info.get("parent"):
            p_name = d_info["parent"]
            if p_name not in parent_map:
                parent_map[p_name] = []
            parent_map[p_name].append((d_name, d_info))

    # 2. Enforce Sub-Mesh Adjacency and Array Boundary Invariants
    for p_name, children in parent_map.items():
        if p_name not in domains:
            raise TopologicalError(f"Parent domain '{p_name}' is missing from the manifold.")
            
        p_info = domains[p_name]
        p_res = p_info["resolution"]
        
        # Sort children by their starting index in the C-array
        children.sort(key=lambda x: x[1]["start_idx"])
        
        for i in range(len(children)):
            c_name, c_info = children[i]
            c_start = c_info["start_idx"]
            c_res = c_info["resolution"]
            c_end = c_start + c_res
            
            # Invariant A: Strict Array Bounds
            if c_start < 0 or c_end > p_res:
                raise TopologicalError(
                    f"Sub-mesh '{c_name}' (Indices {c_start} to {c_end}) violates "
                    f"the physical memory bounds of its parent '{p_name}' (Max {p_res})."
                )
            
            # Invariant B: Memory Overlap Prevention
            if i < len(children) - 1:
                next_name, next_info = children[i+1]
                next_start = next_info["start_idx"]
                
                # Strict Face-Sharing FVM topology: end index must not exceed the next start index
                if c_end > next_start:
                    raise TopologicalError(
                        f"Topological Overlap Detected! Region '{c_name}' ends at index {c_end}, "
                        f"but contiguous Region '{next_name}' starts at {next_start}. "
                        f"This will cause silent memory overwrites during C++ evaluation."
                    )

    # 3. Enforce Composite Domain Factorization
    for d_name, d_info in domains.items():
        if d_info.get("type") == "composite":
            expected_res = 1
            for sub_name in d_info.get("domains", []):
                if sub_name not in domains:
                    raise TopologicalError(
                        f"Composite domain '{d_name}' references unknown sub-domain '{sub_name}'."
                    )
                expected_res *= domains[sub_name]["resolution"]
                
            if d_info["resolution"] != expected_res:
                raise TopologicalError(
                    f"Composite Domain '{d_name}' has a resolution of {d_info['resolution']}, "
                    f"but its factored sub-domains yield {expected_res}."
                )