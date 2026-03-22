from typing import List, Dict, Any

def extract_state_names(node: Dict[str, Any]) -> List[str]:
    """Recursively walks down AST wrappers to find ALL State names driving an equation."""
    node_type = node.get("type")
    
    if node_type == "State":
        return [node["name"]]
        
    names = []
    if node_type in ("UnaryOp", "Boundary", "DomainBoundary", "InitialCondition"):
        names.extend(extract_state_names(node["child"]))
    elif node_type == "BinaryOp":
        names.extend(extract_state_names(node["left"]))
        names.extend(extract_state_names(node["right"]))
        
    # Preserve deterministic discovery order while removing duplicates
    return list(dict.fromkeys(names))

def extract_state_name(node: Dict[str, Any], layout: Any) -> str:
    """Finds the primary spatial State name driving an equation."""
    names = extract_state_names(node)
    if not names:
        raise ValueError(f"Cannot extract state name from LHS node: {node}")
        
    for name in names:
        if layout.state_offsets[name][1] > 1:
            return name
    return names[0]