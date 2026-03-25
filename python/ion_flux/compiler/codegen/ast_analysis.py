from typing import Dict, Any, List

def extract_state_names(node: Dict[str, Any]) -> List[str]:
    """Recursively walks down AST wrappers to find ALL State names driving an equation."""
    if not isinstance(node, dict):
        return []
        
    node_type = node.get("type")

    if node_type == "State":
        return [node["name"]]

    names = []
    
    # Explicit mapping for predictable AST branches
    if node_type in ("UnaryOp", "Boundary", "InitialCondition"):
        if "child" in node:
            names.extend(extract_state_names(node["child"]))
    elif node_type == "BinaryOp":
        if "left" in node:
            names.extend(extract_state_names(node["left"]))
        if "right" in node:
            names.extend(extract_state_names(node["right"]))
    elif node_type == "DomainBoundary":
        # DomainBoundaries explicitly map geometric grid bounds, not underlying states.
        pass
    else:
        # Defensive fallback to exhaustively search for state definitions
        # across unclassified custom DSL node structures.
        for key, val in node.items():
            if isinstance(val, dict):
                names.extend(extract_state_names(val))
            elif isinstance(val, list):
                for item in val:
                    names.extend(extract_state_names(item))

    # Filter duplicates while maintaining deterministic evaluation order
    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]


def extract_state_name(node: Dict[str, Any], layout: Any = None) -> str:
    """Extracts the primary target State name from an AST equation mapping."""
    names = extract_state_names(node)
    if not names:
        raise ValueError(f"Could not resolve a primary State target from AST node: {node}")
    return names[0]