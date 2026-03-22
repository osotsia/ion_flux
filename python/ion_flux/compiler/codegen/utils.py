from typing import List, Dict, Any

def extract_state_names(node: Dict[str, Any]) -> List[str]:
    """Recursively walks down AST wrappers to find ALL State names driving an equation."""
    t = node.get("type")
    if t == "State":
        return [node["name"]]
    names = []
    if t in ("UnaryOp", "Boundary", "DomainBoundary", "InitialCondition"):
        names.extend(extract_state_names(node["child"]))
    elif t == "BinaryOp":
        names.extend(extract_state_names(node["left"]))
        names.extend(extract_state_names(node["right"]))
        
    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]

def extract_state_name(node: Dict[str, Any], layout: Any) -> str:
    """Finds the primary State name driving an equation, prioritizing spatial domains."""
    names = extract_state_names(node)
    if not names:
        raise ValueError(f"Cannot extract state name from LHS node: {node}")
    for name in names:
        if layout.state_offsets[name][1] > 1:
            return name
    return names[0]