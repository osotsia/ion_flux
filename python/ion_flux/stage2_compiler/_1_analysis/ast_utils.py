from typing import Dict, Any, List

def extract_state_names(node: Dict[str, Any]) -> List[str]:
    """Recursively walks down AST wrappers to find ALL State names driving an equation."""
    if not isinstance(node, dict):
        return []
        
    node_type = node.get("type")

    if node_type == "State":
        return [node["name"]]

    names = []
    
    if node_type in ("UnaryOp", "Boundary", "InitialCondition"):
        if "child" in node:
            names.extend(extract_state_names(node["child"]))
    elif node_type == "BinaryOp":
        if "left" in node:
            names.extend(extract_state_names(node["left"]))
        if "right" in node:
            names.extend(extract_state_names(node["right"]))
    elif node_type == "DomainBoundary":
        pass
    else:
        for key, val in node.items():
            if isinstance(val, dict):
                names.extend(extract_state_names(val))
            elif isinstance(val, list):
                for item in val:
                    names.extend(extract_state_names(item))

    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]

def extract_state_name(node: Dict[str, Any], layout: Any = None) -> str:
    """Extracts the primary target State name from an AST equation mapping."""
    names = extract_state_names(node)
    if not names:
        raise ValueError(f"Could not resolve a primary State target from AST node: {node}")
    return names[0]

def extract_div_child(node: Dict[str, Any]) -> Any:
    """Recursively searches the AST for a 'div' operator and returns its child flux node."""
    if not isinstance(node, dict): return None
    if node.get("type") == "UnaryOp" and node.get("op") == "div":
        return node.get("child")
    if "left" in node and "right" in node:
        res = extract_div_child(node["left"])
        if res: return res
        return extract_div_child(node["right"])
    if "child" in node:
        return extract_div_child(node["child"])
    return None