"""
AST Inspection and Traversal Utilities.

Uses Python 3.10+ pattern matching to traverse raw AST dictionary nodes
without manual dictionary sniffing or chained .get() lookups.
"""

from typing import Dict, Any, List


def extract_state_names(node: Dict[str, Any]) -> List[str]:
    """
    Recursively walks an AST payload to extract all unique State variable names.
    Uses structural pattern matching to unpack node dictionaries directly.
    """
    names: List[str] = []

    match node:
        case {"type": "State", "name": str(name)}:
            return [name]

        case {"type": "UnaryOp" | "Boundary" | "InitialCondition", "child": dict() as child}:
            names.extend(extract_state_names(child))

        case {"type": "BinaryOp", "left": dict() as left, "right": dict() as right}:
            names.extend(extract_state_names(left))
            names.extend(extract_state_names(right))

        case {"type": "DomainBoundary"}:
            pass

        case dict():
            # Fallback for composite container dictionaries (e.g., boundaries, regions)
            for val in node.values():
                match val:
                    case dict():
                        names.extend(extract_state_names(val))
                    case list():
                        for item in val:
                            if isinstance(item, dict):
                                names.extend(extract_state_names(item))

        case _:
            return []

    # Preserve traversal order while removing duplicates
    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]


def extract_state_name(node: Dict[str, Any], layout: Any = None) -> str:
    """Extracts the primary target State name from an AST equation mapping."""
    names = extract_state_names(node)
    if not names:
        raise ValueError(f"Could not resolve a primary State target from AST node: {node}")
    return names[0]


def extract_div_child(node: Dict[str, Any]) -> Any:
    """
    Recursively searches the AST for a 'div' operator and returns its child flux node.
    Structural pattern matching isolates the target operator directly.
    """
    match node:
        case {"type": "UnaryOp", "op": "div", "child": child}:
            return child

        case {"left": left, "right": right}:
            return extract_div_child(left) or extract_div_child(right)

        case {"child": child}:
            return extract_div_child(child)

        case _:
            return None