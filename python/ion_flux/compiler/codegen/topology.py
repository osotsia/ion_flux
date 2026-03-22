from typing import Any

def get_stride(domain: Any, axis_name: str) -> str:
    """Calculates spatial strides for hierarchical domains."""
    if hasattr(domain, "domains") and len(domain.domains) == 2:
        if domain.domains[0].name == axis_name:
            return str(domain.domains[1].resolution)
    return "1"

def get_local_index(idx_var: str, domain: Any, axis_name: str) -> str:
    """Extracts the localized dimension index from a flattened global index."""
    if hasattr(domain, "domains") and len(domain.domains) == 2:
        if domain.domains[1].name == axis_name:
            return f"({idx_var} % {domain.domains[1].resolution})"
        elif domain.domains[0].name == axis_name:
            return f"({idx_var} / {domain.domains[1].resolution})"
    return idx_var

def get_coord_sys(domain: Any, axis_name: str) -> str:
    if hasattr(domain, "domains"):
        for d in domain.domains:
            if d.name == axis_name:
                return getattr(d, "coord_sys", "cartesian")
    return getattr(domain, "coord_sys", "cartesian")