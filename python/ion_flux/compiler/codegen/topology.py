from typing import Any

def get_stride(domain: Any, axis_name: str) -> str:
    """Calculates spatial strides for hierarchical cross-product domains."""
    if hasattr(domain, "domains") and len(domain.domains) == 2:
        if domain.domains[0].name == axis_name:
            return str(domain.domains[1].resolution)
    return "1"

def get_local_index(global_idx_var: str, domain: Any, axis_name: str) -> str:
    """Extracts the localized dimensional index from a flattened global hardware index."""
    if hasattr(domain, "domains") and len(domain.domains) == 2:
        if domain.domains[1].name == axis_name:
            res = getattr(domain.domains[1], "parent", domain.domains[1]).resolution if getattr(domain.domains[1], "parent", None) else domain.domains[1].resolution
            return f"({global_idx_var} % {res})"
        elif domain.domains[0].name == axis_name:
            res_inner = getattr(domain.domains[1], "parent", domain.domains[1]).resolution if getattr(domain.domains[1], "parent", None) else domain.domains[1].resolution
            return f"({global_idx_var} / {res_inner})"
    return global_idx_var

def get_coord_sys(domain: Any, axis_name: str) -> str:
    if hasattr(domain, "domains"):
        for d in domain.domains:
            if d.name == axis_name:
                return getattr(d, "coord_sys", "cartesian")
    return getattr(domain, "coord_sys", "cartesian")

def get_resolution(domain: Any, axis_name: str) -> str:
    """Recursively locates the true macroscopic bounds to maintain contiguous meshing."""
    if hasattr(domain, "domains"):
        for d in domain.domains:
            if d.name == axis_name:
                return str(getattr(d, "parent", d).resolution if getattr(d, "parent", None) else d.resolution)
    if getattr(domain, "parent", None):
        return str(domain.parent.resolution)
    return str(getattr(domain, "resolution", "1"))