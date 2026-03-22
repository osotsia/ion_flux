from typing import Any
from .topology import get_local_index

def get_ale_terms(state_name: str, offset: int, size: int, state_map: dict, dynamic_domains: dict, visitor: Any, idx: str) -> list:
    """Generates Arbitrary Lagrangian-Eulerian grid velocity advection terms."""
    ale_terms = []
    state_obj = state_map.get(state_name)
    
    if state_obj and getattr(state_obj, "domain", None) and dynamic_domains:
        ds = state_obj.domain.domains if hasattr(state_obj.domain, "domains") else [state_obj.domain]
        for d in ds:
            if d.name in dynamic_domains:
                binding = dynamic_domains[d.name]
                if binding["side"] == "right":
                    local_idx = get_local_index(idx, state_obj.domain, d.name)
                    x_coord = f"(({local_idx}) * dx_{d.name})"
                    
                    # BUG 12 FIX: Compile the full mathematical expression for L
                    L_expr = visitor.visit(binding["rhs"], idx)
                    
                    # Temporarily flip the visitor to emit derivatives (ydot) for the L expression
                    visitor.use_ydot = True
                    L_dot_expr = visitor.visit(binding["rhs"], idx)
                    visitor.use_ydot = False
                    
                    v_mesh = f"({x_coord} / std::max(1e-12, (double)({L_expr}))) * ({L_dot_expr})"
                    grad_c = f"(((y[{offset} + CLAMP(({idx})+1, {size})]) - (y[{offset} + CLAMP(({idx})-1, {size})])) / (2.0 * dx_{d.name}))"
                    ale_terms.append(f"({v_mesh} * {grad_c})")
                    
    return ale_terms