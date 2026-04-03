from typing import Any
from .topology import get_local_index, get_coord_sys
from . import ir

def get_ale_terms(state_name: str, offset: int, size: int, state_map: dict, dynamic_domains: dict, translator: Any, idx: str) -> list:
    """
    Generates Arbitrary Lagrangian-Eulerian (ALE) mesh kinematics.
    
    Because battery physics utilize material-tracking meshes (the grid stretches 
    in tandem with the physical lattice during intercalation), the diffusion 
    flux is inherently defined relative to the moving mesh. Thus, the advective 
    transport terms (v * grad(c)) perfectly cancel out analytically. 
    
    The only required ALE kinematic is the geometric Dilution term (-c * div(v)) 
    to continuously conserve mass as the finite-volume cells expand.
    """
    ale_terms = []
    state_obj = state_map.get(state_name)
    
    if not (state_obj and getattr(state_obj, "domain", None) and dynamic_domains):
        return ale_terms

    domains = state_obj.domain.domains if hasattr(state_obj.domain, "domains") else [state_obj.domain]
    
    for d in domains:
        if d.name in dynamic_domains:
            binding = dynamic_domains[d.name]
            if binding["side"] == "right":
                idx_expr = ir.Var(idx) if isinstance(idx, str) else idx
                idx_str = idx if isinstance(idx, str) else idx.to_cpp()
                
                # Compile the mathematical expression mapping the moving boundary (L)
                L_expr = translator.translate(binding["rhs"], idx_expr)
                
                # Temporarily flip the translator to emit time derivatives (L_dot)
                translator.use_ydot = True
                L_dot_expr = translator.translate(binding["rhs"], idx_expr)
                translator.use_ydot = False
                
                y_curr = f"y[{offset} + CLAMP({idx_str}, {size})]"
                
                # Calculate geometric divergence of the mesh velocity field.
                # E.g., for a sphere, div(v) = 3 * L_dot / L
                coord_sys = get_coord_sys(state_obj.domain, d.name)
                dim_mult = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)
                
                div_v_mesh = f"({dim_mult} * ({L_dot_expr.to_cpp()}) / std::max(1e-12, (double)({L_expr.to_cpp()})))"
                
                # The volumetric dilution term
                dilution_term = f"(-({y_curr}) * {div_v_mesh})"
                
                ale_terms.append(dilution_term)
                
    return ale_terms