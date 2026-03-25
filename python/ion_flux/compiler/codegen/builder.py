from collections import defaultdict
from typing import List, Dict, Any, Tuple
from .ast_analysis import extract_state_name
from .translator import CppTranslator
from .templates import generate_cpp_skeleton

def generate_cpp(ast_payload: Dict[str, Any], layout: Any, states: List[Any], bandwidth: int = 0, target: str = "cpu") -> str:
    """Orchestrates the conversion of a pure Python AST into a native C++ simulation binary."""
    state_map = {s.name: s for s in states}
    lines = []
    
    # 1. Parse Topology and Boundaries directly from explicit semantic buckets
    dynamic_domains, bulk_eqs, dirichlet_eqs, neumann_bcs = _parse_buckets(ast_payload, layout)
    
    # 2. Emit Constants (dx)
    _emit_domain_constants(lines, states, layout, dynamic_domains)
    
    # 3. Initialize the Math Translator
    translator = CppTranslator(layout, state_map, dynamic_domains, "dx_default", neumann_bcs)

    # 4. Emit Bulk PDE/ODE Residuals mapped to Regions
    lines.append("    // --- Bulk PDE & DAE Residuals ---")
    for eq in bulk_eqs:
        _emit_residual_loop(lines, eq, target, layout, state_map, translator)

    # 5. Emit Dirichlet (DAE) Overrides
    if dirichlet_eqs:
        lines.append("\n    // --- Dirichlet Boundary Condition Overrides (DAE) ---")
        for eq in dirichlet_eqs:
            _emit_dirichlet_override(lines, eq, layout, state_map, translator)

    return generate_cpp_skeleton(layout.n_states, layout.n_params, "\n".join(lines), bandwidth)


# --- Orchestration Helpers ---

def _parse_buckets(ast_payload: Dict[str, Any], layout: Any) -> Tuple[dict, list, list, dict]:
    """Strictly routes physics equations based on their semantic bucket declarations."""
    dynamic_domains = {}
    bulk_eqs = []
    dirichlet_eqs = []
    neumann_bcs = defaultdict(dict)
    
    # Regional physical bounds (PDEs & DAEs)
    for dom_name, eqs in ast_payload.get("regions", {}).items():
        for eq in eqs:
            lhs = eq["lhs"]
            if lhs.get("type") == "DomainBoundary": 
                dynamic_domains[lhs["domain"]] = {"side": lhs["side"], "rhs": eq["rhs"]}
            elif lhs.get("type") == "InitialCondition":
                continue
            else:
                eq_copy = eq.copy()
                eq_copy["target_domain"] = dom_name
                bulk_eqs.append(eq_copy)
                
    # Global algebraic bounds (ODEs & DAEs)
    for eq in ast_payload.get("global", []):
        lhs = eq["lhs"]
        if lhs.get("type") == "DomainBoundary":
            dynamic_domains[lhs["domain"]] = {"side": lhs["side"], "rhs": eq["rhs"]}
        elif lhs.get("type") == "InitialCondition":
            continue
        else:
            eq_copy = eq.copy()
            eq_copy["target_domain"] = None
            bulk_eqs.append(eq_copy)
            
    # Boundaries dictate local Dirichlet overrides or Neumann flux injections
    for eq in ast_payload.get("boundaries", []):
        lhs = eq["lhs"]
        if lhs.get("type") == "DomainBoundary":
            dynamic_domains[lhs["domain"]] = {"side": lhs["side"], "rhs": eq["rhs"]}
        elif lhs.get("type") == "InitialCondition":
            continue
        elif lhs.get("type") == "Boundary":
            if lhs["child"].get("type") == "State":
                dirichlet_eqs.append(eq)
            else:
                state_name = extract_state_name(lhs, layout)
                neumann_bcs[state_name][lhs["side"]] = {"child": lhs["child"], "rhs": eq["rhs"]}
                
    return dynamic_domains, bulk_eqs, dirichlet_eqs, neumann_bcs

def _emit_domain_constants(lines: List[str], states: List[Any], layout: Any, dynamic_domains: dict):
    lines.append("    // --- Physical Domain Constants ---")
    lines.append("    double dx_default = 1.0;")
    
    all_domains = {}
    for s in states:
        if s.domain:
            ds = s.domain.domains if type(s.domain).__name__ in ("CompositeDomain", "ConcatenatedDomain") else [s.domain]
            for d in ds:
                all_domains[d.name] = d
            
    for d in all_domains.values():
        denom = max(d.resolution - 1, 1)
        if d.name in dynamic_domains:
            L_state = extract_state_name(dynamic_domains[d.name]["rhs"], layout)
            offset = layout.state_offsets[L_state][0]
            lines.append(f"    double dx_{d.name} = y[{offset}] / {denom}.0;")
            lines.append(f"    if (dx_{d.name} <= 0.0) {{ for(int i=0; i<{layout.n_states}; ++i) res[i] = std::nan(\"\"); return; }}")
        else:
            dx_val = float(d.bounds[1] - d.bounds[0]) / denom
            lines.append(f"    double dx_{d.name} = {dx_val};")
    lines.append("")

def _emit_residual_loop(lines: List[str], eq: Dict[str, Any], target: str, layout: Any, state_map: dict, translator: CppTranslator):
    try:
        # Standard extraction for equations like: dt(T) == RHS
        state_name = extract_state_name(eq["lhs"], layout)
    except ValueError:
        # Graceful fallback for declarative algebraic equations like: 0 == RHS
        state_name = extract_state_name(eq["rhs"], layout)
        
    offset, total_size = layout.state_offsets[state_name]
    state_obj = state_map[state_name]
    translator.current_domain = getattr(state_obj, "domain", None)
    
    # 1. Resolve sub-offsets for Regionally Bound Mathematics
    loop_start = 0
    loop_size = total_size
    
    target_domain_name = eq.get("target_domain")
    if target_domain_name and getattr(state_obj, "domain", None):
        domain = state_obj.domain
        if type(domain).__name__ == "ConcatenatedDomain":
            current_offset = 0
            for d in domain.domains:
                if getattr(d, "name", "") == target_domain_name:
                    loop_start = current_offset
                    loop_size = getattr(d, "resolution", 1)
                    break
                current_offset += getattr(d, "resolution", 1)
        elif type(domain).__name__ == "CompositeDomain":
            # For macro-micro (e.g., x_n * r_n), if the target_domain matches exactly, keep full layout size.
            pass 

    omp_pragma = "    #pragma omp parallel for\n" if ("omp" in target and loop_size > 50) else ""

    # 2. Emit the native C++ arrays bound to the exact spatial region
    if hasattr(state_obj.domain, "domains") and type(state_obj.domain).__name__ == "CompositeDomain" and len(state_obj.domain.domains) == 2:
        d_mac, d_mic = state_obj.domain.domains[0], state_obj.domain.domains[1]
        mac_start = loop_start // d_mic.resolution
        mac_end = (loop_start + loop_size) // d_mic.resolution
        lines.append(f"{omp_pragma}    for (int i_mac = {mac_start}; i_mac < {mac_end}; ++i_mac) {{")
        lines.append(f"        for (int i_mic = 0; i_mic < {d_mic.resolution}; ++i_mic) {{")
        lines.append(f"            int i = i_mac * {d_mic.resolution} + i_mic;")
        lhs_cpp = translator.translate(eq["lhs"], "i")
        rhs_cpp = translator.translate(eq["rhs"], "i")
        lines.append(f"            res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
        lines.append(f"        }}\n    }}")
    elif loop_size > 1:
        lines.append(f"{omp_pragma}    for (int i = {loop_start}; i < {loop_start + loop_size}; ++i) {{")
        lhs_cpp = translator.translate(eq["lhs"], "i")
        rhs_cpp = translator.translate(eq["rhs"], "i")
        lines.append(f"        res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});\n    }}")
    else:
        idx = str(loop_start) if loop_start > 0 else "0"
        lhs_cpp = translator.translate(eq["lhs"], idx)
        rhs_cpp = translator.translate(eq["rhs"], idx)
        lines.append(f"    res[{offset} + {idx}] = ({lhs_cpp}) - ({rhs_cpp});")

def _emit_dirichlet_override(lines: List[str], eq: Dict[str, Any], layout: Any, state_map: dict, translator: CppTranslator):
    state_name = extract_state_name(eq["lhs"], layout)
    offset, size = layout.state_offsets[state_name]
    side = eq["lhs"]["side"]
    bc_domain = eq["lhs"].get("domain")
    state_obj = state_map[state_name]
    translator.current_domain = getattr(state_obj, "domain", None)
    
    # Enable dynamic 3D surface tag targeting via the p-mask
    if state_obj.domain and getattr(state_obj.domain, "coord_sys", "") == "unstructured":
        if state_obj.domain.name in layout.mesh_offsets and side in layout.mesh_offsets[state_obj.domain.name]["surfaces"]:
            off_surf = layout.mesh_offsets[state_obj.domain.name]["surfaces"][side]
            lines.append(f"    for (int i = 0; i < {size}; ++i) {{")
            lines.append(f"        if (p[{off_surf} + i] > 0.5) {{")
            lhs_cpp = translator.translate(eq["lhs"]["child"], "i")
            rhs_cpp = translator.translate(eq["rhs"], "i")
            lines.append(f"            res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
            lines.append(f"        }}\n    }}")
            return

    # Fallback to standard 1D/2D boundaries
    if hasattr(state_obj.domain, "domains") and type(state_obj.domain).__name__ == "CompositeDomain" and len(state_obj.domain.domains) == 2 and bc_domain == state_obj.domain.domains[1].name:
        d_mac, d_mic = state_obj.domain.domains[0], state_obj.domain.domains[1]
        idx_str = f"i_mac * {d_mic.resolution}" if side == "left" else f"i_mac * {d_mic.resolution} + {d_mic.resolution - 1}"
        lines.append(f"    for (int i_mac = 0; i_mac < {d_mac.resolution}; ++i_mac) {{")
        lines.append(f"        int b_idx = {idx_str};")
        lhs_cpp = translator.translate(eq["lhs"]["child"], "b_idx")
        rhs_cpp = translator.translate(eq["rhs"], "b_idx")
        lines.append(f"        res[{offset} + b_idx] = ({lhs_cpp}) - ({rhs_cpp});\n    }}")
    else:
        idx = "0" if side == "left" else f"{size - 1}"
        lhs_cpp = translator.translate(eq["lhs"]["child"], idx)
        rhs_cpp = translator.translate(eq["rhs"], idx)
        lines.append(f"    res[{offset} + {idx}] = ({lhs_cpp}) - ({rhs_cpp});")