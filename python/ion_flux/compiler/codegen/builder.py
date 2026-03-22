from collections import defaultdict
from typing import List, Dict, Any, Tuple
from .ast_analysis import extract_state_name
from .translator import CppTranslator
from .templates import generate_cpp_skeleton

def generate_cpp(ast_payload: List[Dict[str, Any]], layout: Any, states: List[Any], bandwidth: int = 0, target: str = "cpu") -> str:
    """Orchestrates the conversion of a pure Python AST into a native C++ simulation binary."""
    state_map = {s.name: s for s in states}
    lines = []
    
    # 1. Parse Topology and Boundaries
    dynamic_domains, bulk_eqs, dirichlet_eqs, neumann_bcs = _sort_equations(ast_payload, layout)
    
    # 2. Emit Constants (dx)
    _emit_domain_constants(lines, states, layout, dynamic_domains)
    
    # 3. Initialize the Math Translator
    translator = CppTranslator(layout, state_map, dynamic_domains, "dx_default", neumann_bcs)

    # 4. Emit Bulk PDE/ODE Residuals
    lines.append("    // --- Bulk PDE Residuals ---")
    for eq in bulk_eqs:
        _emit_residual_loop(lines, eq, target, layout, state_map, translator)

    # 5. Emit Dirichlet (DAE) Overrides
    if dirichlet_eqs:
        lines.append("\n    // --- Dirichlet Boundary Condition Overrides (DAE) ---")
        for eq in dirichlet_eqs:
            _emit_dirichlet_override(lines, eq, layout, state_map, translator)

    return generate_cpp_skeleton(layout.n_states, layout.n_params, "\n".join(lines), bandwidth)


# --- Orchestration Helpers ---

def _sort_equations(ast_payload: List[Dict[str, Any]], layout: Any) -> Tuple[dict, list, list, dict]:
    dynamic_domains = {}
    bulk_eqs = []
    dirichlet_eqs = []
    neumann_bcs = defaultdict(dict)
    
    for eq in ast_payload:
        lhs = eq["lhs"]
        if lhs.get("type") == "DomainBoundary":
            dynamic_domains[lhs["domain"]] = {"side": lhs["side"], "rhs": eq["rhs"]}
        elif lhs.get("type") == "Boundary":
            if lhs["child"].get("type") == "State":
                dirichlet_eqs.append(eq)
            else:
                state_name = extract_state_name(lhs, layout)
                neumann_bcs[state_name][lhs["side"]] = {"child": lhs["child"], "rhs": eq["rhs"]}
        elif lhs.get("type") != "InitialCondition":
            bulk_eqs.append(eq)
            
    return dynamic_domains, bulk_eqs, dirichlet_eqs, neumann_bcs

def _emit_domain_constants(lines: List[str], states: List[Any], layout: Any, dynamic_domains: dict):
    lines.append("    // --- Physical Domain Constants ---")
    lines.append("    double dx_default = 1.0;")
    
    all_domains = {}
    for s in states:
        if s.domain:
            ds = s.domain.domains if hasattr(s.domain, "domains") else [s.domain]
            for d in ds:
                all_domains[d.name] = d
                
    for d in all_domains.values():
        denom = max(d.resolution - 1, 1)
        if d.name in dynamic_domains:
            L_state = extract_state_name(dynamic_domains[d.name]["rhs"], layout)
            offset = layout.state_offsets[L_state][0]
            lines.append(f"    double dx_{d.name} = y[{offset}] / {denom}.0;")
        else:
            dx_val = float(d.bounds[1] - d.bounds[0]) / denom
            lines.append(f"    double dx_{d.name} = {dx_val};")
    lines.append("")

def _emit_residual_loop(lines: List[str], eq: Dict[str, Any], target: str, layout: Any, state_map: dict, translator: CppTranslator):
    state_name = extract_state_name(eq["lhs"], layout)
    offset, size = layout.state_offsets[state_name]
    state_obj = state_map[state_name]
    
    omp_pragma = "    #pragma omp parallel for\n" if ("omp" in target and size > 50) else ""

    if hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
        d_mac, d_mic = state_obj.domain.domains[0], state_obj.domain.domains[1]
        lines.append(f"{omp_pragma}    for (int i_mac = 0; i_mac < {d_mac.resolution}; ++i_mac) {{")
        lines.append(f"        for (int i_mic = 0; i_mic < {d_mic.resolution}; ++i_mic) {{")
        lines.append(f"            int i = i_mac * {d_mic.resolution} + i_mic;")
        lhs_cpp = translator.translate(eq["lhs"], "i")
        rhs_cpp = translator.translate(eq["rhs"], "i")
        lines.append(f"            res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
        lines.append(f"        }}\n    }}")
    elif size > 1:
        lines.append(f"{omp_pragma}    for (int i = 0; i < {size}; ++i) {{")
        lhs_cpp = translator.translate(eq["lhs"], "i")
        rhs_cpp = translator.translate(eq["rhs"], "i")
        lines.append(f"        res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});\n    }}")
    else:
        lhs_cpp = translator.translate(eq["lhs"], "0")
        rhs_cpp = translator.translate(eq["rhs"], "0")
        lines.append(f"    res[{offset}] = ({lhs_cpp}) - ({rhs_cpp});")

def _emit_dirichlet_override(lines: List[str], eq: Dict[str, Any], layout: Any, state_map: dict, translator: CppTranslator):
    state_name = extract_state_name(eq["lhs"], layout)
    offset, size = layout.state_offsets[state_name]
    side = eq["lhs"]["side"]
    bc_domain = eq["lhs"].get("domain")
    state_obj = state_map[state_name]
    
    if hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2 and bc_domain == state_obj.domain.domains[1].name:
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