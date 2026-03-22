from typing import List, Dict, Any
from .utils import extract_state_name
from .visitor import CPPEmitter
from .templates import generate_cpp_skeleton

def generate_cpp(ast_payload: List[Dict[str, Any]], layout: Any, states: List[Any], bandwidth: int = 0, target: str = "cpu") -> str:
    state_map = {s.name: s for s in states}
    lines = []
    
    dynamic_domains = {}
    for eq in ast_payload:
        if eq["lhs"].get("type") == "DomainBoundary":
            dynamic_domains[eq["lhs"]["domain"]] = {"side": eq["lhs"]["side"], "rhs": eq["rhs"]}
    
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

    # Filter Dirichlet (State) vs Neumann (Flux) Boundaries
    bulk_eqs = []
    dirichlet_eqs = []
    neumann_bcs = {}
    
    for eq in ast_payload:
        lhs = eq["lhs"]
        if lhs.get("type") == "Boundary":
            if lhs["child"].get("type") == "State":
                dirichlet_eqs.append(eq) # State constraints overwrite the PDE to form a DAE
            else:
                state_name = extract_state_name(lhs, layout)
                if state_name not in neumann_bcs: neumann_bcs[state_name] = {}
                neumann_bcs[state_name][lhs["side"]] = {"child": lhs["child"], "rhs": eq["rhs"]}
        elif lhs.get("type") not in ("InitialCondition", "DomainBoundary"):
            bulk_eqs.append(eq)

    # Initialize the robust refactored visitor
    emitter = CPPEmitter(layout, state_map, dynamic_domains, "dx_default", neumann_bcs)

    lines.append("    // --- Bulk PDE Residuals ---")
    for eq in bulk_eqs:
        state_name = extract_state_name(eq["lhs"], layout)
        offset, size = layout.state_offsets[state_name]
        state_obj = state_map[state_name]
        
        omp_pragma = "    #pragma omp parallel for\n" if ("omp" in target and size > 50) else ""

        if hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
            d_mac, d_mic = state_obj.domain.domains[0], state_obj.domain.domains[1]
            lines.append(omp_pragma + f"    for (int i_mac = 0; i_mac < {d_mac.resolution}; ++i_mac) {{")
            lines.append(f"        for (int i_mic = 0; i_mic < {d_mic.resolution}; ++i_mic) {{")
            lines.append(f"            int i = i_mac * {d_mic.resolution} + i_mic;")
            lhs_cpp = emitter.visit(eq["lhs"], "i")
            rhs_cpp = emitter.visit(eq["rhs"], "i")
            lines.append(f"            res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
            lines.append(f"        }}")
            lines.append(f"    }}")
        elif size > 1:
            lines.append(omp_pragma + f"    for (int i = 0; i < {size}; ++i) {{")
            lhs_cpp = emitter.visit(eq["lhs"], "i")
            rhs_cpp = emitter.visit(eq["rhs"], "i")
            lines.append(f"        res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
            lines.append(f"    }}")
        else:
            lhs_cpp = emitter.visit(eq["lhs"], "0")
            rhs_cpp = emitter.visit(eq["rhs"], "0")
            lines.append(f"    res[{offset}] = ({lhs_cpp}) - ({rhs_cpp});")

    if dirichlet_eqs:
        lines.append("")
        lines.append(f"    // --- Dirichlet Boundary Condition Overrides (DAE) ---")
        for eq in dirichlet_eqs:
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
                lhs_cpp = emitter.visit(eq["lhs"]["child"], "b_idx")
                rhs_cpp = emitter.visit(eq["rhs"], "b_idx")
                lines.append(f"        res[{offset} + b_idx] = ({lhs_cpp}) - ({rhs_cpp});")
                lines.append(f"    }}")
            else:
                idx = "0" if side == "left" else f"{size - 1}"
                lhs_cpp = emitter.visit(eq["lhs"]["child"], idx)
                rhs_cpp = emitter.visit(eq["rhs"], idx)
                lines.append(f"    res[{offset} + {idx}] = ({lhs_cpp}) - ({rhs_cpp});")

    body = "\n".join(lines)
    return generate_cpp_skeleton(layout.n_states, layout.n_params, body, bandwidth)