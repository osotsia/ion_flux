from typing import List, Dict, Any
import textwrap
from .memory import MemoryLayout

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
        
    # Preserve deterministic discovery order while removing duplicates
    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]

def extract_state_name(node: Dict[str, Any], layout: MemoryLayout) -> str:
    """Finds the primary State name driving an equation, prioritizing spatial domains."""
    names = extract_state_names(node)
    if not names:
        raise ValueError(f"Cannot extract state name from LHS node: {node}")
    for name in names:
        if layout.state_offsets[name][1] > 1:
            return name
    return names[0]

def to_cpp(node: Dict[str, Any], layout: MemoryLayout, idx: str = "0", dx_symbol: str = "1.0", state_map: Dict[str, Any] = None, dynamic_domains: Dict[str, Any] = None) -> str:
    """
    Recursively lowers the Python AST into C++ code strings. 
    Injects spatial awareness by routing physical `dx` symbols to differential operators.
    Handles ALE moving boundary advection dynamically based on state mappings.
    """
    t = node.get("type")
    
    if t == "Scalar": return str(node["value"])
    if t == "Parameter": return f"p[{layout.get_param_offset(node['name'])}]"
    if t == "State": 
        offset, size = layout.state_offsets[node["name"]]
        return f"y[{offset} + CLAMP({idx}, {size})]"
        
    if t == "BinaryOp":
        l = to_cpp(node["left"], layout, idx, dx_symbol, state_map, dynamic_domains)
        r = to_cpp(node["right"], layout, idx, dx_symbol, state_map, dynamic_domains)
        op = node["op"]
        
        if op == "add": return f"({l} + {r})"
        if op == "sub": return f"({l} - {r})"
        if op == "mul": return f"({l} * {r})"
        if op == "div": return f"({l} / {r})"
        if op == "pow": return f"std::pow({l}, {r})"
        if op == "max": return f"std::max((double)({l}), (double)({r}))"
        if op == "min": return f"std::min((double)({l}), (double)({r}))"
        if op == "gt": return f"(({l}) > ({r}) ? 1.0 : 0.0)"
        if op == "lt": return f"(({l}) < ({r}) ? 1.0 : 0.0)"
        if op == "ge": return f"(({l}) >= ({r}) ? 1.0 : 0.0)"
        if op == "le": return f"(({l}) <= ({r}) ? 1.0 : 0.0)"
        if op == "eq": return f"(({l}) == ({r}) ? 1.0 : 0.0)"
        if op == "ne": return f"(({l}) != ({r}) ? 1.0 : 0.0)"
        
    if t == "UnaryOp":
        op = node["op"]
        child = node["child"]
        
        if op == "neg": return f"(-{to_cpp(child, layout, idx, dx_symbol, state_map, dynamic_domains)})"
        if op == "abs": return f"std::abs({to_cpp(child, layout, idx, dx_symbol, state_map, dynamic_domains)})"
        if op == "exp": return f"std::exp({to_cpp(child, layout, idx, dx_symbol, state_map, dynamic_domains)})"
        if op == "log": return f"std::log({to_cpp(child, layout, idx, dx_symbol, state_map, dynamic_domains)})"
        if op == "sin": return f"std::sin({to_cpp(child, layout, idx, dx_symbol, state_map, dynamic_domains)})"
        if op == "cos": return f"std::cos({to_cpp(child, layout, idx, dx_symbol, state_map, dynamic_domains)})"
        
        if op == "dt": 
            state_name = extract_state_name(child, layout)
            offset, size = layout.state_offsets[state_name]
            base_dt = f"ydot[{offset} + CLAMP({idx}, {size})]"
            
            # Inject Arbitrary Lagrangian-Eulerian (ALE) Advection
            ale_terms = []
            state_obj = state_map.get(state_name) if state_map else None
            if state_obj and state_obj.domain and dynamic_domains:
                ds = state_obj.domain.domains if hasattr(state_obj.domain, "domains") else [state_obj.domain]
                for d in ds:
                    if d.name in dynamic_domains:
                        binding = dynamic_domains[d.name]
                        if binding["side"] == "right":
                            L_state = extract_state_name(binding["rhs"], layout)
                            L_offset = layout.state_offsets[L_state][0]
                            v_mesh = f"((({idx}) * dx_{d.name}) / std::max(1e-12, y[{L_offset}])) * ydot[{L_offset}]"
                            grad_c = f"(((y[{offset} + CLAMP(({idx})+1, {size})]) - (y[{offset} + CLAMP(({idx})-1, {size})])) / (2.0 * dx_{d.name}))"
                            ale_terms.append(f"({v_mesh} * {grad_c})")
            if ale_terms:
                return f"({base_dt} - " + " - ".join(ale_terms) + ")"
            return base_dt
            
        if op == "grad":
            axis_name = node.get("axis")
            target_dx = f"dx_{axis_name}" if axis_name else dx_symbol
            stride = "1"
            
            if axis_name and state_map:
                try:
                    state_name = extract_state_name(child, layout)
                    s_domain = state_map.get(state_name).domain
                    if s_domain and hasattr(s_domain, "domains") and len(s_domain.domains) == 2:
                        if s_domain.domains[0].name == axis_name:
                            stride = str(s_domain.domains[1].resolution)
                except ValueError:
                    pass
                    
            right = to_cpp(child, layout, f"({idx}) + {stride}", target_dx, state_map, dynamic_domains)
            left = to_cpp(child, layout, f"({idx}) - {stride}", target_dx, state_map, dynamic_domains)
            return f"(({right}) - ({left})) / (2.0 * {target_dx})"
            
        if op == "div":
            axis_name = node.get("axis")
            target_dx = f"dx_{axis_name}" if axis_name else dx_symbol
            stride = "1"
            coord_sys = "cartesian"
            
            if axis_name and state_map:
                try:
                    state_name = extract_state_name(child, layout)
                    s_domain = state_map.get(state_name).domain
                    if s_domain:
                        ds = s_domain.domains if hasattr(s_domain, "domains") else [s_domain]
                        if len(ds) == 2 and ds[0].name == axis_name:
                            stride = str(ds[1].resolution)
                        for d in ds:
                            if d.name == axis_name:
                                coord_sys = getattr(d, "coord_sys", "cartesian")
                except ValueError:
                    pass
            
            right = to_cpp(child, layout, f"({idx}) + {stride}", target_dx, state_map, dynamic_domains)
            left = to_cpp(child, layout, f"({idx}) - {stride}", target_dx, state_map, dynamic_domains)
            
            if coord_sys == "spherical":
                r_right = f"((({idx}) + {stride}) * {target_dx})"
                r_left = f"((({idx}) - {stride}) * {target_dx})"
                r_center = f"(std::max(1e-12, (double)({idx}) * {target_dx}))"
                return f"((({r_right})*({r_right}) * ({right})) - (({r_left})*({r_left}) * ({left}))) / (({r_center})*({r_center}) * 2.0 * {target_dx})"
            else:
                return f"(({right}) - ({left})) / (2.0 * {target_dx})"
            
        if op == "integral":
            state_name = extract_state_name(child, layout)
            size = layout.state_offsets[state_name][1]
            target_dx = f"dx_{node.get('over')}" if node.get('over') else dx_symbol
            sum_expr = to_cpp(child, layout, "j", target_dx, state_map, dynamic_domains)
            return f"[&]() {{ double s = 0.0; for(int j=0; j<{size}; ++j) s += {sum_expr}; return s * {target_dx}; }}()"
            
        if op == "coords": 
            return f"({idx} * {dx_symbol})"
        
    if t == "Boundary":
        try:
            state_name = extract_state_name(node, layout)
            _, size = layout.state_offsets[state_name]
            side = node["side"]
            
            state_obj = state_map.get(state_name) if state_map else None
            if state_obj and hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
                d_mac, d_mic = state_obj.domain.domains[0], state_obj.domain.domains[1]
                bc_domain = node.get("domain")
                if bc_domain == d_mic.name:
                    if str(idx) == "0":
                        b_idx = f"{d_mic.resolution - 1}" if side == "right" else "0"
                    else:
                        b_idx = f"{idx} * {d_mic.resolution}" if side == "left" else f"{idx} * {d_mic.resolution} + {d_mic.resolution - 1}"
                    return to_cpp(node["child"], layout, b_idx, dx_symbol, state_map, dynamic_domains)
                    
            b_idx = "0" if side == "left" else f"{size - 1}"
            return to_cpp(node["child"], layout, b_idx, dx_symbol, state_map, dynamic_domains)
        except ValueError:
            return to_cpp(node["child"], layout, idx, dx_symbol, state_map, dynamic_domains)

    if t == "InitialCondition": 
        return to_cpp(node["child"], layout, idx, dx_symbol, state_map, dynamic_domains)
    
    raise ValueError(f"Unknown AST node type: {t}")

def generate_cpp(ast_payload: List[Dict[str, Any]], layout: MemoryLayout, states: List[Any], bandwidth: int = 0, target: str = "cpu") -> str:
    lines = []
    state_map = {s.name: s for s in states}
    
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

    bulk_eqs = [eq for eq in ast_payload if eq["lhs"].get("type") not in ("InitialCondition", "Boundary", "DomainBoundary")]
    boundary_eqs = [eq for eq in ast_payload if eq["lhs"].get("type") == "Boundary"]

    lines.append("    // --- Bulk PDE Residuals ---")
    for eq in bulk_eqs:
        state_name = extract_state_name(eq["lhs"], layout)
        offset, size = layout.state_offsets[state_name]
        state_obj = state_map[state_name]
        dx_sym = f"dx_{state_obj.domain.name}" if state_obj.domain and not hasattr(state_obj.domain, "domains") else "dx_default"
        
        # Level 5: OpenMP Data Parallelism injection
        omp_pragma = "    #pragma omp parallel for\n" if ("omp" in target and size > 50) else ""

        if hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
            d_mac, d_mic = state_obj.domain.domains[0], state_obj.domain.domains[1]
            lines.append(omp_pragma + f"    for (int i_mac = 0; i_mac < {d_mac.resolution}; ++i_mac) {{")
            lines.append(f"        for (int i_mic = 0; i_mic < {d_mic.resolution}; ++i_mic) {{")
            lines.append(f"            int i = i_mac * {d_mic.resolution} + i_mic;")
            lhs_cpp = to_cpp(eq["lhs"], layout, "i", dx_sym, state_map, dynamic_domains)
            rhs_cpp = to_cpp(eq["rhs"], layout, "i", dx_sym, state_map, dynamic_domains)
            lines.append(f"            res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
            lines.append(f"        }}")
            lines.append(f"    }}")
        elif size > 1:
            lines.append(omp_pragma + f"    for (int i = 0; i < {size}; ++i) {{")
            lhs_cpp = to_cpp(eq["lhs"], layout, "i", dx_sym, state_map, dynamic_domains)
            rhs_cpp = to_cpp(eq["rhs"], layout, "i", dx_sym, state_map, dynamic_domains)
            lines.append(f"        res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
            lines.append(f"    }}")
        else:
            lhs_cpp = to_cpp(eq["lhs"], layout, "0", dx_sym, state_map, dynamic_domains)
            rhs_cpp = to_cpp(eq["rhs"], layout, "0", dx_sym, state_map, dynamic_domains)
            lines.append(f"    res[{offset}] = ({lhs_cpp}) - ({rhs_cpp});")

    if boundary_eqs:
        lines.append("")
        lines.append(f"    // --- Boundary Condition Overrides ---")
        for eq in boundary_eqs:
            state_name = extract_state_name(eq["lhs"], layout)
            offset, size = layout.state_offsets[state_name]
            side = eq["lhs"]["side"]
            bc_domain = eq["lhs"].get("domain")
            state_obj = state_map[state_name]
            dx_sym = f"dx_{state_obj.domain.name}" if state_obj.domain and not hasattr(state_obj.domain, "domains") else "dx_default"
            
            if hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2 and bc_domain == state_obj.domain.domains[1].name:
                d_mac, d_mic = state_obj.domain.domains[0], state_obj.domain.domains[1]
                idx_str = f"i_mac * {d_mic.resolution}" if side == "left" else f"i_mac * {d_mic.resolution} + {d_mic.resolution - 1}"
                lines.append(f"    for (int i_mac = 0; i_mac < {d_mac.resolution}; ++i_mac) {{")
                lines.append(f"        int b_idx = {idx_str};")
                lhs_cpp = to_cpp(eq["lhs"]["child"], layout, "b_idx", dx_sym, state_map, dynamic_domains)
                rhs_cpp = to_cpp(eq["rhs"], layout, "b_idx", dx_sym, state_map, dynamic_domains)
                lines.append(f"        res[{offset} + b_idx] = ({lhs_cpp}) - ({rhs_cpp});")
                lines.append(f"    }}")
            else:
                idx = "0" if side == "left" else f"{size - 1}"
                lhs_cpp = to_cpp(eq["lhs"]["child"], layout, idx, dx_sym, state_map, dynamic_domains)
                rhs_cpp = to_cpp(eq["rhs"], layout, idx, dx_sym, state_map, dynamic_domains)
                lines.append(f"    res[{offset} + {idx}] = ({lhs_cpp}) - ({rhs_cpp});")

    body = "\n".join(lines)
    n_states = layout.n_states
    
    if bandwidth > 0:
        jacobian_logic = textwrap.dedent(f"""\
            int bw = {bandwidth};
            int stride = 2 * bw + 1;
            for (int color = 0; color < stride; ++color) {{
                for (int i = 0; i < N; ++i) {{
                    bool active = ((i % stride) == color);
                    dy[i] = active ? 1.0 : 0.0;
                    dydot[i] = active ? c_j : 0.0;
                    dres[i] = 0.0;
                }}
        #ifdef ENZYME_ACTIVE
                __enzyme_fwddiff((void*)evaluate_residual, enzyme_dup, y, dy.data(), enzyme_dup, ydot, dydot.data(), enzyme_const, p, enzyme_dup, res_dummy.data(), dres.data());
        #endif
                for (int row = 0; row < N; ++row) {{
                    int col_base = row - (row % stride) + color;
                    int actual_col = -1;
                    if (std::abs(row - col_base) <= bw) actual_col = col_base;
                    else if (std::abs(row - (col_base - stride)) <= bw) actual_col = col_base - stride;
                    else if (std::abs(row - (col_base + stride)) <= bw) actual_col = col_base + stride;

                    if (actual_col >= 0 && actual_col < N) {{
                        jac_out[actual_col * N + row] = dres[row]; // SUNDIALS Column-Major
                    }}
                }}
            }}""")
    else:
        jacobian_logic = textwrap.dedent(f"""\
            for (int col = 0; col < N; ++col) {{
                for (int i = 0; i < N; ++i) {{
                    dy[i] = (i == col) ? 1.0 : 0.0;
                    dydot[i] = (i == col) ? c_j : 0.0;
                    dres[i] = 0.0;
                }}
        #ifdef ENZYME_ACTIVE
                __enzyme_fwddiff((void*)evaluate_residual, enzyme_dup, y, dy.data(), enzyme_dup, ydot, dydot.data(), enzyme_const, p, enzyme_dup, res_dummy.data(), dres.data());
        #endif
                for (int row = 0; row < N; ++row) {{
                    jac_out[col * N + row] = dres[row]; // SUNDIALS Column-Major
                }}
            }}""")

    return f"""\
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CLAMP(idx, bound) (std::max(0, std::min((int)(idx), (int)(bound) - 1)))

#ifdef ENZYME_ACTIVE
int enzyme_dup = 1;
int enzyme_const = 2;
extern void __enzyme_fwddiff(void*, ...);
extern void __enzyme_autodiff(void*, ...);
#endif

extern "C" {{

void evaluate_residual(const double* y, const double* ydot, const double* p, double* res) {{
{body}
}}

void evaluate_jacobian(const double* y, const double* ydot, const double* p, double c_j, double* jac_out) {{
    int N = {n_states};
    std::vector<double> dy(N, 0.0);
    std::vector<double> dydot(N, 0.0);
    std::vector<double> res_dummy(N, 0.0);
    std::vector<double> dres(N, 0.0);

{textwrap.indent(jacobian_logic, '    ')}
}}

void evaluate_vjp(const double* y, const double* ydot, const double* p, const double* lambda_vec, double* dp_out, double* dy_out) {{
    int N = {n_states};
    int N_P = {layout.n_params};
    std::vector<double> res_dummy(N, 0.0);
    std::vector<double> ydot_dummy(N, 0.0);
    for(int i=0; i<N_P; ++i) dp_out[i] = 0.0;
    for(int i=0; i<N; ++i) dy_out[i] = 0.0;

#ifdef ENZYME_ACTIVE
    __enzyme_autodiff((void*)evaluate_residual, 
        enzyme_dup, y, dy_out, 
        enzyme_dup, ydot, ydot_dummy.data(), 
        enzyme_dup, p, dp_out, 
        enzyme_dup, res_dummy.data(), (double*)lambda_vec);
#endif
}}

}} // extern "C"
"""