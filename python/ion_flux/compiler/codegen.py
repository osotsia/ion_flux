from typing import List, Dict, Any
import textwrap
from .memory import MemoryLayout

def extract_state_name(node: Dict[str, Any]) -> str:
    """Recursively walks down AST wrappers to find the base State name driving an equation."""
    t = node.get("type")
    if t == "State":
        return node["name"]
    if t in ("UnaryOp", "Boundary", "InitialCondition"):
        return extract_state_name(node["child"])
    raise ValueError(f"Cannot extract state name from LHS node: {node}")

def to_cpp(node: Dict[str, Any], layout: MemoryLayout, idx: str = "0", dx_symbol: str = "1.0") -> str:
    """
    Recursively lowers the Python AST into C++ code strings. 
    Injects spatial awareness by routing physical `dx` symbols to differential operators.
    """
    t = node.get("type")
    
    # Base Cases
    if t == "Scalar": return str(node["value"])
    if t == "Parameter": return f"p[{layout.get_param_offset(node['name'])}]"
    if t == "State": 
        offset, size = layout.state_offsets[node["name"]]
        return f"y[{offset} + CLAMP({idx}, {size})]"
        
    # Recursive Operations
    if t == "BinaryOp":
        l = to_cpp(node["left"], layout, idx, dx_symbol)
        r = to_cpp(node["right"], layout, idx, dx_symbol)
        op = node["op"]
        
        # Standard Arithmetic
        if op == "add": return f"({l} + {r})"
        if op == "sub": return f"({l} - {r})"
        if op == "mul": return f"({l} * {r})"
        if op == "div": return f"({l} / {r})"
        if op == "pow": return f"std::pow({l}, {r})"
        
        # Piecewise bounds
        if op == "max": return f"std::max((double)({l}), (double)({r}))"
        if op == "min": return f"std::min((double)({l}), (double)({r}))"
        
        # Relational Ops (returns 1.0 for True, 0.0 for False)
        if op == "gt": return f"(({l}) > ({r}) ? 1.0 : 0.0)"
        if op == "lt": return f"(({l}) < ({r}) ? 1.0 : 0.0)"
        if op == "ge": return f"(({l}) >= ({r}) ? 1.0 : 0.0)"
        if op == "le": return f"(({l}) <= ({r}) ? 1.0 : 0.0)"
        if op == "eq": return f"(({l}) == ({r}) ? 1.0 : 0.0)"
        if op == "ne": return f"(({l}) != ({r}) ? 1.0 : 0.0)"
        
    if t == "UnaryOp":
        op = node["op"]
        child = node["child"]
        
        if op == "neg": return f"(-{to_cpp(child, layout, idx, dx_symbol)})"
        if op == "abs": return f"std::abs({to_cpp(child, layout, idx, dx_symbol)})"
        if op == "exp": return f"std::exp({to_cpp(child, layout, idx, dx_symbol)})"
        if op == "log": return f"std::log({to_cpp(child, layout, idx, dx_symbol)})"
        if op == "sin": return f"std::sin({to_cpp(child, layout, idx, dx_symbol)})"
        if op == "cos": return f"std::cos({to_cpp(child, layout, idx, dx_symbol)})"
        
        if op == "dt": 
            offset, size = layout.state_offsets[child["name"]]
            return f"ydot[{offset} + CLAMP({idx}, {size})]"
            
        # Physical Spatial Operators (Central Difference Collocated Grid)
        if op in ("grad", "div"):
            target_dx = f"dx_{node.get('axis', 'default')}" if "axis" in node else dx_symbol
            right = to_cpp(child, layout, f"({idx})+1", target_dx)
            left = to_cpp(child, layout, f"({idx})-1", target_dx)
            return f"(({right}) - ({left})) / (2.0 * {target_dx})"
            
        if op == "integral":
            state_name = extract_state_name(child)
            size = layout.state_offsets[state_name][1]
            target_dx = f"dx_{node.get('over', 'default')}" if "over" in node else dx_symbol
            sum_expr = to_cpp(child, layout, "j", target_dx)
            return f"[&]() {{ double s = 0.0; for(int j=0; j<{size}; ++j) s += {sum_expr}; return s * {target_dx}; }}()"
            
        if op == "coords": 
            return f"({idx} * {dx_symbol})"
        
    if t == "Boundary": return to_cpp(node["child"], layout, idx, dx_symbol)
    if t == "InitialCondition": return to_cpp(node["child"], layout, idx, dx_symbol)
    
    raise ValueError(f"Unknown AST node type: {t}")


def generate_cpp(ast_payload: List[Dict[str, Any]], layout: MemoryLayout, states: List[Any], bandwidth: int = 0) -> str:
    """Emits the Topology-Aware C++ residual and Enzyme LLVM intrinsics."""
    lines = []
    
    # 1. Generate Physical Constants (dx mappings from Domain Topology)
    lines.append("    // --- Physical Domain Constants ---")
    lines.append("    double dx_default = 1.0;")
    for state in states:
        if state.domain:
            L = state.domain.bounds[1] - state.domain.bounds[0]
            N = state.domain.resolution
            dx = L / (N - 1) if N > 1 else L
            lines.append(f"    double dx_{state.domain.name} = {dx};")
    lines.append("")

    bulk_eqs = [eq for eq in ast_payload if eq["lhs"].get("type") not in ("InitialCondition", "Boundary")]
    boundary_eqs = [eq for eq in ast_payload if eq["lhs"].get("type") == "Boundary"]

    # 2. Generate Bulk Spatial Loops
    lines.append("    // --- Bulk PDE Residuals ---")
    for eq in bulk_eqs:
        state_name = extract_state_name(eq["lhs"])
        offset, size = layout.state_offsets[state_name]
        
        # Determine driving domain for this equation to pass the correct dx symbol
        state_obj = next(s for s in states if s.name == state_name)
        dx_sym = f"dx_{state_obj.domain.name}" if state_obj.domain else "dx_default"
        
        if size > 1:
            lines.append(f"    for (int i = 0; i < {size}; ++i) {{")
            lhs_cpp = to_cpp(eq["lhs"], layout, "i", dx_sym)
            rhs_cpp = to_cpp(eq["rhs"], layout, "i", dx_sym)
            lines.append(f"        res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
            lines.append(f"    }}")
        else:
            lhs_cpp = to_cpp(eq["lhs"], layout, "0", dx_sym)
            rhs_cpp = to_cpp(eq["rhs"], layout, "0", dx_sym)
            lines.append(f"    res[{offset}] = ({lhs_cpp}) - ({rhs_cpp});")

    # 3. Generate Boundary Overrides
    if boundary_eqs:
        lines.append("")
        lines.append(f"    // --- Boundary Condition Overrides ---")
        for eq in boundary_eqs:
            state_name = extract_state_name(eq["lhs"])
            offset, size = layout.state_offsets[state_name]
            side = eq["lhs"]["side"]
            idx = "0" if side == "left" else f"{size - 1}"
            
            state_obj = next(s for s in states if s.name == state_name)
            dx_sym = f"dx_{state_obj.domain.name}" if state_obj.domain else "dx_default"
            
            lhs_cpp = to_cpp(eq["lhs"]["child"], layout, idx, dx_sym)
            rhs_cpp = to_cpp(eq["rhs"], layout, idx, dx_sym)
            lines.append(f"    res[{offset} + {idx}] = ({lhs_cpp}) - ({rhs_cpp});")

    body = "\n".join(lines)
    n_states = layout.n_states
    
    # CPR Graph Coloring for Sparse Analytical Jacobians
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

// Macro to clamp spatial indices safely to boundaries
#define CLAMP(idx, bound) (std::max(0, std::min((int)(idx), (int)(bound) - 1)))

#ifdef ENZYME_ACTIVE
int enzyme_dup = 1;
int enzyme_const = 2;
extern void __enzyme_fwddiff(void*, ...);
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

}} // extern "C"
"""