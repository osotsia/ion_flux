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

def to_cpp(node: Dict[str, Any], layout: MemoryLayout, idx: str = "0") -> str:
    """
    Recursively lowers the Python AST into C++ code strings. 
    The `idx` parameter acts as a compile-time string macro, allowing spatial 
    operators (like grad) to offset the evaluation index of their entire subtree.
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
        l = to_cpp(node["left"], layout, idx)
        r = to_cpp(node["right"], layout, idx)
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
        
        # Standard Math
        if op == "neg": return f"(-{to_cpp(child, layout, idx)})"
        if op == "abs": return f"std::abs({to_cpp(child, layout, idx)})"
        if op == "exp": return f"std::exp({to_cpp(child, layout, idx)})"
        if op == "log": return f"std::log({to_cpp(child, layout, idx)})"
        if op == "sin": return f"std::sin({to_cpp(child, layout, idx)})"
        if op == "cos": return f"std::cos({to_cpp(child, layout, idx)})"
        
        # Differential Time State
        if op == "dt": 
            offset, size = layout.state_offsets[child["name"]]
            return f"ydot[{offset} + CLAMP({idx}, {size})]"
            
        # Spatial Operators
        if op in ("grad", "div"):
            right = to_cpp(child, layout, f"({idx})+1")
            left = to_cpp(child, layout, f"({idx})-1")
            return f"(({right}) - ({left})) / (2.0 * dx)"
            
        if op == "integral":
            state_name = extract_state_name(child)
            size = layout.state_offsets[state_name][1]
            sum_expr = to_cpp(child, layout, "j")
            return f"[&]() {{ double s = 0.0; for(int j=0; j<{size}; ++j) s += {sum_expr}; return s * dx; }}()"
            
        if op == "coords": 
            return f"({idx} * dx)"
        
    if t == "Boundary": return to_cpp(node["child"], layout, idx)
    if t == "InitialCondition": return to_cpp(node["child"], layout, idx)
    
    raise ValueError(f"Unknown AST node type: {t}")


def generate_cpp(ast_payload: List[Dict[str, Any]], layout: MemoryLayout, bandwidth: int = 0) -> str:
    """Emits the C++ residual function and the Enzyme LLVM intrinsic hooks for the Jacobian."""
    lines = []
    
    # Separate Bulk PDEs from Boundary Condition Overrides
    bulk_eqs = []
    boundary_eqs = []
    
    for eq in ast_payload:
        lhs = eq["lhs"]
        if lhs.get("type") == "InitialCondition":
            continue
        if lhs.get("type") == "Boundary":
            boundary_eqs.append(eq)
        else:
            bulk_eqs.append(eq)

    # 1. Generate Bulk Spatial Loops
    for eq in bulk_eqs:
        state_name = extract_state_name(eq["lhs"])
        offset, size = layout.state_offsets[state_name]
        
        if size > 1:
            lines.append(f"    for (int i = 0; i < {size}; ++i) {{")
            lhs_cpp = to_cpp(eq["lhs"], layout, "i")
            rhs_cpp = to_cpp(eq["rhs"], layout, "i")
            lines.append(f"        res[{offset} + i] = ({lhs_cpp}) - ({rhs_cpp});")
            lines.append(f"    }}")
        else:
            lhs_cpp = to_cpp(eq["lhs"], layout, "0")
            rhs_cpp = to_cpp(eq["rhs"], layout, "0")
            lines.append(f"    res[{offset}] = ({lhs_cpp}) - ({rhs_cpp});")

    # 2. Generate Boundary Overrides
    if boundary_eqs:
        lines.append(f"    // Boundary Condition Overrides")
        for eq in boundary_eqs:
            state_name = extract_state_name(eq["lhs"])
            offset, size = layout.state_offsets[state_name]
            side = eq["lhs"]["side"]
            idx = "0" if side == "left" else f"{size - 1}"
            
            lhs_cpp = to_cpp(eq["lhs"]["child"], layout, idx)
            rhs_cpp = to_cpp(eq["rhs"], layout, idx)
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

// Macro to clamp spatial indices safely to boundaries. 
// Bound parameter explicitly cast to int to avoid template deduction conflicts.
#define CLAMP(idx, bound) (std::max(0, std::min((int)(idx), (int)(bound) - 1)))

#ifdef ENZYME_ACTIVE
int enzyme_dup = 1;
int enzyme_const = 2;
extern void __enzyme_fwddiff(void*, ...);
#endif

extern "C" {{

void evaluate_residual(const double* y, const double* ydot, const double* p, double* res) {{
    double dx = 1e-6; // Topology scale stub.

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