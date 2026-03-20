from typing import List, Dict, Any
import textwrap
from .memory import MemoryLayout

def extract_state_name(node: Dict[str, Any]) -> str:
    """Recursively walks an AST node to find the root State name."""
    t = node.get("type")
    if t == "State":
        return node["name"]
    if t in ("UnaryOp", "Boundary", "InitialCondition"):
        return extract_state_name(node["child"])
    raise ValueError(f"Cannot extract state name from node: {node}")


def to_cpp(node: Dict[str, Any], layout: MemoryLayout) -> str:
    """Recursively lowers an AST node into a C++ string utilizing the flat MemoryLayout."""
    t = node.get("type")
    
    if t == "Scalar": 
        return str(node["value"])
        
    if t == "State": 
        # Map state to its calculated C-array offset
        offset = layout.get_state_offset(node['name'])
        return f"y[{offset}]"
        
    if t == "Parameter": 
        offset = layout.get_param_offset(node['name'])
        return f"p[{offset}]"
        
    if t == "BinaryOp":
        l = to_cpp(node["left"], layout)
        r = to_cpp(node["right"], layout)
        op = node["op"]
        if op == "add": return f"({l} + {r})"
        if op == "sub": return f"({l} - {r})"
        if op == "mul": return f"({l} * {r})"
        if op == "div": return f"({l} / {r})"
        if op == "pow": return f"std::pow({l}, {r})"
        
    if t == "UnaryOp":
        op = node["op"]
        c = to_cpp(node["child"], layout)
        if op == "neg": return f"(-{c})"
        if op == "abs": return f"std::abs({c})"
        if op == "dt": 
            offset = layout.get_state_offset(node['child']['name'])
            return f"ydot[{offset}]"
        
        # Spatial operators remain stubbed until the full FVM traversal pass is built
        if op == "grad": return f"GRAD({c})"
        if op == "div": return f"DIV({c})"
        if op == "coords": return "X_COORD"
        
    if t == "Boundary":
        c = to_cpp(node["child"], layout)
        return f"BOUNDARY_{node['side'].upper()}({c})"
        
    if t == "InitialCondition":
        return to_cpp(node["child"], layout)
        
    raise ValueError(f"Unknown AST node type for C++ emission: {t}")


def generate_cpp(ast_payload: List[Dict[str, Any]], layout: MemoryLayout, bandwidth: int = 0) -> str:
    """
    Generates the C++ execution logic.
    Injects Curtis-Powell-Reid graph coloring if a Jacobian bandwidth is provided,
    reducing AD cost from O(N) to O(2b+1) forward passes.
    """
    lines = []
    res_idx = 0
    
    for eq in ast_payload:
        lhs = eq["lhs"]
        if lhs.get("type") == "InitialCondition":
            lines.append(f"    // IC for {extract_state_name(lhs)} ignored in integration loop")
            continue
            
        lhs_cpp = to_cpp(lhs, layout)
        rhs_cpp = to_cpp(eq["rhs"], layout)
        
        lines.append(f"    res[{res_idx}] = ({lhs_cpp}) - ({rhs_cpp});")
        res_idx += 1
        
    body = "\n".join(lines)
    n_states = layout.n_states
    
    # Conditional macro injection for CPR Banded Seeding vs Dense Seeding
    if bandwidth > 0:
        jacobian_logic = textwrap.dedent(f"""\
            // CPR Graph Coloring: Computing Sparse Banded Jacobian in O(2b+1) passes
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
                __enzyme_fwddiff(
                    (void*)evaluate_residual,
                    enzyme_dup, y, dy.data(),
                    enzyme_dup, ydot, dydot.data(),
                    enzyme_const, p,
                    enzyme_dup, res_dummy.data(), dres.data()
                );
        #endif

                // Unpack sparse directional derivative into the dense C-array layout
                for (int row = 0; row < N; ++row) {{
                    int col_base = row - (row % stride) + color;
                    int actual_col = -1;
                    
                    if (std::abs(row - col_base) <= bw) actual_col = col_base;
                    else if (std::abs(row - (col_base - stride)) <= bw) actual_col = col_base - stride;
                    else if (std::abs(row - (col_base + stride)) <= bw) actual_col = col_base + stride;

                    if (actual_col >= 0 && actual_col < N) {{
                        jac_out[row * N + actual_col] = dres[row];
                    }}
                }}
            }}""")
    else:
        jacobian_logic = textwrap.dedent(f"""\
            // Dense Jacobian Seeding: O(N) passes
            for (int col = 0; col < N; ++col) {{
                for (int i = 0; i < N; ++i) {{
                    dy[i] = (i == col) ? 1.0 : 0.0;
                    dydot[i] = (i == col) ? c_j : 0.0;
                    dres[i] = 0.0;
                }}

        #ifdef ENZYME_ACTIVE
                __enzyme_fwddiff(
                    (void*)evaluate_residual,
                    enzyme_dup, y, dy.data(),
                    enzyme_dup, ydot, dydot.data(),
                    enzyme_const, p,
                    enzyme_dup, res_dummy.data(), dres.data()
                );
        #endif

                for (int row = 0; row < N; ++row) {{
                    jac_out[row * N + col] = dres[row];
                }}
            }}""")

    return f"""\
#include <cmath>
#include <cstdlib>
#include <vector>

#define GRAD(x) (0.0)
#define DIV(x) (0.0)
#define BOUNDARY_LEFT(x) (x)
#define BOUNDARY_RIGHT(x) (x)
#define X_COORD (0.0)

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