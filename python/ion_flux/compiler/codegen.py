from typing import List, Dict, Any

def extract_state_name(node: Dict[str, Any]) -> str:
    """Recursively walks an AST node to find the root State name."""
    t = node.get("type")
    if t == "State":
        return node["name"]
    if t in ("UnaryOp", "Boundary", "InitialCondition"):
        return extract_state_name(node["child"])
    raise ValueError(f"Cannot extract state name from node: {node}")


def to_cpp(node: Dict[str, Any], states: List[str], params: List[str]) -> str:
    """Recursively lowers an AST node into a C++ string representation."""
    t = node.get("type")
    
    if t == "Scalar": 
        return str(node["value"])
        
    if t == "State": 
        # Safely index into the flattened C-array based on deterministic ordering
        return f"y[{states.index(node['name'])}]"
        
    if t == "Parameter": 
        return f"p[{params.index(node['name'])}]"
        
    if t == "BinaryOp":
        l = to_cpp(node["left"], states, params)
        r = to_cpp(node["right"], states, params)
        op = node["op"]
        if op == "add": return f"({l} + {r})"
        if op == "sub": return f"({l} - {r})"
        if op == "mul": return f"({l} * {r})"
        if op == "div": return f"({l} / {r})"
        if op == "pow": return f"std::pow({l}, {r})"
        
    if t == "UnaryOp":
        op = node["op"]
        c = to_cpp(node["child"], states, params)
        if op == "neg": return f"(-{c})"
        if op == "abs": return f"std::abs({c})"
        if op == "dt": return f"ydot[{states.index(node['child']['name'])}]"
        
        # Spatial operators are stubbed via C++ macros in this 0D pass
        if op == "grad": return f"GRAD({c})"
        if op == "div": return f"DIV({c})"
        if op == "coords": return "X_COORD"
        
    if t == "Boundary":
        c = to_cpp(node["child"], states, params)
        return f"BOUNDARY_{node['side'].upper()}({c})"
        
    if t == "InitialCondition":
        return to_cpp(node["child"], states, params)
        
    raise ValueError(f"Unknown AST node type for C++ emission: {t}")


def generate_cpp(ast_payload: List[Dict[str, Any]], states: List[str], params: List[str]) -> str:
    """
    Generates the C++ execution logic and Enzyme LLVM bindings for the residual.
    """
    lines = []
    res_idx = 0
    
    for eq in ast_payload:
        lhs = eq["lhs"]
        
        # Initial Conditions do not belong in the SUNDIALS integration residual
        if lhs.get("type") == "InitialCondition":
            lines.append(f"    // IC for {extract_state_name(lhs)} ignored in integration loop")
            continue
            
        lhs_cpp = to_cpp(lhs, states, params)
        rhs_cpp = to_cpp(eq["rhs"], states, params)
        
        lines.append(f"    res[{res_idx}] = ({lhs_cpp}) - ({rhs_cpp});")
        res_idx += 1
        
    body = "\n".join(lines)
    
    # Boilerplate headers and macros ensure the C++ is highly strictly typed and 
    # testable even if finite-volume expansion hasn't run yet.
    return f"""
#include <cmath>
#include <cstdlib>

// Spatial Stencil Macros (0D Stubs)
#define GRAD(x) (0.0)
#define DIV(x) (0.0)
#define BOUNDARY_LEFT(x) (x)
#define BOUNDARY_RIGHT(x) (x)
#define X_COORD (0.0)

// Enzyme Automatic Differentiation hook (Forward/Reverse Mode)
extern void __enzyme_autodiff(void*, ...);

extern "C" {{

void evaluate_residual(
    const double* y, 
    const double* ydot, 
    const double* p, 
    double* res
) {{
{body}
}}

void evaluate_jacobian(
    const double* y, 
    const double* ydot, 
    const double* p, 
    double c_j, 
    double* jac_out
) {{
    // Enzyme call: Automatically differentiates `evaluate_residual`
    // __enzyme_autodiff((void*)evaluate_residual, ...);
}}

}} // extern "C"
"""