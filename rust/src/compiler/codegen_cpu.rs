use crate::compiler::ast::{Equation, Expr};

pub fn generate_enzyme_cpp(equations: &[Equation]) -> String {
    let mut cpp = String::new();

    cpp.push_str(r#"
#include <cmath>

// Enzyme Automatic Differentiation hook (Forward/Reverse Mode)
extern void __enzyme_autodiff(void*, ...);

extern "C" {

void evaluate_residual(
    const double* y, 
    const double* ydot, 
    const double* p, 
    double* res
) {
"#);

    for (i, eq) in equations.iter().enumerate() {
        let rhs_code = to_cpp_string(&eq.rhs);
        
        match &eq.lhs {
            Expr::UnaryOp { op, child } if op == "dt" => {
                let name = extract_name(child);
                cpp.push_str(&format!("    // Eq for d({})/dt\n", name));
                cpp.push_str(&format!("    res[{}] = ydot[{}] - ({});\n", i, i, rhs_code));
            },
            Expr::Boundary { side, child } => {
                let name = extract_name(child);
                cpp.push_str(&format!("    // {} Boundary for {}\n", side, name));
                cpp.push_str(&format!("    res[{}] = {};\n", i, rhs_code));
            },
            Expr::InitialCondition { child } => {
                let name = extract_name(child);
                cpp.push_str(&format!("    // Initial Condition for {}\n", name));
                cpp.push_str(&format!("    res[{}] = {};\n", i, rhs_code));
            },
            Expr::State { name } => {
                cpp.push_str(&format!("    // Algebraic constraint for {}\n", name));
                cpp.push_str(&format!("    res[{}] = {};\n", i, rhs_code));
            },
            _ => {
                cpp.push_str(&format!("    // Residual constraint\n"));
                cpp.push_str(&format!("    res[{}] = {};\n", i, rhs_code));
            },
        }
    }

    cpp.push_str(r#"
}

void evaluate_jacobian(
    const double* y, 
    const double* ydot, 
    const double* p, 
    double c_j, 
    double* jac_out
) {
    // Enzyme call: Automatically differentiates `evaluate_residual`
    // __enzyme_autodiff((void*)evaluate_residual, ...);
}

} // extern "C"
"#);

    cpp
}

fn extract_name(expr: &Expr) -> String {
    match expr {
        Expr::State { name } => name.clone(),
        Expr::UnaryOp { child, .. } => extract_name(child),
        _ => "unknown".to_string(),
    }
}

fn to_cpp_string(expr: &Expr) -> String {
    match expr {
        Expr::Scalar { value } => format!("{:.6}", value),
        Expr::State { name } => format!("y_{}", name), 
        Expr::Parameter { name, .. } => format!("p_{}", name),
        Expr::BinaryOp { op, left, right } => {
            let l = to_cpp_string(left);
            let r = to_cpp_string(right);
            match op.as_str() {
                "add" => format!("({} + {})", l, r),
                "sub" => format!("({} - {})", l, r),
                "mul" => format!("({} * {})", l, r),
                "div" => format!("({} / {})", l, r),
                "pow" => format!("std::pow({}, {})", l, r), // Added power support
                _ => panic!("Unknown binary operator"),
            }
        },
        Expr::UnaryOp { op, child } => {
            let c = to_cpp_string(child);
            match op.as_str() {
                "neg" => format!("(-{})", c),
                "grad" => format!("GRAD({})", c),
                "div" => format!("DIV({})", c),
                "abs" => format!("std::abs({})", c),
                "coords" => format!("X_COORD"),
                _ => format!("{}({})", op, c),
            }
        },
        Expr::Boundary { side, child } => {
            let c = to_cpp_string(child);
            format!("BOUNDARY_{}({})", side.to_uppercase(), c)
        },
        Expr::InitialCondition { child } => {
            to_cpp_string(child)
        }
    }
}