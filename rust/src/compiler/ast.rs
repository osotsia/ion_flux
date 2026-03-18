use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Expr {
    Scalar { value: f64 },
    State { name: String },
    Parameter { name: String, default: f64 },
    BinaryOp { op: String, left: Box<Expr>, right: Box<Expr> },
    UnaryOp { op: String, child: Box<Expr> },
    Boundary { side: String, child: Box<Expr> },
    InitialCondition { child: Box<Expr> },
}

#[derive(Deserialize, Debug)]
pub struct Equation {
    pub lhs: Expr,
    pub rhs: Expr,
}