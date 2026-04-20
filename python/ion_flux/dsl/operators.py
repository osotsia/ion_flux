from typing import Optional
from .core import UnaryOp, BinaryOp, Node, _wrap, Domain

# Differential / Integral Operators
def dt(state: Node) -> UnaryOp: 
    """Partial derivative with respect to time."""
    return UnaryOp("dt", state)

def grad(state: Node, axis: Optional[Domain] = None) -> UnaryOp: 
    """Topology-agnostic spatial gradient. Defaults to dominant domain if axis is None."""
    return UnaryOp("grad", state, axis=axis.name if axis else None)

def div(expr: Node, axis: Optional[Domain] = None) -> UnaryOp: 
    """Topology-agnostic spatial divergence."""
    return UnaryOp("div", expr, axis=axis.name if axis else None)

def integral(expr: Node, over: Optional[Domain] = None) -> UnaryOp:
    """Definite integral over a specified spatial domain."""
    return UnaryOp("integral", _wrap(expr), over=over)

# Standard Math Operators
def abs_val(expr: Node) -> UnaryOp: return UnaryOp("abs", _wrap(expr))
def exp(expr: Node) -> UnaryOp: return UnaryOp("exp", _wrap(expr))
def log(expr: Node) -> UnaryOp: return UnaryOp("log", _wrap(expr))
def sqrt(expr: Node) -> UnaryOp: return UnaryOp("sqrt", _wrap(expr))
def sin(expr: Node) -> UnaryOp: return UnaryOp("sin", _wrap(expr))
def cos(expr: Node) -> UnaryOp: return UnaryOp("cos", _wrap(expr))

def maximum(a: Node, b: Node) -> BinaryOp: 
    """Element-wise maximum of two AST nodes."""
    return BinaryOp("max", _wrap(a), _wrap(b))

def minimum(a: Node, b: Node) -> BinaryOp: 
    """Element-wise minimum of two AST nodes."""
    return BinaryOp("min", _wrap(a), _wrap(b))