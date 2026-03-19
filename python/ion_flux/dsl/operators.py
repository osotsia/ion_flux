from .core import UnaryOp, Node, _wrap

def dt(state: Node) -> UnaryOp: 
    """Represents the partial derivative with respect to time."""
    return UnaryOp("dt", state)

def grad(state: Node) -> UnaryOp: 
    """Represents the spatial gradient operator."""
    return UnaryOp("grad", state)

def div(expr: Node) -> UnaryOp: 
    """Represents the spatial divergence operator."""
    return UnaryOp("div", expr)

def abs_val(expr: Node) -> UnaryOp: 
    """Represents the absolute value operator."""
    return UnaryOp("abs", _wrap(expr))
