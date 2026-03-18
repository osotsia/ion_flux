from typing import Dict, Any

class Node:
    """Base class for all AST nodes. Implements mathematical tracing."""
    def __add__(self, other): return BinaryOp("add", self, _wrap(other))
    def __radd__(self, other): return BinaryOp("add", _wrap(other), self)
    def __sub__(self, other): return BinaryOp("sub", self, _wrap(other))
    def __rsub__(self, other): return BinaryOp("sub", _wrap(other), self)
    def __mul__(self, other): return BinaryOp("mul", self, _wrap(other))
    def __rmul__(self, other): return BinaryOp("mul", _wrap(other), self)
    def __truediv__(self, other): return BinaryOp("div", self, _wrap(other))
    def __rtruediv__(self, other): return BinaryOp("div", _wrap(other), self)
    def __pow__(self, other): return BinaryOp("pow", self, _wrap(other))
    def __rpow__(self, other): return BinaryOp("pow", _wrap(other), self)
    def __neg__(self): return UnaryOp("neg", self)
    
    @property
    def left(self): return Boundary(self, "left")
    @property
    def right(self): return Boundary(self, "right")
    @property
    def t0(self): return InitialCondition(self)

    def to_dict(self) -> Dict[str, Any]: 
        raise NotImplementedError

def _wrap(val):
    if isinstance(val, (int, float)): return Scalar(val)
    if not isinstance(val, Node): raise TypeError(f"Cannot wrap {type(val)}")
    return val

class Scalar(Node):
    def __init__(self, value: float): self.value = float(value)
    def to_dict(self): return {"type": "Scalar", "value": self.value}

class State(Node):
    def __init__(self, domain=None, name: str = ""):
        self.domain = domain
        self.name = name
    def to_dict(self): return {"type": "State", "name": self.name}

class Parameter(Node):
    def __init__(self, default: float, name: str = ""):
        self.default = default
        self.name = name
    def to_dict(self): return {"type": "Parameter", "name": self.name, "default": self.default}

class BinaryOp(Node):
    def __init__(self, op: str, left_node: Node, right_node: Node):
        self.op = op
        self.left_node = left_node    # Fixed property collision
        self.right_node = right_node  # Fixed property collision
    def to_dict(self): 
        return {"type": "BinaryOp", "op": self.op, "left": self.left_node.to_dict(), "right": self.right_node.to_dict()}

class UnaryOp(Node):
    def __init__(self, op: str, child: Node):
        self.op = op
        self.child = child
    def to_dict(self): return {"type": "UnaryOp", "op": self.op, "child": self.child.to_dict()}

class Boundary(Node):
    def __init__(self, child: Node, side: str):
        self.child = child
        self.side = side
    def to_dict(self): return {"type": "Boundary", "side": self.side, "child": self.child.to_dict()}

class InitialCondition(Node):
    def __init__(self, child: Node):
        self.child = child
    def to_dict(self): return {"type": "InitialCondition", "child": self.child.to_dict()}

class Domain:
    def __init__(self, bounds: tuple, resolution: int):
        self.bounds = bounds
        self.resolution = resolution
    @property
    def coords(self):
        return UnaryOp("coords", Scalar(0.0))

def dt(state: Node) -> UnaryOp: return UnaryOp("dt", state)
def grad(state: Node) -> UnaryOp: return UnaryOp("grad", state)
def div(expr: Node) -> UnaryOp: return UnaryOp("div", expr)
def abs_val(expr: Node) -> UnaryOp: return UnaryOp("abs", _wrap(expr))

class PDE:
    def __init__(self):
        # Metaclass-like auto-binding of variable names for clean ASTs
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, (State, Parameter)):
                setattr(self, name, attr.__class__(**{**attr.__dict__, "name": name}))

    def math(self) -> Dict[Node, Node]:
        raise NotImplementedError("Models must implement the math() method.")

    def ast(self) -> list:
        equations = self.math()
        return [{"lhs": lhs.to_dict(), "rhs": _wrap(rhs).to_dict()} for lhs, rhs in equations.items()]