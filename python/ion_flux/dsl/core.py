from typing import Dict, Any, Union
import copy

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
        raise NotImplementedError("Subclasses must implement to_dict()")


def _wrap(val: Union[int, float, Node]) -> Node:
    """Safely coerces Python primitives into AST Nodes."""
    if isinstance(val, (int, float)): 
        return Scalar(float(val))
    if isinstance(val, Node):
        return val
    raise TypeError(f"Cannot wrap type {type(val)} into an AST Node.")


class Scalar(Node):
    def __init__(self, value: float): 
        self.value = value
        
    def to_dict(self): 
        return {"type": "Scalar", "value": self.value}
        
    def __repr__(self):
        return f"{self.value}"


class State(Node):
    def __init__(self, domain=None, name: str = ""):
        self.domain = domain
        self.name = name
        
    def to_dict(self): 
        return {"type": "State", "name": self.name}
        
    def __repr__(self):
        return self.name or "<Unbound State>"


class Parameter(Node):
    def __init__(self, default: float, name: str = ""):
        self.default = default
        self.name = name
        
    def to_dict(self): 
        return {"type": "Parameter", "name": self.name, "default": self.default}
        
    def __repr__(self):
        return self.name or f"<Unbound Param: {self.default}>"


class BinaryOp(Node):
    def __init__(self, op: str, left_node: Node, right_node: Node):
        self.op = op
        self.left_node = left_node
        self.right_node = right_node
        
    def to_dict(self): 
        return {
            "type": "BinaryOp", 
            "op": self.op, 
            "left": self.left_node.to_dict(), 
            "right": self.right_node.to_dict()
        }
        
    def __repr__(self):
        return f"({self.left_node} {self.op} {self.right_node})"


class UnaryOp(Node):
    def __init__(self, op: str, child: Node):
        self.op = op
        self.child = child
        
    def to_dict(self): 
        return {"type": "UnaryOp", "op": self.op, "child": self.child.to_dict()}
        
    def __repr__(self):
        return f"{self.op}({self.child})"


class Boundary(Node):
    def __init__(self, child: Node, side: str):
        self.child = child
        self.side = side
        
    def to_dict(self): 
        return {"type": "Boundary", "side": self.side, "child": self.child.to_dict()}
        
    def __repr__(self):
        return f"{self.child}.{self.side}"


class InitialCondition(Node):
    def __init__(self, child: Node):
        self.child = child
        
    def to_dict(self): 
        return {"type": "InitialCondition", "child": self.child.to_dict()}
        
    def __repr__(self):
        return f"{self.child}.t0"


class Domain:
    def __init__(self, bounds: tuple, resolution: int, coord_sys: str = "cartesian"):
        self.bounds = bounds
        self.resolution = resolution
        self.coord_sys = coord_sys
        
    @property
    def coords(self) -> Node:
        return UnaryOp("coords", Scalar(0.0))


class PDE:
    """
    Declarative base class for defining Partial Differential Equations.
    Introspects class attributes to bind AST variables to their defined names.
    """
    def __init__(self, **kwargs):
        self._bind_declarations()
        
    def _bind_declarations(self):
        """
        Deep-copies class-level States and Parameters to the instance level.
        Prevents state-leakage between concurrent PDE instances.
        """
        for name in dir(self.__class__):
            if name.startswith("__"):
                continue
            
            attr = getattr(self.__class__, name)
            if isinstance(attr, (State, Parameter)):
                # Clone the node to ensure instance-level isolation
                clone = copy.copy(attr)
                clone.name = name
                setattr(self, name, clone)

    def math(self) -> Dict[Node, Node]:
        """Defines the equations governing the PDE system."""
        raise NotImplementedError("Models must implement the math() method.")

    def ast(self) -> list:
        """Serializes the mathematical intent into a flat list of dictionaries."""
        equations = self.math()
        return [
            {"lhs": lhs.to_dict(), "rhs": _wrap(rhs).to_dict()} 
            for lhs, rhs in equations.items()
        ]
