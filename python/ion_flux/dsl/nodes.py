from typing import Dict, Any, Union, List, Optional, TypedDict
SystemDict = TypedDict('SystemDict', {
    'regions': Dict[Any, List['BinaryOp']],
    'global': List['BinaryOp'],
    'boundaries': List['BinaryOp']
}, total=False)

class Node:
    # Standard Arithmetic
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
    
    # Relational Operators
    def __lt__(self, other): return BinaryOp("lt", self, _wrap(other))
    def __le__(self, other): return BinaryOp("le", self, _wrap(other))
    def __gt__(self, other): return BinaryOp("gt", self, _wrap(other))
    def __ge__(self, other): return BinaryOp("ge", self, _wrap(other))
    def __eq__(self, other): return BinaryOp("eq", self, _wrap(other))
    def __ne__(self, other): return BinaryOp("ne", self, _wrap(other))

    def __hash__(self) -> int:
        return id(self)

    @property
    def left(self) -> "Boundary": return Boundary(self, "left")
    
    @property
    def right(self) -> "Boundary": return Boundary(self, "right")
    
    @property
    def t0(self) -> "InitialCondition": return InitialCondition(self)

    def boundary(self, tag: str, domain: Optional["Domain"] = None) -> "Boundary": 
        return Boundary(self, tag, domain=domain)

    def to_dict(self) -> Dict[str, Any]: 
        raise NotImplementedError("Subclasses must implement to_dict()")



def _wrap(val: Union[int, float, Node]) -> Node:
    if isinstance(val, (int, float)): return Scalar(float(val))
    if isinstance(val, Node): return val
    raise TypeError(f"Cannot wrap type {type(val).__name__} into an AST Node.")



class Scalar(Node):
    __slots__ = ["value"]
    def __init__(self, value: float): 
        self.value = value
    def to_dict(self) -> Dict[str, Any]: 
        return {"type": "Scalar", "value": self.value}
    def __repr__(self) -> str:
        return str(self.value)



class State(Node):
    __slots__ = ["domain", "name", "_original_name"]
    def __init__(self, domain: Optional["Domain"] = None, name: str = ""):
        self.domain = domain
        self.name = name
    def to_dict(self) -> Dict[str, Any]: 
        return {"type": "State", "name": self.name}
    def __repr__(self) -> str:
        return self.name or "<Unbound State>"
    def __set_name__(self, owner, name):
        if not self.name:
            self.name = name



class Parameter(Node):
    __slots__ = ["default", "name", "_original_name"]
    def __init__(self, default: float, name: str = ""):
        self.default = default
        self.name = name
    def to_dict(self) -> Dict[str, Any]: 
        return {"type": "Parameter", "name": self.name, "default": self.default}
    def __repr__(self) -> str:
        return self.name or f"<Unbound Param: {self.default}>"
    def __set_name__(self, owner, name):
        if not self.name:
            self.name = name



class BinaryOp(Node):
    __slots__ = ["op", "left_node", "right_node"]
    
    _SYM_MAP = {
        "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**",
        "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="
    }
    
    def __init__(self, op: str, left_node: Node, right_node: Node):
        self.op = op
        self.left_node = left_node
        self.right_node = right_node
    def to_dict(self) -> Dict[str, Any]: 
        return {
            "type": "BinaryOp", 
            "op": self.op, 
            "left": self.left_node.to_dict(), 
            "right": self.right_node.to_dict()
        }
    def __repr__(self) -> str:
        symbol = self._SYM_MAP.get(self.op, self.op)
        return f"({self.left_node} {symbol} {self.right_node})"



class UnaryOp(Node):
    __slots__ = ["op", "child", "kwargs"]
    def __init__(self, op: str, child: Node, **kwargs):
        self.op = op
        self.child = child
        self.kwargs = kwargs
    def to_dict(self) -> Dict[str, Any]: 
        d = {"type": "UnaryOp", "op": self.op, "child": self.child.to_dict()}
        for k, v in self.kwargs.items():
            if v is not None: d[k] = v.name if hasattr(v, "name") else str(v)
        return d
    def __repr__(self) -> str: 
        return f"{self.op}({self.child})"



class Boundary(Node):
    __slots__ = ["child", "side", "domain"]
    def __init__(self, child: Node, side: str, domain: Optional["Domain"] = None):
        self.child = child
        self.side = side
        self.domain = domain
        
    def __call__(self, domain: Optional["Domain"] = None) -> "Boundary":
        """Allows flexible syntactic access (e.g., flux.right() or flux.right(domain=r))"""
        return Boundary(self.child, self.side, domain=(domain or self.domain))

    def to_dict(self) -> Dict[str, Any]: 
        d = {"type": "Boundary", "side": self.side, "child": self.child.to_dict()}
        if self.domain:
            d["domain"] = self.domain.name
        return d
    def __repr__(self) -> str:
        return f"{self.child}.{self.side}"



class InitialCondition(Node):
    __slots__ = ["child"]
    def __init__(self, child: Node):
        self.child = child
    def __call__(self) -> "InitialCondition":
        return self
    def to_dict(self) -> Dict[str, Any]: 
        return {"type": "InitialCondition", "child": self.child.to_dict()}
    def __repr__(self) -> str:
        return f"{self.child}.t0"



