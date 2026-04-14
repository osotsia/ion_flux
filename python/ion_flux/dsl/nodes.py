import re
from typing import Dict, Any, Union, List, Optional, TypedDict

SystemDict = TypedDict('SystemDict', {
    'equations': Dict[Any, Any],
    'boundaries': Dict[Any, Any],
    'initial_conditions': Dict[Any, Any],
    'observables': Dict[Any, Any]
}, total=False)

def validate_identifier(name: str) -> str:
    """Sanitizes AST names to prevent C++ RCE injection payloads."""
    if not name:
        return name
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(f"Invalid identifier '{name}'. To prevent code injection, names must be valid C/C++ identifiers.")
    return name

def Dirichlet(val: Union[float, 'Node']) -> 'Node':
    """Syntactic sugar marking explicit state override bounds."""
    return _wrap(val)

class Node:
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
    
    def __lt__(self, other): return BinaryOp("lt", self, _wrap(other))
    def __le__(self, other): return BinaryOp("le", self, _wrap(other))
    def __gt__(self, other): return BinaryOp("gt", self, _wrap(other))
    def __ge__(self, other): return BinaryOp("ge", self, _wrap(other))
    def __eq__(self, other): return BinaryOp("eq", self, _wrap(other))
    def __ne__(self, other): return BinaryOp("ne", self, _wrap(other))

    def __hash__(self) -> int: return id(self)

    @property
    def left(self) -> "Boundary": return Boundary(self, "left")
    @property
    def right(self) -> "Boundary": return Boundary(self, "right")
    @property
    def t0(self) -> "InitialCondition": return InitialCondition(self)

    def boundary(self, tag: str, domain: Optional["Domain"] = None) -> "Boundary": 
        return Boundary(self, tag, domain=domain)
        
    def surface(self, domain: Optional["Domain"] = None, side: str = "right") -> "Boundary":
        return Boundary(self, side, domain=domain)

    def to_dict(self) -> Dict[str, Any]: 
        raise NotImplementedError("Subclasses must implement to_dict()")

def _wrap(val: Union[int, float, Node]) -> Node:
    if isinstance(val, (int, float)): return Scalar(float(val))
    if isinstance(val, Node): return val
    raise TypeError(f"Cannot wrap type {type(val).__name__} into an AST Node.")

class Scalar(Node):
    __slots__ = ["value"]
    def __init__(self, value: float): self.value = value
    def to_dict(self) -> Dict[str, Any]: return {"type": "Scalar", "value": self.value}
    def __repr__(self) -> str: return str(self.value)

class State(Node):
    __slots__ = ["domain", "name", "max_newton_step", "_original_name"]
    def __init__(self, domain=None, name: str = "", max_newton_step: Optional[float] = None):
        self.domain = domain
        self.name = validate_identifier(name)
        self.max_newton_step = max_newton_step
    def to_dict(self) -> Dict[str, Any]: 
        d = {"type": "State", "name": self.name}
        if self.max_newton_step is not None: d["max_newton_step"] = self.max_newton_step
        return d
    def __repr__(self) -> str: return self.name or "<Unbound State>"
    def __set_name__(self, owner, name):
        if not self.name: self.name = validate_identifier(name)

class Observable(Node):
    __slots__ = ["domain", "name", "_original_name"]
    def __init__(self, domain=None, name: str = ""):
        self.domain = domain
        self.name = validate_identifier(name)
    def to_dict(self) -> Dict[str, Any]: 
        return {"type": "Observable", "name": self.name}
    def __repr__(self) -> str: return self.name or "<Unbound Observable>"
    def __set_name__(self, owner, name):
        if not self.name: self.name = validate_identifier(name)

class Parameter(Node):
    __slots__ = ["default", "name", "_original_name"]
    def __init__(self, default: float, name: str = ""):
        self.default = default
        self.name = validate_identifier(name)
    def to_dict(self) -> Dict[str, Any]: return {"type": "Parameter", "name": self.name, "default": self.default}
    def __repr__(self) -> str: return self.name or f"<Unbound Param: {self.default}>"
    def __set_name__(self, owner, name):
        if not self.name: self.name = validate_identifier(name)

class BinaryOp(Node):
    __slots__ = ["op", "left_node", "right_node", "_bc_id"]
    _SYM_MAP = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
    
    def __init__(self, op: str, left_node: Node, right_node: Node):
        self.op = op
        self.left_node = left_node
        self.right_node = right_node
        self._bc_id = None
        
    def to_dict(self) -> Dict[str, Any]: 
        d = {"type": "BinaryOp", "op": self.op, "left": self.left_node.to_dict(), "right": self.right_node.to_dict()}
        if getattr(self, "_bc_id", None): d["_bc_id"] = self._bc_id
        return d
    def __repr__(self) -> str: return f"({self.left_node} {self._SYM_MAP.get(self.op, self.op)} {self.right_node})"

class UnaryOp(Node):
    __slots__ = ["op", "child", "kwargs", "_bc_id"]
    def __init__(self, op: str, child: Node, **kwargs):
        self.op = op
        self.child = child
        self.kwargs = kwargs
        self._bc_id = None
        
    def to_dict(self) -> Dict[str, Any]: 
        d = {"type": "UnaryOp", "op": self.op, "child": self.child.to_dict()}
        for k, v in self.kwargs.items():
            if v is not None: d[k] = v.name if hasattr(v, "name") else str(v)
        if getattr(self, "_bc_id", None): d["_bc_id"] = self._bc_id
        return d
    def __repr__(self) -> str: return f"{self.op}({self.child})"

class Piecewise(Node):
    """Maps distinct regional equations spanning a single state's parent domain."""
    __slots__ = ["region_map"]
    def __init__(self, region_map: Dict[Any, BinaryOp]):
        self.region_map = region_map
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Piecewise",
            "regions": [
                {"domain": d.name, "start_idx": getattr(d, "start_idx", 0), 
                 "end_idx": getattr(d, "start_idx", 0) + d.resolution, "eq": eq.to_dict()}
                for d, eq in self.region_map.items()
            ]
        }

class Boundary(Node):
    __slots__ = ["child", "side", "domain"]
    def __init__(self, child: Node, side: str, domain=None):
        self.child = child
        self.side = side
        self.domain = domain
    def __call__(self, domain=None) -> "Boundary": return Boundary(self.child, self.side, domain=(domain or self.domain))
    def to_dict(self) -> Dict[str, Any]: 
        d = {"type": "Boundary", "side": self.side, "child": self.child.to_dict()}
        if self.domain: d["domain"] = self.domain.name
        return d
    def __repr__(self) -> str: return f"{self.child}.{self.side}"

class InitialCondition(Node):
    __slots__ = ["child"]
    def __init__(self, child: Node): self.child = child
    def __call__(self) -> "InitialCondition": return self
    def to_dict(self) -> Dict[str, Any]: return {"type": "InitialCondition", "child": self.child.to_dict()}
    def __repr__(self) -> str: return f"{self.child}.t0"