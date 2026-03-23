from typing import Dict, Any, Union, List, Optional
from typing import TypedDict
import copy
import re

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
    __slots__ = ["domain", "name"]
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
    __slots__ = ["default", "name"]
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
    def left(self, domain: Optional["Domain"] = None) -> "Boundary": return Boundary(self, "left", domain=domain)
    def right(self, domain: Optional["Domain"] = None) -> "Boundary": return Boundary(self, "right", domain=domain)
    def boundary(self, tag: str, domain: Optional["Domain"] = None) -> "Boundary": return Boundary(self, tag, domain=domain)
    def __repr__(self) -> str: return f"{self.op}({self.child})"


class Boundary(Node):
    __slots__ = ["child", "side", "domain"]
    def __init__(self, child: Node, side: str, domain: Optional["Domain"] = None):
        self.child = child
        self.side = side
        self.domain = domain
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
    def to_dict(self) -> Dict[str, Any]: 
        return {"type": "InitialCondition", "child": self.child.to_dict()}
    def __repr__(self) -> str:
        return f"{self.child}.t0"


class DomainBoundary(Node):
    __slots__ = ["domain", "side"]
    def __init__(self, domain: "Domain", side: str):
        self.domain = domain
        self.side = side
    def to_dict(self) -> Dict[str, Any]: 
        return {"type": "DomainBoundary", "domain": self.domain.name, "side": self.side}
    def __repr__(self) -> str:
        return f"{self.domain.name}.{self.side}"


class ConcatenatedDomain:
    """Syntactic sugar allowing users to map physical cell regions logically."""
    __slots__ = ["domains", "name"]
    def __init__(self, domains: List['Domain'], name: str = ""):
        self.domains = domains
        self.name = name or "_plus_".join(d.name for d in domains)
        
    def __add__(self, other: Union['Domain', 'ConcatenatedDomain']) -> 'ConcatenatedDomain':
        if isinstance(other, Domain): return ConcatenatedDomain(self.domains + [other])
        elif isinstance(other, ConcatenatedDomain): return ConcatenatedDomain(self.domains + other.domains)
        raise TypeError("Can only concatenate Domains.")


class Domain:
    __slots__ = ["bounds", "resolution", "coord_sys", "name", "csr_data"]
    def __init__(self, bounds: tuple, resolution: int, coord_sys: str = "cartesian", name: str = "", csr_data: Optional[Dict] = None):
        self.bounds = bounds
        self.resolution = resolution
        self.coord_sys = coord_sys
        self.name = name
        self.csr_data = csr_data

    @classmethod
    def from_mesh(cls, mesh_data: Union[str, dict], name: str = "unstructured_mesh", surfaces: Optional[Dict[str, List[int]]] = None) -> "Domain":
        import numpy as np
        if isinstance(mesh_data, str):
            import json
            with open(mesh_data, "r") as f: mesh_data = json.load(f)
                
        nodes = np.array(mesh_data["nodes"], dtype=float)
        elements = np.array(mesh_data["elements"], dtype=int)
        resolution = len(nodes)
        
        from collections import defaultdict
        K_global = defaultdict(float)
        V_nodes = np.zeros(resolution, dtype=float)
        
        for el in elements:
            if len(el) != 4: continue
            
            p = nodes[el]
            J_mat = np.array([p[1]-p[0], p[2]-p[0], p[3]-p[0]]).T
            detJ = np.linalg.det(J_mat)
            vol = abs(detJ) / 6.0
            if vol < 1e-15: continue
            
            invJ = np.linalg.inv(J_mat)
            gradN = np.zeros((4, 3))
            gradN[1:4, :] = invJ.T
            gradN[0, :] = -np.sum(invJ.T, axis=0)
            
            K_local = vol * (gradN @ gradN.T)
            
            for i in range(4):
                V_nodes[el[i]] += vol / 4.0
                for j in range(4):
                    if i != j:
                        K_global[(el[i], el[j])] -= K_local[i, j]
                        
        row_ptr = [0] * (resolution + 1)
        col_ind, weights = [], []
        
        for i in range(resolution):
            row_ptr[i] = len(col_ind)
            vol_i = max(V_nodes[i], 1e-15)
            for j in range(resolution):
                if (i, j) in K_global:
                    col_ind.append(j)
                    weights.append(K_global[(i, j)] / vol_i)
        row_ptr[resolution] = len(col_ind)
        
        surface_masks = {}
        if surfaces:
            for tag, indices in surfaces.items():
                mask = [0.0] * resolution
                for idx in indices: mask[idx] = 1.0
                surface_masks[tag] = mask
                
        # Pack connectivity map into flat floats to bypass rigid Rust C-ABI homogenization
        csr_data = {
            "row_ptr": [float(x) for x in row_ptr],
            "col_ind": [float(x) for x in col_ind],
            "weights": weights,
            "surface_masks": surface_masks
        }
        return cls(bounds=(0, 1), resolution=resolution, coord_sys="unstructured", name=name, csr_data=csr_data)
        
    @property
    def coords(self) -> Node:
        return UnaryOp("coords", Scalar(0.0))

    @property
    def left(self) -> DomainBoundary:
        return DomainBoundary(self, "left")

    @property
    def right(self) -> DomainBoundary:
        return DomainBoundary(self, "right")
        
    def __mul__(self, other: "Domain") -> "CompositeDomain":
        return CompositeDomain([self, other])
        
    def __add__(self, other: Union["Domain", "ConcatenatedDomain"]) -> "ConcatenatedDomain":
        if isinstance(other, Domain): return ConcatenatedDomain([self, other])
        elif isinstance(other, ConcatenatedDomain): return ConcatenatedDomain([self] + other.domains)
        raise TypeError("Can only concatenate Domains.")
        
    def __repr__(self) -> str:
        return self.name or f"Domain({self.bounds})"

    def __set_name__(self, owner, name):
        if not self.name:
            self.name = name


class CompositeDomain:
    """Represents a hierarchical cross-product of multiple topologies."""
    __slots__ = ["domains", "name"]
    def __init__(self, domains: List[Domain], name: str = ""):
        self.domains = domains
        self.name = name or "_x_".join([d.name for d in domains if d.name])
        
    @property
    def resolution(self) -> int:
        import math
        return math.prod(d.resolution for d in self.domains)
        
    def __repr__(self) -> str:
        return self.name or f"CompositeDomain({[d.name for d in self.domains]})"

    def __set_name__(self, owner, name):
        if not self.name:
            self.name = name


class Condition:
    """Compiled Boolean trigger for event detection and protocol hot-swapping."""
    __slots__ = ["expression", "_compiled_logic"]
    def __init__(self, expression: Union[str, Node]):
        self.expression = expression
        
        if hasattr(expression, "type") and getattr(expression, "type") == "BinaryOp" or type(expression).__name__ == "BinaryOp":
            op_map = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<", "eq": "==", "ne": "!="}
            mapped_op = op_map.get(expression.op)
            left = expression.left_node
            right = expression.right_node
            
            var_name = getattr(left, "name", str(left))
        
            if type(right).__name__ == "Scalar":
                target = right.value
            elif type(right).__name__ in ("Parameter", "State"):
                target = right.name
            else:
                target = float(str(right))
            
            self._compiled_logic = (var_name, mapped_op, target)
            return

        match = re.search(r"([A-Za-z0-9_]+)\s*(>=|<=|>|<|==|!=)\s*([A-Za-z0-9_.-]+)", str(expression))
        if match:
            target_str = match.group(3)
            try: 
                target = float(target_str)
            except ValueError: 
                target = target_str
            self._compiled_logic = (match.group(1), match.group(2), target)
        else:
            self._compiled_logic = None

    def evaluate(self, session: Any) -> bool:
        if not self._compiled_logic: return False
        import numpy as np
        var, op, val_target = self._compiled_logic
        
        try: 
            current_val = session.get_array(var)
        except KeyError: 
            return False
        
        if isinstance(val_target, str):
            val = session.parameters.get(val_target, None)
            if val is None:
                try: 
                    val = session.get_array(val_target)
                except KeyError: 
                    return False
        else:
            val = val_target

        # Evaluate safely across spatial arrays to prevent local failures being averaged out
        if op == ">=": return bool(np.any(current_val >= val))
        if op == "<=": return bool(np.any(current_val <= val))
        if op == ">": return bool(np.any(current_val > val))
        if op == "<": return bool(np.any(current_val < val))
        if op == "==": return bool(np.any(current_val == val))
        if op == "!=": return bool(np.any(current_val != val))
        return False

    def __repr__(self) -> str:
        return f"Condition({self.expression})"

class Terminal:
    """Hardware abstraction representing an electrical battery cycler connection."""
    __slots__ = ["current", "voltage", "name"]
    def __init__(self, current: "State", voltage: "State", name: str = ""):
        self.current = current
        self.voltage = voltage
        self.name = name

    def __set_name__(self, owner, name):
        if not self.name:
            self.name = name

class PDE:
    """Declarative base class for defining Partial Differential Equations."""
    def __init__(self, **kwargs):
        self._bind_declarations()
        
    def _bind_declarations(self) -> None:
        """
        Isolates class-level Nodes to the instance level to prevent state leakage
        across concurrent PDE instantiations, injecting variable names intrinsically.
        """
        has_terminal = False
        for name in dir(self.__class__):
            if name.startswith("__"):
                continue
            
            attr = getattr(self.__class__, name)
            if isinstance(attr, (State, Parameter, Domain, CompositeDomain, Terminal)):
                clone = copy.copy(attr)
                clone.name = name
                setattr(self, name, clone)
                if isinstance(attr, Terminal):
                    has_terminal = True
        
        # Implicitly inject reserved parameters for hardware multiplexing 
        if has_terminal and not hasattr(self, "_term_mode"):
            self._term_mode = Parameter(default=1.0, name="_term_mode")
            self._term_i_target = Parameter(default=0.0, name="_term_i_target")
            self._term_v_target = Parameter(default=0.0, name="_term_v_target")

    def math(self) -> SystemDict:
        raise NotImplementedError("PDE subclasses must implement the math() method returning a SystemDict.")

    def ast(self) -> Dict[str, Any]:
        raw_math = self.math()
        if not isinstance(raw_math, dict):
            raise TypeError("math() must return a dictionary using the SystemDict structure (keys: regions, global, boundaries).")

        valid_keys = {"regions", "global", "boundaries"}
        invalid_keys = set(raw_math.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid keys found: {invalid_keys}.\n"
                f"Please group your equations into 'regions', 'global', or 'boundaries' and use the '==' operator."
            )
        
        # Initialize the strict payload structure expected by the compiler
        compiled_system: Dict[str, Any] = {
            "regions": {}, "global": [], "boundaries": []
        }
        
        # Parse regional equations bound to physical topologies
        if "regions" in raw_math:
            for dom, eqs in raw_math["regions"].items():
                dom_name = getattr(dom, "name", str(dom))
                compiled_system["regions"][dom_name] = []
                for eq in eqs:
                    if getattr(eq, "op", "") != "eq":
                        raise ValueError(f"Equations in regions must be declared using '=='. Got: {eq}")
                    compiled_system["regions"][dom_name].append({
                        "lhs": eq.left_node.to_dict(), 
                        "rhs": _wrap(eq.right_node).to_dict()
                    })
                    
        # Parse global and boundary equations
        for bucket in ["global", "boundaries"]:
            if bucket in raw_math:
                for eq in raw_math[bucket]:
                    if getattr(eq, "op", "") != "eq":
                        raise ValueError(f"Equations in '{bucket}' must be declared using '=='. Got: {eq}")
                    compiled_system[bucket].append({
                        "lhs": eq.left_node.to_dict(), 
                        "rhs": _wrap(eq.right_node).to_dict()
                    })

        # Automagically inject the compiled constraint multiplexer safely into the global bucket
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, Terminal):
                # Mode 1 (CC): i_app = i_target
                # Mode 0 (CV): V_cell = v_target
                m = self._term_mode
                rhs = m * self._term_i_target + (1.0 - m) * (attr.current - attr.voltage + self._term_v_target)
                compiled_system["global"].append({"lhs": attr.current.to_dict(), "rhs": rhs.to_dict()})
                
        return compiled_system