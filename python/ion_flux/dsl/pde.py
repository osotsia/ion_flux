import copy
from typing import Dict, Any, List
from .nodes import Node, State, Parameter, _wrap, SystemDict
from .spatial import Domain, CompositeDomain, ConcatenatedDomain

def merge(*systems: SystemDict) -> SystemDict:
    """Merges multiple SystemDict physics payloads into a single cohesive system."""
    merged: SystemDict = {"regions": {}, "global": [], "boundaries": []}
    for sys in systems:
        if not sys:
            continue
        for dom, eqs in sys.get("regions", {}).items():
            merged["regions"].setdefault(dom, []).extend(eqs)
        merged["global"].extend(sys.get("global", []))
        merged["boundaries"].extend(sys.get("boundaries", []))
    return merged



class Condition:
    """Compiled Boolean trigger for event detection and protocol hot-swapping via pure AST."""
    __slots__ = ["expression", "_compiled_logic"]

    def __init__(self, expression: Any):
        # Validate that the condition is a strict AST Node, eliminating brittle regex string parsing
        if not (hasattr(expression, "type") and getattr(expression, "type") == "BinaryOp" or type(expression).__name__ == "BinaryOp"):
            raise TypeError("Condition must be instantiated with a valid AST BinaryOp node (e.g., model.V_cell <= 3.2)")

        self.expression = expression
        
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

    def evaluate(self, session: Any) -> bool:
        """Evaluates the boolean condition against the active memory session."""
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
    __slots__ = ["current", "voltage", "name", "_original_name"]
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
        self._pde_init_done = True
        
    def _bind_declarations(self) -> None:
        """
        Isolates class-level Nodes to the instance level to prevent state leakage
        across concurrent PDE instantiations, injecting variable names intrinsically.
        """
        to_copy = {}
        for name in dir(self.__class__):
            if name.startswith("__"): continue
            
            attr = getattr(self.__class__, name)
            if isinstance(attr, (State, Parameter, Domain, CompositeDomain, ConcatenatedDomain, Terminal, PDE)):
                to_copy[name] = attr

        clones = copy.deepcopy(to_copy)
        
        has_terminal = False
        for name, clone in clones.items():
            if isinstance(clone, (State, Parameter, Domain, CompositeDomain, ConcatenatedDomain, Terminal)):
                clone.name = name
                setattr(self, name, clone)
                if isinstance(clone, Terminal):
                    has_terminal = True
            elif isinstance(clone, PDE):
                clone._apply_namespace(prefix=name)
                setattr(self, name, clone)
        
        # Implicitly inject reserved parameters for hardware multiplexing 
        if has_terminal and not hasattr(self, "_term_mode"):
            self._term_mode = Parameter(default=1.0, name="_term_mode")
            self._term_i_target = Parameter(default=0.0, name="_term_i_target")
            self._term_v_target = Parameter(default=0.0, name="_term_v_target")

    def _apply_namespace(self, prefix: str) -> None:
        """Recursively prefixes all internal AST nodes to prevent compilation collisions."""
        for name, attr in self.__dict__.items():
            if isinstance(attr, (State, Parameter, Domain, CompositeDomain, ConcatenatedDomain, Terminal)):
                if not hasattr(attr, "_original_name"):
                    attr._original_name = getattr(attr, "name", "") or name
                attr.name = f"{prefix}_{attr._original_name}"
            elif isinstance(attr, PDE):
                attr._apply_namespace(prefix=f"{prefix}_{name}")

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_pde_init_done", False):
            if isinstance(value, PDE):
                clone = copy.deepcopy(value)
                clone._apply_namespace(prefix=name)
                super().__setattr__(name, clone)
                return
        super().__setattr__(name, value)

    def components(self, node_type: type) -> List[Any]:
        gathered = []
        seen = set()
        for attr in self.__dict__.values():
            if isinstance(attr, node_type):
                if id(attr) not in seen:
                    seen.add(id(attr))
                    gathered.append(attr)
            elif isinstance(attr, PDE):
                for sub_attr in attr.components(node_type):
                    if id(sub_attr) not in seen:
                        seen.add(id(sub_attr))
                        gathered.append(sub_attr)
        return gathered

    def math(self) -> SystemDict:
        raise NotImplementedError("PDE subclasses must implement the math() method returning a SystemDict.")

    def ast(self) -> Dict[str, Any]:
        raw_math = self.math()
        if not isinstance(raw_math, dict):
            raise TypeError("math() must return a dictionary using the SystemDict structure (keys: regions, global, boundaries).")

        valid_keys = {"regions", "global", "boundaries"}
        invalid_keys = set(raw_math.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(f"Invalid keys found: {invalid_keys}.")
        
        compiled_system: Dict[str, Any] = {
            "regions": {}, "global": [], "boundaries": []
        }
        
        if "regions" in raw_math:
            for dom, eqs in raw_math["regions"].items():
                dom_name = getattr(dom, "name", str(dom))
                compiled_system["regions"][dom_name] = []
                for eq in eqs:
                    if getattr(eq, "op", "") != "eq":
                        raise ValueError(f"Equations in regions must use '=='. Got: {eq}")
                    compiled_system["regions"][dom_name].append({
                        "lhs": eq.left_node.to_dict(), 
                        "rhs": _wrap(eq.right_node).to_dict()
                    })
                    
        for bucket in ["global", "boundaries"]:
            if bucket in raw_math:
                for eq in raw_math[bucket]:
                    if getattr(eq, "op", "") != "eq":
                        raise ValueError(f"Equations in '{bucket}' must use '=='. Got: {eq}")
                    compiled_system[bucket].append({
                        "lhs": eq.left_node.to_dict(), 
                        "rhs": _wrap(eq.right_node).to_dict()
                    })

        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, Terminal):
                m = self._term_mode
                rhs = m * self._term_i_target + (1.0 - m) * (attr.current - attr.voltage + self._term_v_target)
                compiled_system["global"].append({"lhs": attr.current.to_dict(), "rhs": rhs.to_dict()})
                
        return compiled_system
