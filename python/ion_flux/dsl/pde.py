import copy
from typing import Dict, Any, List
from .nodes import Node, State, Parameter, Boundary, _wrap, SystemDict, Piecewise
from .spatial import Domain, CompositeDomain, ConcatenatedDomain

def merge(*systems: SystemDict) -> SystemDict:
    merged: SystemDict = {"equations": {}, "boundaries": {}, "initial_conditions": {}}
    for sys in systems:
        if not sys: continue
        merged["equations"].update(sys.get("equations", {}))
        merged["boundaries"].update(sys.get("boundaries", {}))
        merged["initial_conditions"].update(sys.get("initial_conditions", {}))
    return merged

class Condition:
    __slots__ = ["expression", "_compiled_logic"]
    def __init__(self, expression: Any):
        if not (hasattr(expression, "type") and getattr(expression, "type") == "BinaryOp" or type(expression).__name__ == "BinaryOp"):
            raise TypeError("Condition must be a valid AST BinaryOp node.")
        self.expression = expression
        op_map = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<", "eq": "==", "ne": "!="}
        left, right = expression.left_node, expression.right_node
        var_name = getattr(left, "name", str(left))
        
        if type(right).__name__ == "Scalar": target = right.value
        elif type(right).__name__ in ("Parameter", "State"): target = right.name
        else: target = float(str(right))
        self._compiled_logic = (var_name, op_map.get(expression.op), target)

    def evaluate(self, session: Any) -> bool:
        if not self._compiled_logic: return False
        import numpy as np
        var, op, val_target = self._compiled_logic
        try: current_val = session.get_array(var)
        except KeyError: return False
        
        val = val_target
        if isinstance(val_target, str):
            val = session.parameters.get(val_target)
            if val is None:
                try: val = session.get_array(val_target)
                except KeyError: return False

        if op == ">=": return bool(np.any(current_val >= val))
        if op == "<=": return bool(np.any(current_val <= val))
        if op == ">": return bool(np.any(current_val > val))
        if op == "<": return bool(np.any(current_val < val))
        if op == "==": return bool(np.any(current_val == val))
        if op == "!=": return bool(np.any(current_val != val))
        return False

class Terminal:
    __slots__ = ["current", "voltage", "name", "_original_name"]
    def __init__(self, current: "State", voltage: "State", name: str = ""):
        self.current = current
        self.voltage = voltage
        self.name = name
    def __set_name__(self, owner, name):
        if not self.name: self.name = name

class PDE:
    def __init__(self, **kwargs):
        self._bind_declarations()
        self._pde_init_done = True
        
    def _bind_declarations(self) -> None:
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
                if isinstance(clone, Terminal): has_terminal = True
            elif isinstance(clone, PDE):
                clone._apply_namespace(prefix=name)
                setattr(self, name, clone)
        
        if has_terminal and not hasattr(self, "_term_mode"):
            self._term_mode = Parameter(default=1.0, name="_term_mode")
            self._term_i_target = Parameter(default=0.0, name="_term_i_target")
            self._term_v_target = Parameter(default=0.0, name="_term_v_target")

    def _apply_namespace(self, prefix: str) -> None:
        for name, attr in self.__dict__.items():
            if isinstance(attr, (State, Parameter, Domain, CompositeDomain, ConcatenatedDomain, Terminal)):
                if not hasattr(attr, "_original_name"): attr._original_name = getattr(attr, "name", "") or name
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
        gathered, seen = [], set()
        for attr in self.__dict__.values():
            if isinstance(attr, node_type) and id(attr) not in seen:
                seen.add(id(attr)); gathered.append(attr)
            elif isinstance(attr, PDE):
                for sub_attr in attr.components(node_type):
                    if id(sub_attr) not in seen:
                        seen.add(id(sub_attr)); gathered.append(sub_attr)
        return gathered

    def math(self) -> SystemDict:
        raise NotImplementedError("PDE subclasses must implement math().")

    def ast(self) -> Dict[str, Any]:
        raw = self.math()
        compiled = {"equations": [], "boundaries": [], "initial_conditions": [], "domains": {}}
        
        for d in self.components(Domain):
            compiled["domains"][d.name] = {
                "bounds": d.bounds, 
                "resolution": d.resolution,
                "start_idx": getattr(d, "start_idx", 0),
                "coord_sys": getattr(d, "coord_sys", "cartesian")
            }
            
        for target, bcs in raw.get("boundaries", {}).items():
            if isinstance(target, State):
                compiled["boundaries"].append({
                    "type": "dirichlet", "state": target.name,
                    "bcs": {k: _wrap(v).to_dict() for k, v in bcs.items()}
                })
            elif isinstance(target, Domain):
                compiled["boundaries"].append({
                    "type": "moving_domain", "domain": target.name,
                    "bcs": {k: _wrap(v).to_dict() for k, v in bcs.items()}
                })
            elif isinstance(target, Boundary):
                target.child._bc_id = str(id(target.child))
                bc_entry = {
                    "type": "neumann", "node_id": target.child._bc_id,
                    "bcs": {target.side: _wrap(bcs).to_dict()}
                }
                if target.domain is not None:
                    bc_entry["domain"] = target.domain.name
                compiled["boundaries"].append(bc_entry)
            else:
                target._bc_id = str(id(target))
                compiled["boundaries"].append({
                    "type": "neumann", "node_id": target._bc_id,
                    "bcs": {k: _wrap(v).to_dict() for k, v in bcs.items()}
                })
                
        for state, eq in raw.get("equations", {}).items():
            if isinstance(eq, Piecewise):
                compiled["equations"].append({"state": state.name, "type": "piecewise", "regions": eq.to_dict()["regions"]})
                for reg in eq.region_map.keys():
                    compiled["domains"][reg.name] = {
                        "bounds": reg.bounds, 
                        "resolution": reg.resolution,
                        "start_idx": getattr(reg, "start_idx", 0),
                        "coord_sys": getattr(reg, "coord_sys", "cartesian")
                    }
            else:
                compiled["equations"].append({"state": state.name, "type": "standard", "eq": eq.to_dict()})
                
        for state, val in raw.get("initial_conditions", {}).items():
            compiled["initial_conditions"].append({"state": state.name, "value": _wrap(val).to_dict()})
            
        if hasattr(self, "terminal") and self.terminal:
            m = self._term_mode
            rhs = m * self._term_i_target + (1.0 - m) * (self.terminal.current - self.terminal.voltage + self._term_v_target)
            compiled["equations"].append({"state": self.terminal.current.name, "type": "standard", "eq": (self.terminal.current == rhs).to_dict()})
            
        return compiled