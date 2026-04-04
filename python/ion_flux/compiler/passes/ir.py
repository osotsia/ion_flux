"""
Intermediate Representation (IR) for Spatial Lowering.
Acts as the bridge between Abstract Math and C++ Emission.
"""
from typing import List, Optional

class IRNode:
    """Base class for all Intermediate Representation nodes."""
    def to_cpp(self) -> str:
        raise NotImplementedError("IRNodes must implement C++ emission.")

class Expr(IRNode): pass
class Stmt(IRNode): pass

# --- 1. Memory Access & Variables ---

class Literal(Expr):
    def __init__(self, val: float | int | str): self.val = val
    def to_cpp(self): return str(self.val)

class Var(Expr):
    def __init__(self, name: str): self.name = name
    def to_cpp(self): return self.name

class ArrayAccess(Expr):
    """Represents a flat C-array memory lookup: array[index]"""
    def __init__(self, array_name: str, index: Expr):
        self.array_name = array_name
        self.index = index
    def to_cpp(self): return f"{self.array_name}[{self.index.to_cpp()}]"

# --- 2. Mathematical Operations ---

class BinaryOp(Expr):
    def __init__(self, op: str, left: Expr, right: Expr): 
        self.op, self.left, self.right = op, left, right
    def to_cpp(self): return f"({self.left.to_cpp()} {self.op} {self.right.to_cpp()})"

class FuncCall(Expr):
    def __init__(self, func: str, args: List[Expr]):
        self.func, self.args = func, args
    def to_cpp(self): return f"{self.func}({', '.join(a.to_cpp() for a in self.args)})"

class Ternary(Expr):
    """Inline conditional (cond ? true_val : false_val). Critical for boundary masking."""
    def __init__(self, cond: Expr, true_val: Expr, false_val: Expr):
        self.cond, self.true_val, self.false_val = cond, true_val, false_val
    def to_cpp(self): return f"({self.cond.to_cpp()} ? {self.true_val.to_cpp()} : {self.false_val.to_cpp()})"

# --- 3. Control Flow ---

class Assign(Stmt):
    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs, self.rhs = lhs, rhs
    def to_cpp(self): return f"{self.lhs.to_cpp()} = {self.rhs.to_cpp()};"

class Loop(Stmt):
    """Represents a spatial iteration over a Finite Volume mesh."""
    def __init__(self, var: str, start: Expr, end: Expr, body: List[Stmt], pragma: str = ""):
        self.var, self.start, self.end, self.body, self.pragma = var, start, end, body, pragma
        
    def to_cpp(self):
        body_str = '\n    '.join(b.to_cpp() for b in self.body)
        pragma_str = f"{self.pragma}\n" if self.pragma else ""
        return f"{pragma_str}for (int {self.var} = {self.start.to_cpp()}; {self.var} < {self.end.to_cpp()}; ++{self.var}) {{\n    {body_str}\n}}"

class RawCpp(Expr):
    """The Escape Hatch. Used for highly complex macros or isolated block limits."""
    def __init__(self, code: str): self.code = code
    def to_cpp(self): return self.code