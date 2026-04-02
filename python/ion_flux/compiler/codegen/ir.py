from typing import List

class IRNode:
    def to_cpp(self) -> str: raise NotImplementedError()

class Expr(IRNode): pass
class Stmt(IRNode): pass

class Literal(Expr):
    def __init__(self, val): self.val = val
    def to_cpp(self): return str(self.val)

class Var(Expr):
    def __init__(self, name: str): self.name = name
    def to_cpp(self): return self.name

class BinaryOp(Expr):
    def __init__(self, op: str, left: Expr, right: Expr): 
        self.op, self.left, self.right = op, left, right
    def to_cpp(self): return f"({self.left.to_cpp()} {self.op} {self.right.to_cpp()})"

class UnaryOp(Expr):
    def __init__(self, op: str, child: Expr):
        self.op, self.child = op, child
    def to_cpp(self): return f"{self.op}({self.child.to_cpp()})"

class Ternary(Expr):
    def __init__(self, cond: Expr, true_val: Expr, false_val: Expr):
        self.cond, self.true_val, self.false_val = cond, true_val, false_val
    def to_cpp(self): return f"({self.cond.to_cpp()} ? {self.true_val.to_cpp()} : {self.false_val.to_cpp()})"

class FuncCall(Expr):
    def __init__(self, func: str, args: List[Expr]):
        self.func, self.args = func, args
    def to_cpp(self): return f"{self.func}({', '.join(a.to_cpp() for a in self.args)})"

class ArrayAccess(Expr):
    def __init__(self, array: str, index: Expr):
        self.array, self.index = array, index
    def to_cpp(self): return f"{self.array}[{self.index.to_cpp()}]"

class RawCpp(Expr):
    def __init__(self, code: str): self.code = code
    def to_cpp(self): return self.code

class Assign(Stmt):
    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs, self.rhs = lhs, rhs
    def to_cpp(self): return f"{self.lhs.to_cpp()} = {self.rhs.to_cpp()};"

class Loop(Stmt):
    def __init__(self, var: str, start: Expr, end: Expr, body: List[Stmt], pragma: str = ""):
        self.var, self.start, self.end, self.body, self.pragma = var, start, end, body, pragma
    def to_cpp(self):
        body_str = '\n    '.join(b.to_cpp() for b in self.body)
        return f"{self.pragma}\nfor (int {self.var} = {self.start.to_cpp()}; {self.var} < {self.end.to_cpp()}; ++{self.var}) {{\n    {body_str}\n}}"