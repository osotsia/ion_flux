"""
Intermediate Representation (IR) for Spatial Lowering.
Strictly defines the data structure of the mathematical operations.
"""
from typing import List

class IRNode: pass
class Expr(IRNode): pass
class Stmt(IRNode): pass

class Literal(Expr):
    def __init__(self, val: float | int | str): 
        self.val = val

class Var(Expr):
    def __init__(self, name: str): 
        self.name = name

class ArrayAccess(Expr):
    def __init__(self, array_name: str, index: Expr):
        self.array_name = array_name
        self.index = index

class BinaryOp(Expr):
    def __init__(self, op: str, left: Expr, right: Expr): 
        self.op = op
        self.left = left
        self.right = right

class UnaryMinus(Expr):
    def __init__(self, expr: Expr):
        self.expr = expr

class FuncCall(Expr):
    def __init__(self, func: str, args: List[Expr]):
        self.func = func
        self.args = args

class Ternary(Expr):
    def __init__(self, cond: Expr, true_val: Expr, false_val: Expr):
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val

class UnstructuredRead(Expr):
    def __init__(self, state_offset: Expr, rp_offset: Expr, ci_offset: Expr, w_offset: Expr, idx_expr: Expr):
        self.state_offset = state_offset
        self.rp_offset = rp_offset
        self.ci_offset = ci_offset
        self.w_offset = w_offset
        self.idx_expr = idx_expr

class Reduction(Expr):
    def __init__(self, loop_vars: List[str], loop_ends: List[Expr], child_expr: Expr, cpp_code: str):
        self.loop_vars = loop_vars
        self.loop_ends = loop_ends
        self.child_expr = child_expr
        self.cpp_code = cpp_code

class Assign(Stmt):
    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs = lhs
        self.rhs = rhs

class Loop(Stmt):
    def __init__(self, var: str, start: Expr, end: Expr, body: List[Stmt], pragma: str = ""):
        self.var = var
        self.start = start
        self.end = end
        self.body = body
        self.pragma = pragma

class RawCpp(Expr):
    def __init__(self, code: str): 
        self.code = code