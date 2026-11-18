"""
Compute Intermediate Representation (IR).
Strictly defines the data structure of the mathematical operations, 
decoupled from physical topology or syntax formatting.
"""
from typing import List, Tuple, Union, Optional

class IRNode: pass
class Expr(IRNode): pass
class Stmt(IRNode): pass

class Literal(Expr):
    def __init__(self, val: Union[float, int, str]): 
        self.val = val

class Var(Expr):
    def __init__(self, name: str): 
        self.name = name

class ArrayAccess(Expr):
    def __init__(self, array_name: str, index: Expr, symbol: str = "", field: str = "", base_offset: int = 0):
        self.array_name = array_name
        self.index = index
        self.symbol = symbol
        self.field = field
        self.base_offset = base_offset

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
    """Represents a discrete integral accumulation loop."""
    def __init__(self, loops: List[Tuple[str, Expr]], child_expr: Expr, vol_exprs: List[Expr], pre_stmts: Optional[List[Stmt]] = None):
        self.loops = loops
        self.child_expr = child_expr
        self.vol_exprs = vol_exprs
        self.pre_stmts = pre_stmts or []

class VarDecl(Stmt):
    def __init__(self, var_type: str, name: str, init: Expr):
        self.var_type = var_type
        self.name = name
        self.init = init

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

class IfElse(Stmt):
    def __init__(self, cond: Expr, then_body: List[Stmt], else_body: List[Stmt]):
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body

class Comment(Stmt):
    def __init__(self, text: str, is_section: bool = False):
        self.text = text
        self.is_section = is_section

class RawCpp(Expr):
    def __init__(self, code: str): 
        self.code = code


def format_compute_ir(stmts: List[Stmt], indent: int = 0) -> str:
    """Renders Compute IR statements with symbolic MemoryRefs and indented loop blocks."""
    pad = "  " * indent

    def fmt_expr(node: Expr) -> str:
        match node:
            case Literal(val=v):
                return str(v)
            case Var(name=n):
                return n
            case ArrayAccess(array_name=arr, index=idx, symbol=sym, field=fld):
                if sym and fld:
                    return f"{arr}[{sym}.{fld} @ {fmt_expr(idx)}]"
                if sym:
                    return f"{arr}[{sym} @ {fmt_expr(idx)}]"
                return f"{arr}[{fmt_expr(idx)}]"
            case BinaryOp(op=op, left=l, right=r):
                return f"({fmt_expr(l)} {op} {fmt_expr(r)})"
            case UnaryMinus(expr=e):
                return f"-({fmt_expr(e)})"
            case FuncCall(func=f, args=args):
                return f"{f}({', '.join(fmt_expr(a) for a in args)})"
            case Ternary(cond=c, true_val=tv, false_val=fv):
                return f"({fmt_expr(c)} ? {fmt_expr(tv)} : {fmt_expr(fv)})"
            case Reduction(loops=loops, child_expr=ce):
                dom_s = " x ".join(f"{v}<{fmt_expr(end)}" for v, end in loops)
                return f"SUM[{dom_s}]({fmt_expr(ce)})"
            case UnstructuredRead(idx_expr=ie):
                return f"CSR_DIV(idx={fmt_expr(ie)})"
            case RawCpp(code=code):
                return code
            case _:
                return str(node)

    lines: List[str] = []
    for stmt in stmts:
        match stmt:
            case Comment(text=t, is_section=True):
                lines.append(f"{pad}// === {t} ===")
            case Comment(text=t, is_section=False):
                lines.append(f"{pad}// {t}")
            case VarDecl(var_type=vt, name=n, init=init):
                lines.append(f"{pad}{vt} {n} = {fmt_expr(init)};")
            case Assign(lhs=lhs, rhs=rhs):
                lines.append(f"{pad}{fmt_expr(lhs)} = {fmt_expr(rhs)};")
            case IfElse(cond=c, then_body=tb, else_body=eb):
                lines.append(f"{pad}if ({fmt_expr(c)}):")
                lines.append(format_compute_ir(tb, indent + 1))
                if eb:
                    lines.append(f"{pad}else:")
                    lines.append(format_compute_ir(eb, indent + 1))
            case Loop(var=v, start=s, end=e, body=body, pragma=p):
                if p:
                    lines.append(f"{pad}{p}")
                lines.append(f"{pad}for {v} in [{fmt_expr(s)} .. {fmt_expr(e)}):")
                lines.append(format_compute_ir(body, indent + 1))
    return "\n".join(lines)