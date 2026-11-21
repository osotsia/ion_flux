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
    def __init__(self, cond: Expr, then_body: List[Stmt], else_body: Optional[List[Stmt]] = None):
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body or []

class Comment(Stmt):
    def __init__(self, text: str, is_section: bool = False):
        self.text = text
        self.is_section = is_section

class RawCpp(Expr):
    def __init__(self, code: str): 
        self.code = code


def compute_expr_children(expr: Expr) -> Tuple[Expr, ...]:
    """
    Canonical exhaustive child accessor for Compute IR Expr nodes.
    Raises TypeError on unknown Expr subclasses to prevent silent traversal omissions.
    """
    match expr:
        case Literal() | Var() | RawCpp():
            return ()
        case ArrayAccess(index=idx):
            return (idx,)
        case UnaryMinus(expr=child):
            return (child,)
        case BinaryOp(left=l, right=r):
            return (l, r)
        case FuncCall(args=args):
            return tuple(args)
        case Ternary(cond=c, true_val=tv, false_val=fv):
            return (c, tv, fv)
        case UnstructuredRead(state_offset=so, rp_offset=rp, ci_offset=ci, w_offset=wo, idx_expr=ie):
            return (so, rp, ci, wo, ie)
        case Reduction(loops=loops, child_expr=ce, vol_exprs=ves):
            return tuple(end_e for _, end_e in loops) + (ce,) + tuple(ves)
        case _:
            raise TypeError(f"Unhandled Compute IR Expr node: {type(expr).__name__}")


_IR_PREC = {
    "==": 1, "!=": 1, "<": 2, "<=": 2, ">": 2, ">=": 2,
    "+": 3, "-": 3,
    "*": 4, "/": 4,
}


def format_compute_ir(stmts: List[Stmt], indent: int = 0) -> str:
    """Renders Compute IR statements with symbolic MemoryRefs and precedence-minimal syntax."""
    pad = "  " * indent

    def fmt_expr(node: Expr, parent_prec: int = 0) -> str:
        match node:
            case Literal(val=v):
                if isinstance(v, float):
                    return f"{v:.6g}"
                return str(v)
            case Var(name=n):
                return n
            case ArrayAccess(array_name=arr, index=idx, symbol=sym, field=fld, base_offset=bo):
                if sym:
                    tag = f"{sym}.{fld}" if fld else sym
                    match idx:
                        case Literal(val=v) if int(float(v)) == bo:
                            return f"{arr}[{tag}]"
                        case BinaryOp(op="+", left=Literal(val=v), right=rest) if int(float(v)) == bo:
                            return f"{arr}[{tag} + {fmt_expr(rest, 3)}]"
                    return f"{arr}[{tag} @ {fmt_expr(idx, 0)}]"
                return f"{arr}[{fmt_expr(idx, 0)}]"
            case BinaryOp(op="+", left=l, right=UnaryMinus(expr=inner)):
                return fmt_expr(BinaryOp("-", l, inner), parent_prec)
            case BinaryOp(op="-", left=l, right=UnaryMinus(expr=inner)):
                return fmt_expr(BinaryOp("+", l, inner), parent_prec)
            case BinaryOp(op=op, left=l, right=r):
                prec = _IR_PREC.get(op, 0)
                l_str = fmt_expr(l, prec)
                r_prec = prec + (1 if op in ("-", "/", "==", "!=", "<", "<=", ">", ">=") else 0)
                r_str = fmt_expr(r, r_prec)
                out = f"{l_str} {op} {r_str}"
                return f"({out})" if prec < parent_prec else out
            case UnaryMinus(expr=UnaryMinus(expr=inner)):
                return fmt_expr(inner, parent_prec)
            case UnaryMinus(expr=e):
                out = f"-{fmt_expr(e, 5)}"
                return f"({out})" if parent_prec > 5 else out
            case FuncCall(func=f, args=args):
                return f"{f}({', '.join(fmt_expr(a, 0) for a in args)})"
            case Ternary(cond=c, true_val=tv, false_val=fv):
                out = f"{fmt_expr(c, 1)} ? {fmt_expr(tv, 1)} : {fmt_expr(fv, 1)}"
                return f"({out})" if parent_prec > 0 else out
            case Reduction(loops=loops, child_expr=ce):
                dom_s = " x ".join(f"{v}<{fmt_expr(end, 0)}" for v, end in loops)
                return f"SUM[{dom_s}]({fmt_expr(ce, 0)})"
            case UnstructuredRead(idx_expr=ie):
                return f"CSR_DIV(idx={fmt_expr(ie, 0)})"
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
            case VarDecl(var_type=vt, name=n, init=Ternary(cond=c, true_val=tv, false_val=fv)):
                lines.append(f"{pad}{vt} {n} = {fmt_expr(c, 1)}")
                lines.append(f"{pad}  ? {fmt_expr(tv, 1)}")
                lines.append(f"{pad}  : {fmt_expr(fv, 1)};")
            case VarDecl(var_type=vt, name=n, init=init):
                lines.append(f"{pad}{vt} {n} = {fmt_expr(init, 0)};")
            case Assign(lhs=lhs, rhs=rhs):
                lines.append(f"{pad}{fmt_expr(lhs, 0)} = {fmt_expr(rhs, 0)};")
            case Loop(var=v, start=s, end=e, body=body, pragma=p):
                if p:
                    lines.append(f"{pad}{p}")
                lines.append(f"{pad}for {v} in [{fmt_expr(s, 0)} .. {fmt_expr(e, 0)}):")
                lines.append(format_compute_ir(body, indent + 1))
            case IfElse(cond=c, then_body=tb, else_body=eb):
                lines.append(f"{pad}if {fmt_expr(c, 0)}:")
                lines.append(format_compute_ir(tb, indent + 1))
                if eb:
                    lines.append(f"{pad}else:")
                    lines.append(format_compute_ir(eb, indent + 1))
    return "\n".join(lines)