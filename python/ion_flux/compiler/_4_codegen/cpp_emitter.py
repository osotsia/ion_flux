"""
C++ Emitter.

Mechanically stringifies the Compute IR into exact C++ syntax.
Uses Python 3.10+ pattern matching to decode Compute IR nodes.
Performs no mathematical or topological transformations.
"""

from typing import Dict, Tuple, Optional
from ion_flux.compiler._3_backend.compute_ir import (
    IRNode, Literal, Var, ArrayAccess, BinaryOp, UnaryMinus,
    FuncCall, Ternary, VarDecl, Assign, IfElse, Loop, Comment, RawCpp, UnstructuredRead, Reduction
)


_CPP_PREC = {
    "==": 1, "!=": 1, "<": 2, "<=": 2, ">": 2, ">=": 2,
    "+": 3, "-": 3,
    "*": 4, "/": 4,
}


class CppEmitter:
    """Stringifies Compute IR nodes into deterministic, precedence-minimal C++ code."""

    def __init__(self, alias_map: Optional[Dict[Tuple[str, str, str], str]] = None):
        self.alias_map = alias_map or {}

    def _alias_for(self, node: ArrayAccess) -> Optional[str]:
        if not node.symbol:
            return None
        space = "y" if node.array_name in ("y", "ydot", "res", "y0") else node.array_name
        return self.alias_map.get((space, node.symbol, node.field))

    def emit(self, node: IRNode, parent_prec: int = 0) -> str:
        match node:
            case Literal(val=val):
                return str(val)

            case Var(name=name):
                return name

            case ArrayAccess(array_name=arr, index=idx, base_offset=base_off):
                alias = self._alias_for(node)
                if alias:
                    match idx:
                        case Literal(val=v) if int(float(v)) == base_off:
                            return f"{arr}[{alias}]"
                        case BinaryOp(op="+", left=Literal(val=v), right=rest) if int(float(v)) == base_off:
                            return f"{arr}[{alias} + {self.emit(rest, 3)}]"
                return f"{arr}[{self.emit(idx)}]"

            case BinaryOp(op="+", left=left, right=UnaryMinus(expr=inner)):
                return self.emit(BinaryOp("-", left, inner), parent_prec)

            case BinaryOp(op="-", left=left, right=UnaryMinus(expr=inner)):
                return self.emit(BinaryOp("+", left, inner), parent_prec)

            case BinaryOp(op=op, left=left, right=right):
                prec = _CPP_PREC.get(op, 0)
                l_str = self.emit(left, prec)
                r_prec = prec + (1 if op in ("-", "/", "==", "!=", "<", "<=", ">", ">=") else 0)
                r_str = self.emit(right, r_prec)
                out = f"{l_str} {op} {r_str}"
                return f"({out})" if prec < parent_prec else out

            case UnaryMinus(expr=UnaryMinus(expr=inner)):
                return self.emit(inner, parent_prec)

            case UnaryMinus(expr=expr):
                out = f"-{self.emit(expr, 5)}"
                return f"({out})" if parent_prec > 5 else out

            case FuncCall(func=func, args=args):
                return f"{func}({', '.join(self.emit(a, 0) for a in args)})"

            case Ternary(cond=cond, true_val=t, false_val=f):
                out = f"{self.emit(cond, 0)} ? {self.emit(t, 0)} : {self.emit(f, 0)}"
                return f"({out})" if parent_prec > 0 else out

            case Comment(text=text, is_section=True):
                return f"\n// === {text} ==="

            case Comment(text=text, is_section=False):
                return f"// {text}"

            case VarDecl(var_type=var_type, name=name, init=Reduction(loops=loops, child_expr=child_expr, vol_exprs=vol_exprs, pre_stmts=pre_stmts)):
                lines = [f"{var_type} {name} = 0.0;"]
                ind = ""
                for v_name, end_expr in loops:
                    lines.append(f"{ind}#pragma clang loop unroll(full)")
                    lines.append(f"{ind}for (int {v_name} = 0; {v_name} < {self.emit(end_expr)}; ++{v_name}) {{")
                    ind += "    "
                for ps in pre_stmts:
                    for p_line in self.emit(ps).splitlines():
                        lines.append(f"{ind}{p_line}")
                lines.append(f"{ind}double _vol = 1.0;")
                for ve in vol_exprs:
                    lines.append(f"{ind}_vol *= {self.emit(ve, 4)};")
                lines.append(f"{ind}{name} += ({self.emit(child_expr, 0)}) * _vol;")
                for _ in loops:
                    ind = ind[:-4]
                    lines.append(f"{ind}}}")
                return "\n".join(lines)

            case VarDecl(var_type=var_type, name=name, init=UnstructuredRead(state_offset=s_off, rp_offset=rp, ci_offset=ci, w_offset=w, idx_expr=idx)):
                s_e = self.emit(s_off)
                i_e = self.emit(idx)
                return (
                    f"double {name} = 0.0;\n"
                    f"for (int _k = (int)m[{self.emit(rp)} + {i_e}]; "
                    f"_k < (int)m[{self.emit(rp)} + {i_e} + 1]; ++_k) {{\n"
                    f"    {name} += m[{self.emit(w)} + _k] * (y[{s_e} + (int)m[{self.emit(ci)} + _k]] - y[{s_e} + {i_e}]);\n"
                    f"}}"
                )

            case VarDecl(var_type=var_type, name=name, init=init):
                return f"{var_type} {name} = {self.emit(init)};"

            case Assign(lhs=lhs, rhs=rhs):
                return f"{self.emit(lhs)} = {self.emit(rhs)};"

            case IfElse(cond=cond, then_body=tb, else_body=eb):
                tb_lines = []
                for s in tb:
                    tb_lines.extend(self.emit(s).splitlines())
                tb_str = "\n    ".join(tb_lines)
                out = f"if ({self.emit(cond, 0)}) {{\n    {tb_str}\n}}"
                if eb:
                    eb_lines = []
                    for s in eb:
                        eb_lines.extend(self.emit(s).splitlines())
                    eb_str = "\n    ".join(eb_lines)
                    out += f" else {{\n    {eb_str}\n}}"
                return out

            case Loop(var=var, start=start, end=end, body=body, pragma=pragma):
                lines = []
                for b in body:
                    lines.extend(self.emit(b).splitlines())
                body_str = "\n    ".join(lines)
                pragma_str = f"{pragma}\n" if pragma else ""
                return (
                    f"{pragma_str}for (int {var} = {self.emit(start)}; "
                    f"{var} < {self.emit(end)}; ++{var}) {{\n    {body_str}\n}}"
                )

            case UnstructuredRead(state_offset=s_off, rp_offset=rp, ci_offset=ci, w_offset=w, idx_expr=idx):
                return (
                    f"[&]() {{\n    double sum = 0.0;\n"
                    f"    for(int k = (int)m[{self.emit(rp)} + {self.emit(idx)}]; "
                    f"k < (int)m[{self.emit(rp)} + {self.emit(idx)} + 1]; ++k) {{\n"
                    f"        sum += m[{self.emit(w)} + k] * (y[{self.emit(s_off)} + "
                    f"(int)m[{self.emit(ci)} + k]] - y[{self.emit(s_off)} + {self.emit(idx)}]);\n"
                    f"    }}\n    return sum;\n}}()"
                )

            case Reduction(loops=loops, child_expr=child_expr, vol_exprs=vol_exprs):
                cpp_code = "[&]() {\n    double sum = 0.0;\n"
                for var_name, end_expr in loops:
                    res_str = self.emit(end_expr)
                    cpp_code += (
                        f"    #pragma clang loop unroll(full)\n"
                        f"    for(int {var_name} = 0; {var_name} < {res_str}; ++{var_name}) {{\n"
                    )

                cpp_code += "        double vol = 1.0;\n"
                for vol_expr in vol_exprs:
                    cpp_code += f"        vol *= {self.emit(vol_expr)};\n"

                child_cpp = self.emit(child_expr)
                cpp_code += f"        sum += {child_cpp} * vol;\n"

                for _ in loops:
                    cpp_code += "    }\n"
                cpp_code += "    return sum;\n}()"
                return cpp_code

            case RawCpp(code=code):
                return code

            case _:
                raise ValueError(f"Unknown IR Node: {type(node)}")