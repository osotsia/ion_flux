"""
C++ Emitter.

Mechanically stringifies the Compute IR into exact C++ syntax.
Uses Python 3.10+ pattern matching to decode Compute IR nodes.
Performs no mathematical or topological transformations.
"""

from ion_flux.compiler._4_codegen.compute_ir import (
    IRNode, Literal, Var, ArrayAccess, BinaryOp, UnaryMinus,
    FuncCall, Ternary, Assign, Loop, RawCpp, UnstructuredRead, Reduction
)


class CppEmitter:
    """Stringifies Compute IR nodes into deterministic C++ code."""

    def emit(self, node: IRNode) -> str:
        match node:
            case Literal(val=val):
                return str(val)

            case Var(name=name):
                return name

            case ArrayAccess(array_name=arr, index=idx):
                return f"{arr}[{self.emit(idx)}]"

            case BinaryOp(op=op, left=left, right=right):
                return f"({self.emit(left)} {op} {self.emit(right)})"

            case UnaryMinus(expr=expr):
                return f"(-{self.emit(expr)})"

            case FuncCall(func=func, args=args):
                return f"{func}({', '.join(self.emit(a) for a in args)})"

            case Ternary(cond=cond, true_val=t, false_val=f):
                return f"({self.emit(cond)} ? {self.emit(t)} : {self.emit(f)})"

            case Assign(lhs=lhs, rhs=rhs):
                return f"{self.emit(lhs)} = {self.emit(rhs)};"

            case Loop(var=var, start=start, end=end, body=body, pragma=pragma):
                body_str = '\n    '.join(self.emit(b) for b in body)
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