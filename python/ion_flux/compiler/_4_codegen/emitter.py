from ion_flux.compiler._2_lowering.ir import (
    IRNode, Literal, Var, ArrayAccess, BinaryOp, UnaryMinus, 
    FuncCall, Ternary, Assign, Loop, RawCpp, UnstructuredRead, Reduction
)

class CppEmitter:
    """
    Mechanically stringifies the Loop-Level Math Intermediate Representation (MIR) 
    into exact C++ syntax. Performs no mathematical or topological logic.
    """
    def emit(self, node: IRNode) -> str:
        if isinstance(node, Literal): 
            return str(node.val)
        if isinstance(node, Var): 
            return node.name
        if isinstance(node, ArrayAccess): 
            return f"{node.array_name}[{self.emit(node.index)}]"
        if isinstance(node, BinaryOp): 
            return f"({self.emit(node.left)} {node.op} {self.emit(node.right)})"
        if isinstance(node, UnaryMinus): 
            return f"(-{self.emit(node.expr)})"
        if isinstance(node, FuncCall): 
            return f"{node.func}({', '.join(self.emit(a) for a in node.args)})"
        if isinstance(node, Ternary): 
            return f"({self.emit(node.cond)} ? {self.emit(node.true_val)} : {self.emit(node.false_val)})"
        if isinstance(node, Assign): 
            return f"{self.emit(node.lhs)} = {self.emit(node.rhs)};"
        if isinstance(node, Loop):
            body_str = '\n    '.join(self.emit(b) for b in node.body)
            pragma_str = f"{node.pragma}\n" if node.pragma else ""
            return f"{pragma_str}for (int {node.var} = {self.emit(node.start)}; {node.var} < {self.emit(node.end)}; ++{node.var}) {{\n    {body_str}\n}}"
        if isinstance(node, UnstructuredRead):
            rp = self.emit(node.rp_offset)
            ci = self.emit(node.ci_offset)
            w = self.emit(node.w_offset)
            s_off = self.emit(node.state_offset)
            idx = self.emit(node.idx_expr)
            return (
                f"[&]() {{\n    double sum = 0.0;\n"
                f"    for(int k = (int)m[{rp} + {idx}]; k < (int)m[{rp} + {idx} + 1]; ++k) {{\n"
                f"        sum += m[{w} + k] * (y[{s_off} + (int)m[{ci} + k]] - y[{s_off} + {idx}]);\n"
                f"    }}\n    return sum;\n}}()"
            )
        if isinstance(node, Reduction):
            return node.cpp_code
        if isinstance(node, RawCpp): 
            return node.code
            
        raise ValueError(f"Unknown IR Node: {type(node)}")