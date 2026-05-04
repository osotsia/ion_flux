from ion_flux.compiler.passes.ir import (
    IRNode, Literal, Var, ArrayAccess, BinaryOp, UnaryMinus, 
    FuncCall, Ternary, Assign, Loop, RawCpp
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
        if isinstance(node, RawCpp): 
            return node.code
            
        raise ValueError(f"Unknown IR Node: {type(node)}")