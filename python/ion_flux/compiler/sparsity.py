from typing import Dict, Any, List, Set, Tuple, Optional
from ion_flux.compiler.passes.ir import Loop, Assign, ArrayAccess, BinaryOp, Ternary, FuncCall, Literal, Var, UnaryMinus, UnstructuredRead, Reduction

class IndexEvaluator:
    """
    Executes the Loop-Level Math Intermediate Representation (MIR) dynamically
    in Python to perfectly extract the Jacobian (Row, Column) sparsity triplets.
    Guarantees single-source-of-truth convergence with the emitted C++ binary.
    """
    def __init__(self, layout):
        self.layout = layout
        self.mesh_cache = layout.mesh_cache
        self.sparse_triplets: Set[Tuple[int, int]] = set()

    def evaluate(self, stmts: List[Any], env: Optional[Dict[str, int]] = None):
        if env is None: env = {}
        for stmt in stmts:
            if isinstance(stmt, Loop):
                start = self.eval_idx(stmt.start, env)
                end = self.eval_idx(stmt.end, env)
                for val in range(start, end):
                    env[stmt.var] = val
                    self.evaluate(stmt.body, env)
                if stmt.var in env:
                    del env[stmt.var]
            elif isinstance(stmt, Assign):
                if isinstance(stmt.lhs, ArrayAccess) and stmt.lhs.array_name == "res":
                    row = self.eval_idx(stmt.lhs.index, env)
                    cols = self.extract_cols(stmt.rhs, env)
                    self.sparse_triplets.add((row, row)) # Guarantee strict diagonal elements
                    for col in cols:
                        self.sparse_triplets.add((row, col))

    def eval_idx(self, expr: Any, env: Dict[str, int]) -> int:
        if isinstance(expr, Literal): return int(float(expr.val))
        if isinstance(expr, Var): return env.get(expr.name, 0)
        if isinstance(expr, BinaryOp):
            l = self.eval_idx(expr.left, env)
            r = self.eval_idx(expr.right, env)
            if expr.op == "+": return l + r
            if expr.op == "-": return l - r
            if expr.op == "*": return l * r
            if expr.op == "/": return l // r if r != 0 else 0
        if isinstance(expr, FuncCall) and expr.func == "CLAMP":
            val = self.eval_idx(expr.args[0], env)
            bound = self.eval_idx(expr.args[1], env)
            return max(0, min(val, bound - 1))
        return 0

    def eval_cond(self, expr: Any, env: Dict[str, int]) -> Optional[bool]:
        if isinstance(expr, BinaryOp):
            l = self.eval_idx(expr.left, env)
            r = self.eval_idx(expr.right, env)
            if expr.op == "==": return l == r
            if expr.op == "!=": return l != r
            if expr.op == ">": return l > r
            if expr.op == "<": return l < r
            if expr.op == ">=": return l >= r
            if expr.op == "<=": return l <= r
        if isinstance(expr, Literal):
            return bool(float(expr.val))
        return None

    def extract_cols(self, expr: Any, env: Dict[str, int]) -> List[int]:
        cols = []
        if isinstance(expr, ArrayAccess):
            if expr.array_name == "y" or expr.array_name == "ydot":
                cols.append(self.eval_idx(expr.index, env))
            else:
                cols.extend(self.extract_cols(expr.index, env))
        elif isinstance(expr, BinaryOp):
            cols.extend(self.extract_cols(expr.left, env))
            cols.extend(self.extract_cols(expr.right, env))
        elif isinstance(expr, UnaryMinus):
            cols.extend(self.extract_cols(expr.expr, env))
        elif isinstance(expr, Ternary):
            cond_val = self.eval_cond(expr.cond, env)
            if cond_val is True:
                cols.extend(self.extract_cols(expr.true_val, env))
            elif cond_val is False:
                cols.extend(self.extract_cols(expr.false_val, env))
            else:
                cols.extend(self.extract_cols(expr.true_val, env))
                cols.extend(self.extract_cols(expr.false_val, env))
        elif isinstance(expr, FuncCall):
            for arg in expr.args:
                cols.extend(self.extract_cols(arg, env))
        elif isinstance(expr, UnstructuredRead):
            idx = self.eval_idx(expr.idx_expr, env)
            rp_off = self.eval_idx(expr.rp_offset, env)
            ci_off = self.eval_idx(expr.ci_offset, env)
            s_off = self.eval_idx(expr.state_offset, env)
            
            rp_start = int(self.mesh_cache.get(rp_off + idx, 0))
            rp_end = int(self.mesh_cache.get(rp_off + idx + 1, 0))
            
            cols.append(s_off + idx)
            for k in range(rp_start, rp_end):
                neighbor = int(self.mesh_cache.get(ci_off + k, 0))
                cols.append(s_off + neighbor)
        elif isinstance(expr, Reduction):
            def eval_loops(depth, current_env):
                if depth == len(expr.loop_vars):
                    cols.extend(self.extract_cols(expr.child_expr, current_env))
                    return
                var = expr.loop_vars[depth]
                end = self.eval_idx(expr.loop_ends[depth], current_env)
                for val in range(end):
                    current_env[var] = val
                    eval_loops(depth + 1, current_env)
                if var in current_env: del current_env[var]

            eval_loops(0, env.copy())
            
        return cols

class SparsityAnalyzer:
    """Backwards compatibility wrapper for extracting CPR elements."""
    def __init__(self, eq_stmts: List[Any], layout: Any):
        evaluator = IndexEvaluator(layout)
        evaluator.evaluate(eq_stmts)
        self.sparse_triplets = evaluator.sparse_triplets