"""
Sparsity Tracer.

Evaluates Compute IR loop statements symbolically in Python to trace exact
Jacobian (row, col) sparsity triplets prior to C++ compilation.
Uses structural pattern matching over Compute IR nodes.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
from ion_flux.compiler._4_codegen.compute_ir import (
    Loop, Assign, ArrayAccess, BinaryOp, Ternary, FuncCall,
    Literal, Var, UnaryMinus, UnstructuredRead, Reduction
)


class IndexEvaluator:
    """
    Executes the Loop-Level Math Intermediate Representation (MIR) dynamically
    in Python to extract Jacobian (Row, Column) sparsity triplets.
    """
    def __init__(self, layout):
        self.layout = layout
        self.mesh_cache = layout.mesh_cache
        self.sparse_triplets: Set[Tuple[int, int]] = set()

    def evaluate(self, stmts: List[Any], env: Optional[Dict[str, int]] = None):
        if env is None:
            env = {}

        for stmt in stmts:
            match stmt:
                case Loop(var=var, start=start, end=end, body=body):
                    s_val = self.eval_idx(start, env)
                    e_val = self.eval_idx(end, env)
                    for val in range(s_val, e_val):
                        env[var] = val
                        self.evaluate(body, env)
                    env.pop(var, None)

                case Assign(lhs=ArrayAccess(array_name="res", index=idx), rhs=rhs):
                    row = self.eval_idx(idx, env)
                    cols = self.extract_cols(rhs, env)
                    self.sparse_triplets.add((row, row))  # Guarantee strict diagonal elements
                    for col in cols:
                        self.sparse_triplets.add((row, col))

                case _:
                    pass

    def eval_idx(self, expr: Any, env: Dict[str, int]) -> int:
        match expr:
            case Literal(val=val):
                return int(float(val))

            case Var(name=name):
                return env.get(name, 0)

            case BinaryOp(op="+", left=l, right=r):
                return self.eval_idx(l, env) + self.eval_idx(r, env)

            case BinaryOp(op="-", left=l, right=r):
                return self.eval_idx(l, env) - self.eval_idx(r, env)

            case BinaryOp(op="*", left=l, right=r):
                return self.eval_idx(l, env) * self.eval_idx(r, env)

            case BinaryOp(op="/", left=l, right=r):
                denom = self.eval_idx(r, env)
                return self.eval_idx(l, env) // denom if denom != 0 else 0

            case FuncCall(func="CLAMP", args=[arg0, arg1]):
                val = self.eval_idx(arg0, env)
                bound = self.eval_idx(arg1, env)
                return max(0, min(val, bound - 1))

            case _:
                return 0

    def eval_cond(self, expr: Any, env: Dict[str, int]) -> Optional[bool]:
        match expr:
            case BinaryOp(op=op, left=l, right=r):
                lv = self.eval_idx(l, env)
                rv = self.eval_idx(r, env)
                match op:
                    case "==": return lv == rv
                    case "!=": return lv != rv
                    case ">": return lv > rv
                    case "<": return lv < rv
                    case ">=": return lv >= rv
                    case "<=": return lv <= rv
                    case _: return None

            case Literal(val=val):
                return bool(float(val))

            case _:
                return None

    def extract_cols(self, expr: Any, env: Dict[str, int]) -> List[int]:
        cols: List[int] = []

        match expr:
            case ArrayAccess(array_name="y" | "ydot", index=idx):
                cols.append(self.eval_idx(idx, env))

            case ArrayAccess(index=idx):
                cols.extend(self.extract_cols(idx, env))

            case BinaryOp(left=l, right=r):
                cols.extend(self.extract_cols(l, env))
                cols.extend(self.extract_cols(r, env))

            case UnaryMinus(expr=inner):
                cols.extend(self.extract_cols(inner, env))

            case Ternary(cond=cond, true_val=tv, false_val=fv):
                match self.eval_cond(cond, env):
                    case True:
                        cols.extend(self.extract_cols(tv, env))
                    case False:
                        cols.extend(self.extract_cols(fv, env))
                    case None:
                        cols.extend(self.extract_cols(tv, env))
                        cols.extend(self.extract_cols(fv, env))

            case FuncCall(args=args):
                for arg in args:
                    cols.extend(self.extract_cols(arg, env))

            case UnstructuredRead(state_offset=so, rp_offset=rpo, ci_offset=cio, idx_expr=ie):
                idx = self.eval_idx(ie, env)
                rp_off = self.eval_idx(rpo, env)
                ci_off = self.eval_idx(cio, env)
                s_off = self.eval_idx(so, env)

                rp_start = int(self.mesh_cache.get(rp_off + idx, 0))
                rp_end = int(self.mesh_cache.get(rp_off + idx + 1, 0))

                cols.append(s_off + idx)
                for k in range(rp_start, rp_end):
                    neighbor = int(self.mesh_cache.get(ci_off + k, 0))
                    cols.append(s_off + neighbor)

            case Reduction(loops=loops, child_expr=child_expr):
                def eval_loops(depth, current_env):
                    if depth == len(loops):
                        cols.extend(self.extract_cols(child_expr, current_env))
                        return
                    var, end_expr = loops[depth]
                    end = self.eval_idx(end_expr, current_env)
                    for val in range(end):
                        current_env[var] = val
                        eval_loops(depth + 1, current_env)
                    current_env.pop(var, None)

                eval_loops(0, env.copy())

            case _:
                pass

        return cols


class SparsityAnalyzer:
    """Backwards compatibility wrapper for extracting CPR elements."""
    def __init__(self, eq_stmts: List[Any], layout: Any):
        evaluator = IndexEvaluator(layout)
        evaluator.evaluate(eq_stmts)
        self.sparse_triplets = evaluator.sparse_triplets