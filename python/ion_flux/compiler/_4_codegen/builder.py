"""
Codegen Builder.

Coordinates FVM spatial discretization and mechanical C++ emission from Math IR.
Does not perform semantic normalization or AST manipulation.
"""

from typing import List, Dict, Any, Tuple, Optional
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._3_backend.math_ir import MathSystem
from ion_flux.compiler._3_backend.discretizer import FVMDiscretizer
from ion_flux.compiler._3_backend.compute_ir import (
    Stmt, Expr, Var, ArrayAccess, BinaryOp, UnaryMinus, FuncCall,
    Ternary, VarDecl, Assign, Loop, IfElse, UnstructuredRead, Reduction, format_compute_ir
)
from ion_flux.compiler._4_codegen.cpp_emitter import CppEmitter
from ion_flux.compiler._4_codegen.templates import generate_cpp_skeleton


def _build_offset_aliases(layout: Any) -> Tuple[Dict[Tuple[str, str, str], str], List[str]]:
    alias_map: Dict[Tuple[str, str, str], str] = {}
    decls: List[str] = []

    for s_name, (off, _) in getattr(layout, "state_offsets", {}).items():
        c_name = f"Y_{s_name}"
        alias_map[("y", s_name, "")] = c_name
        decls.append(f"constexpr int {c_name} = {off};")

    for p_name, (off, _) in getattr(layout, "param_offsets", {}).items():
        c_name = f"P_{p_name}"
        alias_map[("p", p_name, "")] = c_name
        decls.append(f"constexpr int {c_name} = {off};")

    for o_name, (off, _) in getattr(layout, "obs_offsets", {}).items():
        c_name = f"OBS_{o_name}"
        alias_map[("obs", o_name, "")] = c_name
        decls.append(f"constexpr int {c_name} = {off};")

    for d_name, fields in getattr(layout, "mesh_offsets", {}).items():
        for field_name, val in fields.items():
            if isinstance(val, int):
                c_name = f"M_{d_name}_{field_name}"
                alias_map[("m", d_name, field_name)] = c_name
                decls.append(f"constexpr int {c_name} = {val};")

    return alias_map, decls


def _collect_used_vars(stmts: List[Stmt]) -> set:
    used: set = set()

    def walk_expr(e: Expr) -> None:
        match e:
            case Var(name=n):
                used.add(n)
            case ArrayAccess(index=idx):
                walk_expr(idx)
            case BinaryOp(left=l, right=r):
                walk_expr(l); walk_expr(r)
            case UnaryMinus(expr=c):
                walk_expr(c)
            case FuncCall(args=args):
                for a in args: walk_expr(a)
            case Ternary(cond=c, true_val=t, false_val=f):
                walk_expr(c); walk_expr(t); walk_expr(f)
            case Reduction(loops=loops, child_expr=ce, vol_exprs=ves, pre_stmts=ps):
                for _, end_e in loops: walk_expr(end_e)
                walk_stmts(ps)
                walk_expr(ce)
                for ve in ves: walk_expr(ve)
            case UnstructuredRead(state_offset=so, rp_offset=rp, ci_offset=ci, w_offset=wo, idx_expr=ie):
                walk_expr(so); walk_expr(rp); walk_expr(ci); walk_expr(wo); walk_expr(ie)

    def walk_stmts(ss: List[Stmt]) -> None:
        for s in ss:
            match s:
                case VarDecl(init=init): walk_expr(init)
                case Assign(lhs=lhs, rhs=rhs): walk_expr(lhs); walk_expr(rhs)
                case Loop(start=st, end=en, body=b): walk_expr(st); walk_expr(en); walk_stmts(b)
                case IfElse(cond=c, then_body=tb, else_body=eb): walk_expr(c); walk_stmts(tb); walk_stmts(eb)

    walk_stmts(stmts)
    return used


def _filter_l_phys(l_phys_stmts: List[Stmt], stmts: List[Stmt]) -> List[Stmt]:
    used = _collect_used_vars(stmts)
    return [s for s in l_phys_stmts if not isinstance(s, VarDecl) or s.name in used]


def _emit_block(emitter: CppEmitter, preamble: List[str], stmts: List[Stmt]) -> str:
    if not stmts:
        return ""
    lines: List[str] = list(preamble)
    for stmt in stmts:
        lines.extend(emitter.emit(stmt).splitlines())
    return "\n".join(lines)


def generate_cpp_with_audit(
    math_sys: MathSystem,
    layout: Any,
    topo: TopologyAnalyzer,
    semantic_ctx: Optional[SemanticContext],
    state_map: Dict[str, Any],
    target: str = "cpu",
    audit_compiler: bool = False,
) -> Dict[str, Any]:
    """
    Lowers Math IR to Compute IR, stringifies statements to readable C++, and emits
    optional audit watchpoints into evaluate_audit_observables().
    """
    discretizer = FVMDiscretizer(layout, topo, semantic_ctx, state_map, target, math_sys=math_sys)
    l_phys_stmts, eq_stmts, obs_stmts = discretizer.discretize_system(math_sys, audit_compiler=audit_compiler)
    l_phys_ic_stmts, ic_stmts = discretizer.discretize_initial_conditions(math_sys)
    audit_obs_stmts, clamp_meta = discretizer.discretize_audit_observables()

    alias_map, preamble_decls = _build_offset_aliases(layout)
    emitter = CppEmitter(alias_map=alias_map)

    body_str = _emit_block(emitter, preamble_decls, _filter_l_phys(l_phys_stmts, eq_stmts) + eq_stmts)
    obs_body_str = _emit_block(emitter, preamble_decls, _filter_l_phys(l_phys_stmts, obs_stmts) + obs_stmts) if obs_stmts else ""
    ic_body_str = _emit_block(emitter, preamble_decls, _filter_l_phys(l_phys_ic_stmts, ic_stmts) + ic_stmts)
    audit_obs_body_str = _emit_block(emitter, preamble_decls, _filter_l_phys(l_phys_stmts, audit_obs_stmts) + audit_obs_stmts) if audit_obs_stmts else ""

    cpp_str = generate_cpp_skeleton(
        n_states=layout.n_states,
        n_params=layout.n_params,
        n_obs=layout.n_obs,
        body=body_str,
        obs_body=obs_body_str,
        ic_body=ic_body_str,
        audit_obs_body=audit_obs_body_str,
    )

    return {
        "cpp_source": cpp_str,
        "eq_stmts": eq_stmts,
        "compute_pretty": format_compute_ir(eq_stmts) if audit_compiler else "",
        "clamp_meta": clamp_meta,
    }


def generate_cpp(
    math_sys: MathSystem,
    layout: Any,
    topo: TopologyAnalyzer,
    semantic_ctx: Optional[SemanticContext],
    state_map: Dict[str, Any],
    target: str = "cpu"
) -> Tuple[str, List[Stmt]]:
    """
    Lowers Math IR to Compute IR, stringifies statements to C++, and wraps them in
    the native execution and Enzyme AD template.
    """
    bundle = generate_cpp_with_audit(
        math_sys=math_sys,
        layout=layout,
        topo=topo,
        semantic_ctx=semantic_ctx,
        state_map=state_map,
        target=target,
        audit_compiler=False,
    )
    return bundle["cpp_source"], bundle["eq_stmts"]