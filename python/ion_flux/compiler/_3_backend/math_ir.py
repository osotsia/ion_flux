"""
Math Intermediate Representation (Math IR).

Represents N-Dimensional continuum mathematics, tensor differential operators,
and domain boundary constraints prior to spatial discretization and memory layout flattening.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple


class MathNode:
    """Base class for all Math IR nodes."""
    pass


class MathExpr(MathNode):
    """Base class for mathematical expressions in Math IR."""
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathScalar(MathExpr):
    value: float
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathParameter(MathExpr):
    name: str
    offset: int
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathState(MathExpr):
    name: str
    domain_name: Optional[str]
    offset: int
    size: int
    is_ydot: bool = False
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathBinaryOp(MathExpr):
    op: str
    left: MathExpr
    right: MathExpr
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathUnaryOp(MathExpr):
    op: str
    child: MathExpr
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathClamp(MathExpr):
    child: MathExpr
    lower: MathExpr
    upper: MathExpr
    label: str = ""
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathGrad(MathExpr):
    child: MathExpr
    axis: Optional[str] = None
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathDiv(MathExpr):
    child: MathExpr
    axis: Optional[str] = None
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathDt(MathExpr):
    child: MathExpr
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathCoords(MathExpr):
    axis: Optional[str] = None
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathIntegral(MathExpr):
    child: MathExpr
    over_domain: str
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathBoundaryRef(MathExpr):
    child: MathExpr
    side: str
    domain: Optional[str] = None
    bc_id: Optional[str] = None


@dataclass(frozen=True)
class MathDirichletOverride(MathNode):
    state_name: str
    side: str
    axis: Optional[str]
    value_expr: MathExpr


@dataclass
class MathPiecewiseRegion:
    domain_name: str
    start_idx: int
    end_idx: int
    expr: MathExpr
    div_flux: Optional[MathExpr] = None


@dataclass
class MathEquation:
    state_name: str
    target_domain: Optional[str]
    lhs: MathExpr
    rhs: MathExpr
    bounds_override: Optional[Dict[str, tuple]] = None
    is_piecewise: bool = False
    regions: Optional[List[MathPiecewiseRegion]] = None
    current_region: Optional[MathPiecewiseRegion] = None


@dataclass
class MathObservable:
    name: str
    target_domain: Optional[str]
    expr: MathExpr
    bounds_override: Optional[Dict[str, tuple]] = None


@dataclass(frozen=True)
class MathInitialCondition:
    state_name: str
    target_domain: Optional[str]
    expr: MathExpr


@dataclass(frozen=True)
class MathNeumannBC:
    expr: MathExpr
    domain: Optional[str] = None


@dataclass
class MathSystem:
    equations: List[MathEquation]
    observables: List[MathObservable]
    dirichlet_overrides: List[MathDirichletOverride]
    dynamic_domain_bindings: Dict[str, Dict[str, Any]]
    initial_conditions: List[MathInitialCondition] = field(default_factory=list)
    neumann_bcs: Dict[str, Dict[str, MathNeumannBC]] = field(default_factory=dict)
    dirichlet_bcs: Dict[str, Dict[str, MathExpr]] = field(default_factory=dict)

    def pretty(self, state_filter: Optional[str] = None) -> str:
        """Formats MathSystem as readable infix continuum equations with CSE bindings and inline BCs."""
        return format_math_system(self, state_filter=state_filter)


def math_children(expr: MathExpr) -> Tuple[MathExpr, ...]:
    """
    Canonical exhaustive child accessor for MathExpr.
    Raises TypeError on unknown MathExpr subclasses to prevent silent node drops.
    """
    match expr:
        case MathScalar() | MathParameter() | MathState() | MathCoords():
            return ()
        case (
            MathUnaryOp(child=child)
            | MathGrad(child=child)
            | MathDiv(child=child)
            | MathDt(child=child)
            | MathIntegral(child=child)
            | MathBoundaryRef(child=child)
        ):
            return (child,)
        case MathBinaryOp(left=left, right=right):
            return (left, right)
        case MathClamp(child=child, lower=lower, upper=upper):
            return (child, lower, upper)
        case _:
            raise TypeError(f"Unhandled MathExpr node: {type(expr).__name__}")


def any_math(expr: MathExpr, predicate: Callable[[MathExpr], bool]) -> bool:
    """Returns True if predicate holds for expr or any descendant."""
    if predicate(expr):
        return True
    return any(any_math(child, predicate) for child in math_children(expr))


def extract_domain_name(expr: MathExpr) -> Optional[str]:
    """Recursively resolves the governing spatial domain identifier of an expression."""
    match expr:
        case MathState(domain_name=domain_name):
            return domain_name
        case MathBoundaryRef(domain=domain, child=child):
            return domain or extract_domain_name(child)
        case MathGrad(axis=axis, child=child) | MathDiv(axis=axis, child=child):
            return axis or extract_domain_name(child)
        case MathCoords(axis=axis):
            return axis
        case MathIntegral():
            return None
        case _:
            for child in math_children(expr):
                dom = extract_domain_name(child)
                if dom is not None:
                    return dom
            return None


def collect_bc_ids(expr: MathExpr, out: List[str]) -> None:
    """Collects unique boundary-condition tags across expr and all descendants."""
    bid = getattr(expr, "bc_id", None)
    if bid and bid not in out:
        out.append(bid)
    for child in math_children(expr):
        collect_bc_ids(child, out)


def extract_math_state_names(expr: MathExpr) -> List[str]:
    """Recursively extracts all unique State variable names referenced in a MathExpr."""
    names: List[str] = []

    def _walk(node: MathExpr) -> None:
        if isinstance(node, MathState):
            names.append(node.name)
            return
        for child in math_children(node):
            _walk(child)

    _walk(expr)
    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]


def has_time_derivative(expr: MathExpr) -> bool:
    """Returns True if a MathExpr contains any time derivative (MathDt or is_ydot MathState)."""
    return any_math(expr, lambda n: isinstance(n, MathDt) or (isinstance(n, MathState) and n.is_ydot))


_MATH_OP_PREC = {
    "lt": (1, "<"), "le": (1, "<="), "gt": (1, ">"), "ge": (1, ">="), "eq": (1, "=="), "ne": (1, "!="),
    "add": (2, "+"), "sub": (2, "-"),
    "mul": (3, "*"), "div": (3, "/"),
    "pow": (4, "^"),
}


def _math_expr_key(expr: MathExpr) -> tuple:
    match expr:
        case MathScalar(value=v):
            return ("S", v)
        case MathParameter(name=n):
            return ("P", n)
        case MathState(name=n, is_ydot=yd):
            return ("Y", n, yd)
        case MathCoords(axis=a):
            return ("X", a)
        case MathBinaryOp(op=op, left=l, right=r):
            return ("B", op, _math_expr_key(l), _math_expr_key(r))
        case MathUnaryOp(op=op, child=c):
            return ("U", op, _math_expr_key(c))
        case MathClamp(child=c, lower=lo, upper=hi, label=lbl):
            return ("C", lbl, _math_expr_key(c), _math_expr_key(lo), _math_expr_key(hi))
        case MathGrad(child=c, axis=a):
            return ("G", a, _math_expr_key(c))
        case MathDiv(child=c, axis=a):
            return ("D", a, _math_expr_key(c))
        case MathDt(child=c):
            return ("DT", _math_expr_key(c))
        case MathIntegral(child=c, over_domain=d):
            return ("I", d, _math_expr_key(c))
        case MathBoundaryRef(child=c, side=s, domain=d):
            return ("BR", s, d, _math_expr_key(c))
        case _:
            return ("?", str(expr))


def _math_node_size(expr: MathExpr) -> int:
    return 1 + sum(_math_node_size(c) for c in math_children(expr))


def _collect_math_cse(expr: MathExpr, counts: Dict[tuple, int], samples: Dict[tuple, MathExpr]) -> None:
    if _math_node_size(expr) >= 5:
        k = _math_expr_key(expr)
        counts[k] = counts.get(k, 0) + 1
        if k not in samples:
            samples[k] = expr
    for child in math_children(expr):
        _collect_math_cse(child, counts, samples)


def _format_math_expr(expr: MathExpr, parent_prec: int = 0, cse_aliases: Optional[Dict[tuple, str]] = None) -> str:
    if cse_aliases:
        k = _math_expr_key(expr)
        if k in cse_aliases:
            return cse_aliases[k]

    match expr:
        case MathScalar(value=v):
            return f"{v:g}" if abs(v) >= 1e-3 and abs(v) < 1e6 or v == 0.0 else f"{v:.4e}"
        case MathParameter(name=n):
            return n
        case MathState(name=n, is_ydot=True):
            return f"dt({n})"
        case MathState(name=n, is_ydot=False):
            return n
        case MathCoords(axis=a):
            return f"{a}.coords" if a else "coords"
        case MathUnaryOp(op="neg", child=c):
            s = f"-{_format_math_expr(c, 4, cse_aliases)}"
            return f"({s})" if parent_prec > 4 else s
        case MathUnaryOp(op=op, child=c):
            return f"{op}({_format_math_expr(c, 0, cse_aliases)})"
        case MathClamp(child=c, lower=lo, upper=hi, label=lbl):
            tag = f"[{lbl}]" if lbl else ""
            return f"clamp{tag}({_format_math_expr(c, 0, cse_aliases)}, {_format_math_expr(lo, 0, cse_aliases)}, {_format_math_expr(hi, 0, cse_aliases)})"
        case MathGrad(child=c, axis=a):
            ax = f", {a}" if a else ""
            return f"grad({_format_math_expr(c, 0, cse_aliases)}{ax})"
        case MathDiv(child=c, axis=a):
            ax = f", {a}" if a else ""
            return f"div({_format_math_expr(c, 0, cse_aliases)}{ax})"
        case MathDt(child=c):
            return f"dt({_format_math_expr(c, 0, cse_aliases)})"
        case MathIntegral(child=c, over_domain=d):
            return f"integral({_format_math_expr(c, 0, cse_aliases)}, over={d})"
        case MathBoundaryRef(child=c, side=s, domain=d):
            dom = f"({d})" if d else ""
            return f"{_format_math_expr(c, 5, cse_aliases)}.{s}{dom}"
        case MathBinaryOp(op=op, left=l, right=r):
            if op in ("max", "min"):
                return f"{op}({_format_math_expr(l, 0, cse_aliases)}, {_format_math_expr(r, 0, cse_aliases)})"
            prec, sym = _MATH_OP_PREC.get(op, (0, op))
            if op == "add" and isinstance(r, MathScalar) and r.value < 0:
                l_str = _format_math_expr(l, prec, cse_aliases)
                r_str = _format_math_expr(MathScalar(-r.value), prec + 1, cse_aliases)
                out = f"{l_str} - {r_str}"
                return f"({out})" if prec < parent_prec else out
            if op == "add" and isinstance(r, MathUnaryOp) and r.op == "neg":
                l_str = _format_math_expr(l, prec, cse_aliases)
                r_str = _format_math_expr(r.child, prec + 1, cse_aliases)
                out = f"{l_str} - {r_str}"
                return f"({out})" if prec < parent_prec else out
            if op == "sub" and isinstance(r, MathUnaryOp) and r.op == "neg":
                l_str = _format_math_expr(l, prec, cse_aliases)
                r_str = _format_math_expr(r.child, prec, cse_aliases)
                out = f"{l_str} + {r_str}"
                return f"({out})" if prec < parent_prec else out
            l_str = _format_math_expr(l, prec, cse_aliases)
            r_str = _format_math_expr(r, prec + (1 if op in ("sub", "div", "pow") else 0), cse_aliases)
            out = f"{l_str} {sym} {r_str}"
            return f"({out})" if prec < parent_prec else out
        case _:
            return str(expr)


def format_math_system(math_sys: "MathSystem", state_filter: Optional[str] = None) -> str:
    counts: Dict[tuple, int] = {}
    samples: Dict[tuple, MathExpr] = {}
    for eq in math_sys.equations:
        if state_filter and eq.state_name != state_filter:
            continue
        _collect_math_cse(eq.lhs, counts, samples)
        _collect_math_cse(eq.rhs, counts, samples)

    cse_keys = [k for k, cnt in counts.items() if cnt >= 2]
    cse_keys.sort(key=lambda k: _math_node_size(samples[k]))
    cse_aliases: Dict[tuple, str] = {}

    lines: List[str] = ["=== MathSystem ==="]
    if cse_keys:
        lines.append("Shared Subexpressions (CSE):")
        for idx, k in enumerate(cse_keys):
            expr_str = _format_math_expr(samples[k], 0, cse_aliases)
            alias = f"%t{idx}"
            lines.append(f"  let {alias} = {expr_str}")
            cse_aliases[k] = alias

    by_state: Dict[str, List[MathEquation]] = {}
    for eq in math_sys.equations:
        if state_filter and eq.state_name != state_filter:
            continue
        by_state.setdefault(eq.state_name, []).append(eq)

    lines.append("Equations:")
    for s_name, eqs in by_state.items():
        dom = eqs[0].target_domain or "0D"
        is_pde = any(has_time_derivative(e.lhs) or has_time_derivative(e.rhs) for e in eqs)
        kind = "PDE/ODE" if is_pde else "DAE"
        lines.append(f"  [{kind}] {s_name} (domain={dom}):")
        eq_bc_ids: List[str] = []
        for eq in eqs:
            collect_bc_ids(eq.lhs, eq_bc_ids)
            collect_bc_ids(eq.rhs, eq_bc_ids)
            lhs_s = _format_math_expr(eq.lhs, 0, cse_aliases)
            rhs_s = _format_math_expr(eq.rhs, 0, cse_aliases)
            if eq.is_piecewise and eq.current_region:
                reg = eq.current_region
                lines.append(f"    region {reg.domain_name}[{reg.start_idx}:{reg.end_idx}]: {lhs_s} == {rhs_s}")
            else:
                lines.append(f"    {lhs_s} == {rhs_s}")
        if s_name in math_sys.dirichlet_bcs:
            bc_parts = [f"{side}={_format_math_expr(val, 0, cse_aliases)}" for side, val in math_sys.dirichlet_bcs[s_name].items()]
            lines.append(f"    Dirichlet BCs: {', '.join(bc_parts)}")
        neumann_parts: List[str] = []
        for bid in eq_bc_ids:
            for side, nbc in math_sys.neumann_bcs.get(bid, {}).items():
                dom_tag = f"({nbc.domain})" if nbc.domain else ""
                neumann_parts.append(f"{side}{dom_tag}={_format_math_expr(nbc.expr, 0, cse_aliases)}")
        if neumann_parts:
            lines.append(f"    Neumann BCs:   {', '.join(neumann_parts)}")

    if math_sys.dynamic_domain_bindings and not state_filter:
        lines.append("ALE Moving Boundaries:")
        for d_name, binding in math_sys.dynamic_domain_bindings.items():
            lines.append(f"  {d_name}.{binding['side']} == {_format_math_expr(binding['rhs_expr'], 0, cse_aliases)}")

    ics = [ic for ic in math_sys.initial_conditions if not state_filter or ic.state_name == state_filter]
    if ics:
        lines.append("Initial Conditions:")
        for ic in ics:
            dom = ic.target_domain or "0D"
            lines.append(f"  {ic.state_name}({dom}) = {_format_math_expr(ic.expr, 0, cse_aliases)}")

    obs_list = [o for o in math_sys.observables if not state_filter or o.name == state_filter]
    if obs_list:
        lines.append("Observables:")
        for o in obs_list:
            dom = o.target_domain or "0D"
            lines.append(f"  {o.name}({dom}) = {_format_math_expr(o.expr, 0, cse_aliases)}")

    return "\n".join(lines)