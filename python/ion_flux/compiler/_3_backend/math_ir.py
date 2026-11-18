"""
Math Intermediate Representation (Math IR).

Represents N-Dimensional continuum mathematics, tensor differential operators,
and domain boundary constraints prior to spatial discretization and memory layout flattening.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


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


def extract_math_state_names(expr: MathExpr) -> List[str]:
    """
    Recursively extracts all unique State variable names referenced in a MathExpr.
    Uses class pattern matching on typed Math IR expressions.
    """
    names: List[str] = []

    match expr:
        case MathState(name=name):
            return [name]

        case (
            MathUnaryOp(child=child)
            | MathGrad(child=child)
            | MathDiv(child=child)
            | MathDt(child=child)
            | MathIntegral(child=child)
            | MathBoundaryRef(child=child)
        ):
            names.extend(extract_math_state_names(child))

        case MathBinaryOp(left=left, right=right):
            names.extend(extract_math_state_names(left))
            names.extend(extract_math_state_names(right))

        case MathClamp(child=child, lower=lower, upper=upper):
            names.extend(extract_math_state_names(child))
            names.extend(extract_math_state_names(lower))
            names.extend(extract_math_state_names(upper))

        case _:
            pass

    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]


def has_time_derivative(expr: MathExpr) -> bool:
    """Returns True if a MathExpr contains any time derivative (MathDt or is_ydot MathState)."""
    match expr:
        case MathState(is_ydot=True) | MathDt():
            return True

        case (
            MathUnaryOp(child=child)
            | MathGrad(child=child)
            | MathDiv(child=child)
            | MathIntegral(child=child)
            | MathBoundaryRef(child=child)
        ):
            return has_time_derivative(child)

        case MathBinaryOp(left=left, right=right):
            return has_time_derivative(left) or has_time_derivative(right)

        case MathClamp(child=child, lower=lower, upper=upper):
            return (
                has_time_derivative(child)
                or has_time_derivative(lower)
                or has_time_derivative(upper)
            )

        case _:
            return False


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
    match expr:
        case MathScalar() | MathParameter() | MathState() | MathCoords():
            return 1
        case MathUnaryOp(child=c) | MathGrad(child=c) | MathDiv(child=c) | MathDt(child=c) | MathIntegral(child=c) | MathBoundaryRef(child=c):
            return 1 + _math_node_size(c)
        case MathBinaryOp(left=l, right=r):
            return 1 + _math_node_size(l) + _math_node_size(r)
        case MathClamp(child=c, lower=lo, upper=hi):
            return 1 + _math_node_size(c) + _math_node_size(lo) + _math_node_size(hi)
        case _:
            return 1


def _collect_math_cse(expr: MathExpr, counts: Dict[tuple, int], samples: Dict[tuple, MathExpr]) -> None:
    if _math_node_size(expr) >= 5:
        k = _math_expr_key(expr)
        counts[k] = counts.get(k, 0) + 1
        if k not in samples:
            samples[k] = expr
    match expr:
        case MathUnaryOp(child=c) | MathGrad(child=c) | MathDiv(child=c) | MathDt(child=c) | MathIntegral(child=c) | MathBoundaryRef(child=c):
            _collect_math_cse(c, counts, samples)
        case MathBinaryOp(left=l, right=r):
            _collect_math_cse(l, counts, samples)
            _collect_math_cse(r, counts, samples)
        case MathClamp(child=c, lower=lo, upper=hi):
            _collect_math_cse(c, counts, samples)
            _collect_math_cse(lo, counts, samples)
            _collect_math_cse(hi, counts, samples)


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
        for eq in eqs:
            lhs_s = _format_math_expr(eq.lhs, 0, cse_aliases)
            rhs_s = _format_math_expr(eq.rhs, 0, cse_aliases)
            if eq.is_piecewise and eq.current_region:
                lines.append(f"    region {eq.current_region.domain_name}: {lhs_s} == {rhs_s}")
            else:
                lines.append(f"    {lhs_s} == {rhs_s}")
        if s_name in math_sys.dirichlet_bcs:
            bc_parts = [f"{side}={_format_math_expr(val, 0, cse_aliases)}" for side, val in math_sys.dirichlet_bcs[s_name].items()]
            lines.append(f"    Dirichlet BCs: {', '.join(bc_parts)}")

    return "\n".join(lines)