"""
Math Intermediate Representation (Math IR).

Represents N-Dimensional continuum mathematics, tensor differential operators,
and domain boundary constraints prior to spatial discretization and memory layout flattening.
"""

from dataclasses import dataclass
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


@dataclass
class MathSystem:
    equations: List[MathEquation]
    observables: List[MathObservable]
    dirichlet_overrides: List[MathDirichletOverride]
    dynamic_domain_bindings: Dict[str, Dict[str, Any]]


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

        case _:
            pass

    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]