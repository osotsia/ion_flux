"""Lowering Stage: Math IR and FVM Discretization."""
from .math_ir import MathSystem, MathEquation, MathObservable, MathExpr
from .normalization import NormalizationPass
from .discretizer import FVMDiscretizer, IndexManager

__all__ = [
    "MathSystem",
    "MathEquation",
    "MathObservable",
    "MathExpr",
    "NormalizationPass",
    "FVMDiscretizer",
    "IndexManager",
]