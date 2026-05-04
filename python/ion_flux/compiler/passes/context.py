from dataclasses import dataclass, replace
from typing import Optional, Dict, Any, List

@dataclass(frozen=True)
class SpatialContext:
    """
    Immutable context passed down the AST visitor call stack.
    Eliminates stateful bugs during deep recursive traversals (e.g., nested integrals)
    by forcing a unidirectional functional data flow.
    """
    axis: Optional[str] = None
    use_ydot: bool = False
    is_piecewise: bool = False
    piecewise_regions: Optional[List[Dict[str, Any]]] = None
    region_divs: Optional[Dict[str, Any]] = None
    current_region_data: Optional[Dict[str, Any]] = None

    def with_updates(self, **kwargs) -> 'SpatialContext':
        """Creates a functional clone of the context with specific mutations."""
        return replace(self, **kwargs)