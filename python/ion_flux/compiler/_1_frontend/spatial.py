from typing import Dict, Any, Union, List, Optional
from .nodes import Node, UnaryOp, Scalar, validate_identifier

class DomainBoundary(Node):
    __slots__ = ["domain", "side"]
    def __init__(self, domain: "Domain", side: str):
        self.domain = domain
        self.side = side
    def __call__(self) -> "DomainBoundary": return self
    def to_dict(self, tags: Optional[Dict[int, str]] = None) -> Dict[str, Any]: 
        return {"type": "DomainBoundary", "domain": self.domain.name, "side": self.side}
    def __repr__(self) -> str: return f"{self.domain.name}.{self.side}"

class Domain:
    __slots__ = ["bounds", "_explicit_resolution", "coord_sys", "name", "csr_data", "parent", "start_idx", "_original_name", "_sub_regions"]
    def __init__(self, bounds: tuple, resolution: Optional[int] = None, coord_sys: str = "cartesian", name: str = "", csr_data: Optional[Dict] = None, parent=None, start_idx=0):
        self.bounds = bounds
        self._explicit_resolution = resolution
        self.coord_sys = coord_sys
        self.name = validate_identifier(name)
        self.csr_data = csr_data
        self.parent = parent
        self.start_idx = start_idx
        self._sub_regions = []

    @property
    def resolution(self) -> int:
        if self._sub_regions:
            return sum(r.resolution for r in self._sub_regions)
        if self._explicit_resolution is None:
            return 1
        return self._explicit_resolution

    def region(self, bounds: tuple, resolution: int, name: str) -> "Domain":
        """
        Creates a topological sub-mesh mapped securely to the parent domain's contiguous memory.
        Enforces a strict Top-Down grid assembly paradigm.
        """
        name = validate_identifier(name)
        
        if self._explicit_resolution is not None and not self._sub_regions:
            import logging
            logging.warning(f"Domain '{self.name}' has an explicit resolution ({self._explicit_resolution}) but is being sliced into regions. The parent resolution will be dynamically overridden by the sum of its regions.")
            
        # Guarantee strict topological contiguity without floating-point mapping drift
        start_idx = sum(r.resolution for r in self._sub_regions)
        
        new_region = Domain(bounds, resolution, coord_sys=self.coord_sys, name=name, parent=self, start_idx=start_idx)
        self._sub_regions.append(new_region)
        return new_region

    @classmethod
    def from_mesh(cls, mesh_data: Union[str, dict], name: str = "unstructured_mesh", surfaces: Optional[Dict[str, List[int]]] = None) -> "Domain":
        name = validate_identifier(name)
        from ion_flux.compiler._2_middle_end.unstructured import assemble_mesh_csr
        resolution, csr_data = assemble_mesh_csr(mesh_data, surfaces)
        return cls(bounds=(0, 1), resolution=resolution, coord_sys="unstructured", name=name, csr_data=csr_data)
        
    @property
    def coords(self) -> Node: return UnaryOp("coords", Scalar(0.0), axis=self)
    @property
    def left(self) -> DomainBoundary: return DomainBoundary(self, "left")
    @property
    def right(self) -> DomainBoundary: return DomainBoundary(self, "right")
        
    def __mul__(self, other: "Domain") -> "CompositeDomain":
        return CompositeDomain([self, other])
        
    def __add__(self, other: Any):
        raise TypeError("Bottom-Up domain concatenation (domain1 + domain2) is deprecated. Please use Top-Down region slicing (`parent.region(...)`) to guarantee topological contiguity and exact ALE kinematics.")
        
    def __repr__(self) -> str: return self.name or f"Domain({self.bounds})"
    def __set_name__(self, owner, name):
        if not self.name: self.name = validate_identifier(name)

class CompositeDomain:
    __slots__ = ["domains", "_original_name"]
    def __init__(self, domains: List[Domain], name: str = ""):
        self.domains = domains
        self._original_name = validate_identifier(name)
        
    @property
    def name(self) -> str:
        if self._original_name: return self._original_name
        return validate_identifier("_x_".join([d.name for d in self.domains if d.name]))

    @name.setter
    def name(self, value: str) -> None:
        self._original_name = validate_identifier(value)
        
    @property
    def resolution(self) -> int:
        import math
        return math.prod(d.resolution for d in self.domains)
        
    def __repr__(self) -> str: return self.name or f"CompositeDomain({[d.name for d in self.domains]})"
    def __set_name__(self, owner, name):
        if not self._original_name: self._original_name = validate_identifier(name)