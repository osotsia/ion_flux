from typing import Dict, Any, Union, List, Optional
from .nodes import Node, UnaryOp, Scalar

class DomainBoundary(Node):
    __slots__ = ["domain", "side"]
    def __init__(self, domain: "Domain", side: str):
        self.domain = domain
        self.side = side
    def __call__(self) -> "DomainBoundary": return self
    def to_dict(self) -> Dict[str, Any]: 
        return {"type": "DomainBoundary", "domain": self.domain.name, "side": self.side}
    def __repr__(self) -> str: return f"{self.domain.name}.{self.side}"

class ConcatenatedDomain:
    __slots__ = ["domains", "name", "_original_name"]
    def __init__(self, domains: List['Domain'], name: str = ""):
        self.domains = domains
        self.name = name or "_plus_".join(d.name for d in domains)
        
    def __add__(self, other: Union['Domain', 'ConcatenatedDomain']) -> 'ConcatenatedDomain':
        if isinstance(other, Domain): return ConcatenatedDomain(self.domains + [other])
        elif isinstance(other, ConcatenatedDomain): return ConcatenatedDomain(self.domains + other.domains)
        raise TypeError("Can only concatenate Domains.")

class Domain:
    __slots__ = ["bounds", "resolution", "coord_sys", "name", "csr_data", "parent", "start_idx", "_original_name"]
    def __init__(self, bounds: tuple, resolution: int, coord_sys: str = "cartesian", name: str = "", csr_data: Optional[Dict] = None, parent=None, start_idx=0):
        self.bounds = bounds
        self.resolution = resolution
        self.coord_sys = coord_sys
        self.name = name
        self.csr_data = csr_data
        self.parent = parent
        self.start_idx = start_idx

    def region(self, bounds: tuple, resolution: int, name: str) -> "Domain":
        """Creates a topological sub-mesh that shares the memory array stride of the parent Domain."""
        # Use fractional length * parent resolution to prevent floating point orphaned nodes
        # when users perfectly tile face-sharing sub-regions.
        frac = (bounds[0] - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        start_idx = int(round(frac * self.resolution))
        return Domain(bounds, resolution, coord_sys=self.coord_sys, name=name, parent=self, start_idx=start_idx)

    @classmethod
    def from_mesh(cls, mesh_data: Union[str, dict], name: str = "unstructured_mesh", surfaces: Optional[Dict[str, List[int]]] = None) -> "Domain":
        import numpy as np
        if isinstance(mesh_data, str):
            import json
            with open(mesh_data, "r") as f: mesh_data = json.load(f)
                
        nodes = np.array(mesh_data["nodes"], dtype=float)
        elements = np.array(mesh_data["elements"], dtype=int)
        resolution = len(nodes)
        
        from collections import defaultdict
        K_global = defaultdict(float)
        V_nodes = np.zeros(resolution, dtype=float)
        
        for el in elements:
            if len(el) != 4: continue
            p = nodes[el]
            J_mat = np.array([p[1]-p[0], p[2]-p[0], p[3]-p[0]]).T
            detJ = np.linalg.det(J_mat)
            vol = abs(detJ) / 6.0
            if vol < 1e-30: continue
            
            invJ = np.linalg.inv(J_mat)
            gradN = np.zeros((4, 3))
            gradN[1:4, :] = invJ.T
            gradN[0, :] = -np.sum(invJ.T, axis=0)
            
            K_local = vol * (gradN @ gradN.T)
            
            for i in range(4):
                V_nodes[el[i]] += vol / 4.0
                for j in range(4):
                    if i != j: K_global[(el[i], el[j])] -= K_local[i, j]
                        
        row_ptr = [0] * (resolution + 1)
        col_ind, weights = [], []
        
        for i in range(resolution):
            row_ptr[i] = len(col_ind)
            vol_i = max(V_nodes[i], 1e-30)
            for j in range(resolution):
                if (i, j) in K_global:
                    col_ind.append(j)
                    weights.append(K_global[(i, j)] / vol_i)
        row_ptr[resolution] = len(col_ind)
        
        surface_masks = {}
        if surfaces:
            for tag, indices in surfaces.items():
                mask = [0.0] * resolution
                for idx in indices: mask[idx] = 1.0
                surface_masks[tag] = mask
                
        csr_data = {
            "row_ptr": [float(x) for x in row_ptr],
            "col_ind": [float(x) for x in col_ind],
            "weights": weights,
            "volumes": [float(x) for x in V_nodes], # Passed to compilation layer
            "surface_masks": surface_masks
        }
        return cls(bounds=(0, 1), resolution=resolution, coord_sys="unstructured", name=name, csr_data=csr_data)
        
    @property
    def coords(self) -> Node: return UnaryOp("coords", Scalar(0.0), axis=self)
    @property
    def left(self) -> DomainBoundary: return DomainBoundary(self, "left")
    @property
    def right(self) -> DomainBoundary: return DomainBoundary(self, "right")
        
    def __mul__(self, other: "Domain") -> "CompositeDomain":
        return CompositeDomain([self, other])
        
    def __add__(self, other: Union["Domain", "ConcatenatedDomain"]) -> "ConcatenatedDomain":
        if isinstance(other, Domain): return ConcatenatedDomain([self, other])
        elif isinstance(other, ConcatenatedDomain): return ConcatenatedDomain([self] + other.domains)
        raise TypeError("Can only concatenate Domains.")
        
    def __repr__(self) -> str: return self.name or f"Domain({self.bounds})"
    def __set_name__(self, owner, name):
        if not self.name: self.name = name

class CompositeDomain:
    __slots__ = ["domains", "name", "_original_name"]
    def __init__(self, domains: List[Domain], name: str = ""):
        self.domains = domains
        self.name = name or "_x_".join([d.name for d in domains if d.name])
        
    @property
    def resolution(self) -> int:
        import math
        return math.prod(d.resolution for d in self.domains)
        
    def __repr__(self) -> str: return self.name or f"CompositeDomain({[d.name for d in self.domains]})"
    def __set_name__(self, owner, name):
        if not self.name: self.name = name