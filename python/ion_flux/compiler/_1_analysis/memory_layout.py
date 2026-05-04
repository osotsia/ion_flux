from typing import List, Dict, Tuple, Any
from ion_flux.dsl.core import State, Parameter, Observable
import math

class MemoryLayout:
    def __init__(self, states: List[State], parameters: List[Parameter], observables: List[Observable] = None, all_domains: List[Any] = None):
        self.state_offsets: Dict[str, Tuple[int, int]] = {}
        self.param_offsets: Dict[str, Tuple[int, int]] = {}
        self.obs_offsets: Dict[str, Tuple[int, int]] = {}
        
        self.n_states = 0
        self.n_params = 0
        self.n_obs = 0

        sorted_states = sorted(states, key=lambda s: s.name)
        for s in sorted_states:
            size = s.domain.resolution if s.domain else 1
            self.state_offsets[s.name] = (self.n_states, size)
            self.n_states += size

        sorted_params = sorted(parameters, key=lambda p: p.name)
        for p in sorted_params:
            self.param_offsets[p.name] = (self.n_params, 1)
            self.n_params += 1
            
        if observables:
            sorted_obs = sorted(observables, key=lambda o: o.name)
            for o in sorted_obs:
                size = o.domain.resolution if o.domain else 1
                self.obs_offsets[o.name] = (self.n_obs, size)
                self.n_obs += size
            
        self.p_length = self.n_params
        self.m_length = 0
        self.mesh_offsets = {}
        self.mesh_cache = {}
        
        def _get_domains(d):
            if d is None: return []
            if getattr(d, "type", None) == "composite" or type(d).__name__ == "CompositeDomain":
                res = []
                for sub in getattr(d, "domains", []):
                    res.extend(_get_domains(sub))
                return res
            return [d]

        def get_root_domain(d):
            while getattr(d, "parent", None) is not None:
                d = d.parent
            return d

        # Track unstructured CSR graph arrays and standard Top-Down 1D grids
        root_domains = set()
        
        for s in sorted_states + (observables or []):
            for d in _get_domains(getattr(s, "domain", None)):
                if getattr(d, "csr_data", None):
                    if d.name not in self.mesh_offsets:
                        csr = d.csr_data
                        offsets = {}
                        for key in ["weights", "row_ptr", "col_ind", "volumes"]:
                            if key in csr:
                                offsets[key] = self.m_length
                                for v in csr[key]:
                                    self.mesh_cache[self.m_length] = float(v)
                                    self.m_length += 1
                        offsets["surfaces"] = {}
                        for tag, mask in csr.get("surface_masks", {}).items():
                            offsets["surfaces"][tag] = self.m_length
                            for v in mask:
                                self.mesh_cache[self.m_length] = float(v)
                                self.m_length += 1
                        self.mesh_offsets[d.name] = offsets
                else:
                    root_domains.add(get_root_domain(d))
                    
        # Extract metadata for domains even if they lack bound states (e.g. for unbound fx.integrals)
        for d in (all_domains or []):
            for sub_d in _get_domains(d):
                if getattr(sub_d, "csr_data", None):
                    pass
                else:
                    root_domains.add(get_root_domain(sub_d))

        # Phase 1: Normalized Geometry Mapping (Node-Centered FVM)
        for root_d in sorted(list(root_domains), key=lambda d: d.name):
            if root_d.name not in self.mesh_offsets:
                self.mesh_offsets[root_d.name] = {}
                
            bounds = root_d.bounds
            L_phys = bounds[1] - bounds[0]
            coord_sys = root_d.coord_sys
            
            regions = root_d._sub_regions if root_d._sub_regions else [root_d]
            regions = sorted(regions, key=lambda r: r.bounds[0])
            
            faces = [0.0]
            centers = []
            
            num_regions = len(regions)
            for i, reg in enumerate(regions):
                L_k = reg.bounds[1] - reg.bounds[0]
                L_k_norm = L_k / L_phys if L_phys > 0 else 0.0
                N_k = reg.resolution
                
                # Resolving "Effective Cells" for node-centered geometry mapping.
                # Boundary regions contain nodes that rest exactly on the absolute edge (half-volume).
                if num_regions == 1:
                    C_k = max(N_k - 1.0, 1.0)
                else:
                    if i == 0 or i == num_regions - 1:
                        C_k = N_k - 0.5
                    else:
                        C_k = N_k
                        
                du_k = L_k_norm / C_k if C_k > 0 else 0.0
                
                for j in range(N_k):
                    is_first = (i == 0 and j == 0)
                    is_last = (i == num_regions - 1 and j == N_k - 1)
                    
                    if is_first and is_last:
                        center = 0.5 * L_k_norm
                        face = L_k_norm
                    elif is_first:
                        center = 0.0
                        face = 0.5 * du_k
                    elif is_last:
                        center = 1.0
                        face = 1.0
                    else:
                        center = faces[-1] + 0.5 * du_k
                        face = faces[-1] + du_k
                        
                    centers.append(center)
                    faces.append(face)
                    
            N = len(centers)
            w_dx_faces = [centers[i+1] - centers[i] for i in range(N-1)]
            
            w_V_nodes = []
            for i in range(N):
                u_L = faces[i]
                u_R = faces[i+1]
                if coord_sys == "spherical":
                    vol = (4.0/3.0) * math.pi * (u_R**3 - u_L**3)
                elif coord_sys == "cylindrical":
                    vol = 0.5 * (u_R**2 - u_L**2)
                else:
                    vol = u_R - u_L
                w_V_nodes.append(vol)
                
            w_A_faces = []
            for i in range(N):
                u_f = faces[i]
                if coord_sys == "spherical":
                    area = 4.0 * math.pi * (u_f**2)
                elif coord_sys == "cylindrical":
                    area = u_f
                else:
                    area = 1.0
                w_A_faces.append(area)
                
            # Cap the final boundary face area mapping
            u_f = faces[-1]
            if coord_sys == "spherical":
                w_A_faces.append(4.0 * math.pi * (u_f**2))
            elif coord_sys == "cylindrical":
                w_A_faces.append(u_f)
            else:
                w_A_faces.append(1.0)
                
            # Volume sanity check to ensure no geometric distortion
            total_vol = sum(w_V_nodes)
            expected_vol = 1.0
            if coord_sys == "spherical": expected_vol = (4.0/3.0) * math.pi
            elif coord_sys == "cylindrical": expected_vol = 0.5
            
            if abs(total_vol - expected_vol) > 1e-10:
                raise RuntimeError(f"Normalized volume integration failed for domain '{root_d.name}'. Expected {expected_vol}, got {total_vol}.")
                    
            for arr_name, arr_data in [("w_dx_faces", w_dx_faces), ("w_V_nodes", w_V_nodes), ("w_A_faces", w_A_faces), ("w_centers", centers)]:
                self.mesh_offsets[root_d.name][arr_name] = self.m_length
                for val in arr_data:
                    self.mesh_cache[self.m_length] = float(val)
                    self.m_length += 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryLayout":
        obj = cls([], [], [], [])
        obj.state_offsets = data["state_offsets"]
        obj.param_offsets = data["param_offsets"]
        obj.obs_offsets = data.get("obs_offsets", {})
        obj.n_states = data["n_states"]
        obj.n_params = data["n_params"]
        obj.n_obs = data.get("n_obs", 0)
        obj.p_length = data.get("p_length", obj.n_params)
        obj.m_length = data.get("m_length", 0)
        obj.mesh_offsets = data.get("mesh_offsets", {})
        
        raw_cache = data.get("mesh_cache", {})
        obj.mesh_cache = {int(k): float(v) for k, v in raw_cache.items()}
        return obj

    def get_state_offset(self, name: str) -> int:
        return self.state_offsets[name][0]

    def get_param_offset(self, name: str) -> int:
        return self.param_offsets[name][0]

    def get_mesh_data(self) -> List[float]:
        """Returns the isolated unstructured mesh and normalized geometric arrays."""
        m_list = [0.0] * self.m_length
        for k, v in self.mesh_cache.items():
            m_list[k] = v
        return m_list