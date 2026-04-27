from typing import List, Dict, Tuple, Any
from ion_flux.dsl.core import State, Parameter, Observable

class MemoryLayout:
    def __init__(self, states: List[State], parameters: List[Parameter], observables: List[Observable] = None):
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

        # Deeply walk all domain boundaries/multi-scale dependencies to extract CSR graph matrices
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryLayout":
        obj = cls([], [], [])
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
        """Returns the isolated unstructured mesh data array."""
        m_list = [0.0] * self.m_length
        for k, v in self.mesh_cache.items():
            m_list[k] = v
        return m_list