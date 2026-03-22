from typing import List, Dict, Tuple, Any
from ion_flux.dsl.core import State, Parameter

class MemoryLayout:
    def __init__(self, states: List[State], parameters: List[Parameter]):
        self.state_offsets: Dict[str, Tuple[int, int]] = {}
        self.param_offsets: Dict[str, Tuple[int, int]] = {}
        
        self.n_states = 0
        self.n_params = 0

        sorted_states = sorted(states, key=lambda s: s.name)
        for s in sorted_states:
            size = s.domain.resolution if s.domain else 1
            self.state_offsets[s.name] = (self.n_states, size)
            self.n_states += size

        sorted_params = sorted(parameters, key=lambda p: p.name)
        for p in sorted_params:
            self.param_offsets[p.name] = (self.n_params, 1)
            self.n_params += 1
            
        self.p_length = self.n_params
        self.mesh_offsets = {}
        self.mesh_cache = {}
        
        # Sequentially allocate parameter vector space for unstructured graph buffers
        # and pre-cache it so stateless Engines can instantly re-pack it
        for s in sorted_states:
            if s.domain and getattr(s.domain, "csr_data", None):
                d = s.domain
                if d.name not in self.mesh_offsets:
                    csr = d.csr_data
                    offsets = {}
                    
                    for key in ["weights", "row_ptr", "col_ind"]:
                        offsets[key] = self.p_length
                        for v in csr[key]:
                            self.mesh_cache[self.p_length] = float(v)
                            self.p_length += 1
                            
                    offsets["surfaces"] = {}
                    for tag, mask in csr.get("surface_masks", {}).items():
                        offsets["surfaces"][tag] = self.p_length
                        for v in mask:
                            self.mesh_cache[self.p_length] = float(v)
                            self.p_length += 1
                            
                    self.mesh_offsets[d.name] = offsets

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryLayout":
        obj = cls([], [])
        obj.state_offsets = data["state_offsets"]
        obj.param_offsets = data["param_offsets"]
        obj.n_states = data["n_states"]
        obj.n_params = data["n_params"]
        obj.p_length = data.get("p_length", obj.n_params)
        obj.mesh_offsets = data.get("mesh_offsets", {})
        
        # Keys in JSON convert to string, explicitly cast offsets back to int
        raw_cache = data.get("mesh_cache", {})
        obj.mesh_cache = {int(k): float(v) for k, v in raw_cache.items()}
        return obj

    def get_state_offset(self, name: str) -> int:
        return self.state_offsets[name][0]

    def get_param_offset(self, name: str) -> int:
        return self.param_offsets[name][0]

    def pack_mesh_data(self, p_list: List[float]):
        """Injects static unstructured mesh topological vectors directly into parameters."""
        for k, v in self.mesh_cache.items():
            p_list[k] = v