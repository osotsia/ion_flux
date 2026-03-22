from typing import List, Dict, Tuple, Any
from ion_flux.dsl.core import State, Parameter

class MemoryLayout:
    """
    Computes contiguous memory offsets for flat C-arrays passed across the FFI.
    Translates multidimensional physics states into a 1D vector schema.
    """
    def __init__(self, states: List[State], parameters: List[Parameter]):
        self.state_offsets: Dict[str, Tuple[int, int]] = {}
        self.param_offsets: Dict[str, Tuple[int, int]] = {}
        
        self.n_states = 0
        self.n_params = 0

        # Sort alphabetically to guarantee deterministic compilation hashes
        sorted_states = sorted(states, key=lambda s: s.name)
        for s in sorted_states:
            size = s.domain.resolution if s.domain else 1
            self.state_offsets[s.name] = (self.n_states, size)
            self.n_states += size

        sorted_params = sorted(parameters, key=lambda p: p.name)
        for p in sorted_params:
            size = 1 # Parameters are strictly scalars in this architecture
            self.param_offsets[p.name] = (self.n_params, size)
            self.n_params += size

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryLayout":
        """Reconstructs the precise memory topology from a serialized JSON manifest."""
        obj = cls([], [])
        obj.state_offsets = data["state_offsets"]
        obj.param_offsets = data["param_offsets"]
        obj.n_states = data["n_states"]
        obj.n_params = data["n_params"]
        return obj

    def get_state_offset(self, name: str) -> int:
        if name not in self.state_offsets:
            raise KeyError(f"State '{name}' not found in memory layout.")
        return self.state_offsets[name][0]

    def get_param_offset(self, name: str) -> int:
        if name not in self.param_offsets:
            raise KeyError(f"Parameter '{name}' not found in memory layout.")
        return self.param_offsets[name][0]