from typing import Dict, Any, List

class TopologyAnalyzer:
    """Resolves hierarchical and composite domain topologies into N-dimensional traits."""
    def __init__(self, domains: Dict[str, Any]):
        self.domains = domains

    def get_axes(self, domain_name: str) -> List[str]:
        """Flattens a composite domain into its fundamental 1D axes."""
        if not domain_name: return []
        d = self.domains.get(domain_name)
        if not d: return [domain_name]
        
        if d.get("type") == "composite":
            axes = []
            for sub in d.get("domains", []):
                axes.extend(self.get_axes(sub))
            return axes
        return [domain_name]

    def get_base_axis(self, domain_name: str) -> str:
        """Resolves a sub-region to its foundational parent axis."""
        d = self.domains.get(domain_name, {})
        parent = d.get("parent")
        return self.get_base_axis(parent) if parent else domain_name

    def get_strides(self, domain_name: str) -> Dict[str, int]:
        """Calculates the flat C-array memory stride for each axis in the domain."""
        axes = self.get_axes(domain_name)
        strides = {}
        current_stride = 1
        for axis in reversed(axes):
            strides[axis] = current_stride
            res = self.domains.get(axis, {}).get("resolution", 1)
            current_stride *= res
        return strides