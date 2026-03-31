class TelemetryReport:
    """Diagnostic metrics guiding memory and performance optimizations."""
    __slots__ = ["model_len", "l1_cache_hit_estimate", "avg_jump_distance", "sparsity"]
    def __init__(self, n_states: int, bandwidth: int):
        self.model_len = n_states
        
        # Correct L1 hit-rate mapping reflecting actual hardware footprints
        if n_states <= 1:
            self.avg_jump_distance = 0.0
            self.l1_cache_hit_estimate = 1.0
            self.sparsity = 0.0
        else:
            total_elements = n_states ** 2
            if bandwidth == 0:
                self.avg_jump_distance = float(n_states)
                active_elements = total_elements
            elif bandwidth == -1:
                self.avg_jump_distance = 5.0  # Common average for 3D unstructured nodes
                active_elements = n_states * 5
            else:
                self.avg_jump_distance = float(bandwidth)
                active_elements = min(total_elements, n_states * (2 * bandwidth + 1))
                
            self.sparsity = 1.0 - (active_elements / total_elements)
            working_set_bytes = active_elements * 8
            
            if working_set_bytes <= 32768: # Standard 32KB L1 Data Cache
                self.l1_cache_hit_estimate = 0.99
            else:
                cache_lines = working_set_bytes / 64.0
                penalty = min((self.avg_jump_distance * n_states) / cache_lines, 1.0)
                self.l1_cache_hit_estimate = max(0.01, 1.0 - penalty)

    def __repr__(self) -> str:
        return (f"TelemetryReport(states={self.model_len}, "
                f"L1_hit_rate={self.l1_cache_hit_estimate:.1%}, "
                f"avg_jump={self.avg_jump_distance:.1f}, "
                f"sparsity={self.sparsity:.1%})")



