"""
Dynamic Runtime Initial Condition Evaluator.

Evaluates initial conditions (y0, ydot0) using active runtime parameter overrides.
Pattern matching cleanly handles algebraic expressions and spatial coordinate mappings.
"""

import math
from typing import Dict, Any, Tuple, List
from ion_flux.runtime.manifest import ExecutableManifest
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer


def evaluate_ic(manifest: ExecutableManifest, current_parameters: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """
    Dynamically evaluates the Initial Conditions (y0) AST using current parameters.
    Occurs at runtime prior to FFI pointer marshaling.
    """
    layout = manifest.layout
    ast_payload = manifest.ast_payload

    y0 = [0.0] * layout.n_states
    ydot0 = [0.0] * layout.n_states

    if not ast_payload:
        return y0, ydot0

    topo = TopologyAnalyzer(ast_payload.get("domains", {}))

    def _eval_ic(node: Dict[str, Any], flat_idx: int, d_name: str) -> float:
        match node:
            case {"type": "Scalar", "value": val}:
                return float(val)

            case {"type": "Parameter", "name": p_name}:
                return current_parameters.get(p_name, manifest.default_parameters.get(p_name, 0.0))

            case {"type": "BinaryOp", "op": op, "left": left, "right": right}:
                l = _eval_ic(left, flat_idx, d_name)
                r = _eval_ic(right, flat_idx, d_name)
                match op:
                    case "add": return l + r
                    case "sub": return l - r
                    case "mul": return l * r
                    case "div": return l / r if r != 0 else 0.0
                    case "pow": return l ** r
                    case "max": return max(l, r)
                    case "min": return min(l, r)
                    case _: return 0.0

            case {"type": "UnaryOp", "op": "coords", **rest}:
                b_axis = rest.get("axis")
                if b_axis and d_name:
                    axes = topo.get_axes(d_name)
                    strides = topo.get_strides(d_name)
                    if b_axis in axes:
                        stride = strides[b_axis]
                        res = topo.domains.get(b_axis, {}).get("resolution", 1)
                        start = topo.domains.get(b_axis, {}).get("start_idx", 0)
                        local_idx = (flat_idx // stride) % res

                        b_base_axis = topo.get_base_axis(b_axis)
                        if b_base_axis in layout.mesh_offsets and "w_centers" in layout.mesh_offsets[b_base_axis]:
                            centers_offset = layout.mesh_offsets[b_base_axis]["w_centers"]
                            norm_center = layout.mesh_cache.get(centers_offset + start + local_idx, 0.0)
                            bounds = topo.domains.get(b_base_axis, {}).get("bounds", (0.0, 1.0))
                            l_phys = float(bounds[1] - bounds[0])
                            return bounds[0] + norm_center * l_phys
                return 0.0

            case {"type": "UnaryOp", "op": op, "child": child}:
                c = _eval_ic(child, flat_idx, d_name)
                match op:
                    case "neg": return -c
                    case "sin": return math.sin(c)
                    case "cos": return math.cos(c)
                    case "exp": return math.exp(c)
                    case "log": return math.log(c) if c > 0 else 0.0
                    case "sqrt": return math.sqrt(c) if c > 0 else 0.0
                    case "abs": return abs(c)
                    case _: return 0.0

            case _:
                return 0.0

    for ic_data in ast_payload.get("initial_conditions", []):
        state_name = ic_data["state"]
        offset, size = layout.state_offsets[state_name]
        d_name = manifest.state_domain_map.get(state_name, "")

        for i in range(size):
            y0[offset + i] = _eval_ic(ic_data["value"], i, d_name)

    return y0, ydot0