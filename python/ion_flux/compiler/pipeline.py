"""
Central Compiler Orchestration Pipeline.

Directs the unidirectional staged compilation flow:
  Stage 1: Intent Capture (Python AST)
  Stage 2: Topology Verification & Semantic Parsing (Middle-End)
  Stage 3: Normalization & Lowering to Math IR (Backend)
  Stage 4: FVM Discretization & Mechanical C++ Codegen
  Stage 5: Static AD Analysis via CPR Graph Coloring
"""

import itertools
from typing import Dict, Any, List, Optional
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._2_middle_end.verification import verify_manifold
from ion_flux.compiler._3_backend.normalization import NormalizationPass
from ion_flux.compiler._3_backend.cpr_orchestrator import compute_cpr
from ion_flux.compiler._3_backend.math_ir import MathSystem, extract_math_state_names, has_time_derivative
from ion_flux.compiler._4_codegen.builder import generate_cpp


class Compiler:
    """The central orchestrator for the Staged Lowering Pipeline."""

    @staticmethod
    def compile(
        ast_payload: Dict[str, Any],
        layout: Any,
        states: List[Any],
        observables: List[Any],
        target: str,
        jacobian_bandwidth: Optional[int] = None
    ) -> Dict[str, Any]:
        # --- Stage 2: Topological Validation & Semantic Resolution ---
        topo = TopologyAnalyzer(ast_payload.get("domains", {}))
        verify_manifold(ast_payload)
        semantic_ctx = SemanticContext(ast_payload)

        state_map = {s.name: s for s in states}
        state_map.update({o.name: o for o in observables})

        # --- Stage 3: Normalization (AST Dict -> Math IR) ---
        norm_pass = NormalizationPass(ast_payload, topo, semantic_ctx, state_map, layout)
        math_sys: MathSystem = norm_pass.lower_to_math_ir()

        Compiler._verify_system_rank(math_sys, layout)

        if jacobian_bandwidth is None:
            jacobian_bandwidth = Compiler._compute_symbolic_bandwidth(layout, states, math_sys)

        # --- Stage 4: Discretization & Codegen (Math IR -> Compute IR -> C++) ---
        cpp_source, eq_stmts = generate_cpp(
            math_sys=math_sys,
            layout=layout,
            topo=topo,
            semantic_ctx=semantic_ctx,
            state_map=state_map,
            target=target
        )

        # --- Stage 5: CPR Sparsity Analysis on Compute IR ---
        cpr_cache = compute_cpr(eq_stmts, layout, jacobian_bandwidth)

        state_domain_map = {s.name: getattr(s.domain, "name", "") for s in states}
        id_arr = Compiler._compute_differential_mask(math_sys, layout, topo, state_domain_map)

        return {
            "cpp_source": cpp_source,
            "cpr_cache": cpr_cache,
            "id_arr": id_arr,
            "ast_payload": ast_payload,
            "topo": topo,
            "jacobian_bandwidth": jacobian_bandwidth,
            "math_sys": math_sys
        }

    @staticmethod
    def _compute_differential_mask(
        math_sys: MathSystem,
        layout: Any,
        topo: TopologyAnalyzer,
        state_domain_map: Dict[str, str],
    ) -> List[float]:
        """Derives the DAE differential mask (1.0=ODE/PDE, 0.0=Algebraic) directly from MathSystem."""
        id_arr = [0.0] * layout.n_states

        for eq in math_sys.equations:
            if not (has_time_derivative(eq.lhs) or has_time_derivative(eq.rhs)):
                continue
            offset, size = layout.state_offsets[eq.state_name]

            if eq.is_piecewise and eq.current_region:
                d_name = state_domain_map.get(eq.state_name, "")
                if not d_name:
                    id_arr[offset] = 1.0
                    continue

                axes = topo.get_axes(d_name)
                strides = topo.get_strides(d_name)
                b_axis = topo.get_base_axis(eq.current_region.domain_name)

                ranges = []
                for axis in axes:
                    base = topo.get_base_axis(axis)
                    if base == b_axis:
                        ranges.append(range(eq.current_region.start_idx, eq.current_region.end_idx))
                    else:
                        ranges.append(range(topo.domains.get(axis, {}).get("resolution", 1)))

                for indices in itertools.product(*ranges):
                    flat_idx = 0
                    for axis, idx in zip(axes, indices):
                        flat_idx += idx * strides[axis]
                    id_arr[offset + flat_idx] = 1.0
            else:
                for i in range(offset, offset + size):
                    id_arr[i] = 1.0

        for state_name, bcs in math_sys.dirichlet_bcs.items():
            if state_name not in layout.state_offsets:
                continue
            offset, size = layout.state_offsets[state_name]
            d_name = state_domain_map.get(state_name, "")

            if not d_name:
                if "left" in bcs:
                    id_arr[offset] = 0.0
                if "right" in bcs:
                    id_arr[offset + size - 1] = 0.0
                continue

            axes = topo.get_axes(d_name)
            strides = topo.get_strides(d_name)
            b_axis = axes[-1]
            coord_sys = topo.domains.get(b_axis, {}).get("coord_sys", "cartesian")

            if coord_sys == "unstructured":
                surfaces = layout.mesh_offsets.get(b_axis, {}).get("surfaces", {})
                for side in bcs:
                    if side in surfaces:
                        mask_off = surfaces[side]
                        for i in range(size):
                            if layout.mesh_cache.get(mask_off + i, 0.0) > 0.5:
                                id_arr[offset + i] = 0.0
                continue

            b_res = topo.domains.get(b_axis, {}).get("resolution", 1)
            ranges = [
                [0] if axis == b_axis else range(topo.domains.get(axis, {}).get("resolution", 1))
                for axis in axes
            ]
            for indices in itertools.product(*ranges):
                base_flat = sum(idx * strides[axis] for axis, idx in zip(axes, indices))
                if "left" in bcs:
                    id_arr[offset + base_flat] = 0.0
                if "right" in bcs:
                    id_arr[offset + base_flat + (b_res - 1) * strides[b_axis]] = 0.0

        return id_arr

    @staticmethod
    def _verify_system_rank(math_sys: MathSystem, layout: Any) -> None:
        """Validates that all declared states have at least one governing equation."""
        targeted_states = {eq.state_name for eq in math_sys.equations}
        for state_name in layout.state_offsets.keys():
            if state_name not in targeted_states:
                raise ValueError(f"Unconstrained state detected: '{state_name}'. Rank deficiency in system.")

    @staticmethod
    def _compute_symbolic_bandwidth(layout: Any, states: List[Any], math_sys: MathSystem) -> int:
        """
        Determines structural Jacobian bandwidth from the Math IR.
        Returns:
            -1: Unstructured CSR mesh (Matrix-Free GMRES).
             0: Dense coupling or dynamic ALE grid kinematics.
            >0: Banded system width.
        """
        if any(getattr(s.domain, "coord_sys", "") == "unstructured" for s in states):
            return -1

        if math_sys.dynamic_domain_bindings:
            return 0

        max_bw = 0
        for eq in math_sys.equations:
            target_state = eq.state_name
            if target_state not in layout.state_offsets:
                continue

            off_t, size_t = layout.state_offsets[target_state]
            if size_t > 1:
                max_bw = max(max_bw, 2)

            deps = extract_math_state_names(eq.lhs) + extract_math_state_names(eq.rhs)
            for dep in deps:
                if dep not in layout.state_offsets:
                    continue
                off_d, _ = layout.state_offsets[dep]
                if abs(off_t - off_d) > 0:
                    return 0

        return max_bw if max_bw > 0 else 0