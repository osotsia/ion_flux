"""
Normalization Pass: AST -> Math IR Lowering.

Lowers raw AST dictionary trees into a strongly-typed MathSystem (Math IR).
Resolves piecewise regional bounds, Dirichlet boundary exclusions, and moving domains.
"""

from typing import Dict, Any, List, Optional
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._2_middle_end.ast_utils import extract_div_child
from ion_flux.compiler._3_backend.math_ir import (
    MathExpr, MathScalar, MathParameter, MathState, MathBinaryOp, MathUnaryOp,
    MathGrad, MathDiv, MathDt, MathCoords, MathIntegral, MathBoundaryRef,
    MathEquation, MathObservable, MathPiecewiseRegion, MathDirichletOverride, MathSystem
)


class NormalizationPass:
    """Transforms raw AST payloads directly into typed MathSystem representations."""

    def __init__(
        self,
        ast_payload: Dict[str, Any],
        topo: TopologyAnalyzer,
        semantic_ctx: SemanticContext,
        state_map: Dict[str, Any],
        layout: Optional[Any] = None
    ):
        self.ast_payload = ast_payload
        self.topo = topo
        self.semantic_ctx = semantic_ctx
        self.state_map = state_map
        self.layout = layout

    def lower_to_math_ir(self) -> MathSystem:
        """Translates the AST payload into a typed MathSystem."""
        equations = self._lower_equations()
        dirichlet_overrides = self._lower_dirichlet_overrides()
        observables = self._lower_observables()
        dynamic_bindings = self._lower_dynamic_domains()

        return MathSystem(
            equations=equations,
            observables=observables,
            dirichlet_overrides=dirichlet_overrides,
            dynamic_domain_bindings=dynamic_bindings
        )

    def _lower_equations(self) -> List[MathEquation]:
        equations: List[MathEquation] = []

        for eq_data in self.ast_payload.get("equations", []):
            state_name = eq_data["state"]
            state_obj = self.state_map.get(state_name)
            target_domain = getattr(state_obj, "domain", None)
            d_name = target_domain.name if target_domain else None
            d_bcs = self.semantic_ctx.get_dirichlet_bc(state_name)
            last_axis = self.topo.get_axes(d_name)[-1] if d_name else None

            if eq_data.get("type") == "piecewise":
                regions_ast = eq_data["regions"]
                region_divs = {r["domain"]: extract_div_child(r["eq"]) for r in regions_ast}

                lowered_regions = [
                    MathPiecewiseRegion(
                        domain_name=r["domain"],
                        start_idx=r["start_idx"],
                        end_idx=r["end_idx"],
                        expr=self._lower_expr(r["eq"], state_name, d_name),
                        div_flux=(
                            self._lower_expr(region_divs[r["domain"]], state_name, d_name)
                            if region_divs.get(r["domain"]) else None
                        )
                    )
                    for r in regions_ast
                ]

                for reg, reg_ir in zip(regions_ast, lowered_regions):
                    b_axis = self.topo.get_base_axis(reg["domain"])
                    r_start = reg["start_idx"]
                    r_res = reg["end_idx"] - reg["start_idx"]

                    # Exclude boundary nodes claimed by Dirichlet conditions
                    if d_bcs and last_axis and self.topo.get_base_axis(last_axis) == b_axis:
                        domain_start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
                        domain_res = self.topo.domains.get(last_axis, {}).get("resolution", 1)

                        if "left" in d_bcs and r_start == domain_start:
                            r_start += 1
                            r_res -= 1
                        if "right" in d_bcs and r_start + r_res == domain_start + domain_res:
                            r_res -= 1

                    if r_res > 0:
                        lhs_ast = reg["eq"].get("left") if isinstance(reg["eq"], dict) else None
                        rhs_ast = reg["eq"].get("right") if isinstance(reg["eq"], dict) else None

                        equations.append(MathEquation(
                            state_name=state_name,
                            target_domain=d_name,
                            lhs=self._lower_expr(lhs_ast, state_name, d_name),
                            rhs=self._lower_expr(rhs_ast, state_name, d_name),
                            bounds_override={b_axis: (r_start, r_res)},
                            is_piecewise=True,
                            regions=lowered_regions,
                            current_region=reg_ir
                        ))
            else:
                bounds_override = eq_data.get("bounds_override") or {}
                if not bounds_override and d_bcs and last_axis:
                    start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
                    res = self.topo.domains.get(last_axis, {}).get("resolution", 1)
                    b_axis = self.topo.get_base_axis(last_axis)

                    if "left" in d_bcs:
                        start += 1
                        res -= 1
                    if "right" in d_bcs:
                        res -= 1
                    if res > 0:
                        bounds_override[b_axis] = (start, res)

                b_axis_key = self.topo.get_base_axis(last_axis) if last_axis else None
                is_valid = not (d_bcs and last_axis and bounds_override.get(b_axis_key, (0, 1))[1] <= 0)

                if is_valid:
                    eq_dict = eq_data.get("eq", {})
                    lhs_ast = eq_dict.get("left") if isinstance(eq_dict, dict) else None
                    rhs_ast = eq_dict.get("right") if isinstance(eq_dict, dict) else None

                    equations.append(MathEquation(
                        state_name=state_name,
                        target_domain=d_name,
                        lhs=self._lower_expr(lhs_ast, state_name, d_name),
                        rhs=self._lower_expr(rhs_ast, state_name, d_name),
                        bounds_override=bounds_override if bounds_override else None,
                        is_piecewise=False
                    ))

        return equations

    def _lower_dirichlet_overrides(self) -> List[MathDirichletOverride]:
        overrides: List[MathDirichletOverride] = []
        for bc_data in self.ast_payload.get("boundaries", []):
            if bc_data.get("type") == "dirichlet":
                s_name = bc_data["state"]
                s_obj = self.state_map.get(s_name)
                s_dom = getattr(s_obj, "domain", None)
                last_ax = self.topo.get_axes(s_dom.name)[-1] if s_dom else None
                base_ax = self.topo.get_base_axis(last_ax) if last_ax else None

                for side, val_dict in bc_data.get("bcs", {}).items():
                    overrides.append(MathDirichletOverride(
                        state_name=s_name,
                        side=side,
                        axis=base_ax,
                        value_expr=self._lower_expr(val_dict, s_name, s_dom.name if s_dom else None)
                    ))
        return overrides

    def _lower_observables(self) -> List[MathObservable]:
        observables: List[MathObservable] = []
        for obs_data in self.ast_payload.get("observables", []):
            o_name = obs_data["state"]
            o_obj = self.state_map.get(o_name)
            o_dom = getattr(o_obj, "domain", None)
            d_name = o_dom.name if o_dom else None

            if obs_data.get("type") == "piecewise":
                for reg in obs_data.get("regions", []):
                    b_axis = self.topo.get_base_axis(reg["domain"])
                    r_start = reg["start_idx"]
                    r_res = reg["end_idx"] - reg["start_idx"]
                    observables.append(MathObservable(
                        name=o_name,
                        target_domain=d_name,
                        expr=self._lower_expr(reg.get("eq"), o_name, d_name),
                        bounds_override={b_axis: (r_start, r_res)}
                    ))
            else:
                observables.append(MathObservable(
                    name=o_name,
                    target_domain=d_name,
                    expr=self._lower_expr(obs_data.get("eq"), o_name, d_name),
                    bounds_override=obs_data.get("bounds_override")
                ))
        return observables

    def _lower_dynamic_domains(self) -> Dict[str, Dict[str, Any]]:
        dynamic_bindings = {}
        for d_name, binding in self.semantic_ctx.dynamic_domains.items():
            dynamic_bindings[d_name] = {
                "side": binding["side"],
                "rhs_expr": self._lower_expr(binding["rhs"], "", d_name),
                "rhs_ast": binding["rhs"]
            }
        return dynamic_bindings

    def _lower_expr(self, node: Any, current_state: str, current_domain: Optional[str]) -> MathExpr:
        if not isinstance(node, dict):
            return MathScalar(float(node) if isinstance(node, (int, float)) else 0.0)

        bc_id = node.get("_bc_id")
        t = node.get("type")

        if t == "Scalar":
            return MathScalar(float(node["value"]), bc_id=bc_id)

        if t == "Parameter":
            p_name = node["name"]
            p_off = self.layout.get_param_offset(p_name) if self.layout else 0
            return MathParameter(name=p_name, offset=p_off, bc_id=bc_id)

        if t == "State":
            s_name = node["name"]
            s_obj = self.state_map.get(s_name)
            s_dom = getattr(s_obj, "domain", None)
            s_dom_name = s_dom.name if s_dom else None
            off, size = self.layout.state_offsets[s_name] if (self.layout and s_name in self.layout.state_offsets) else (0, 1)
            return MathState(name=s_name, domain_name=s_dom_name, offset=off, size=size, is_ydot=False, bc_id=bc_id)

        if t == "Boundary":
            child_ir = self._lower_expr(node["child"], current_state, current_domain)
            return MathBoundaryRef(child=child_ir, side=node["side"], domain=node.get("domain"), bc_id=bc_id)

        if t == "BinaryOp":
            return MathBinaryOp(
                op=node["op"],
                left=self._lower_expr(node["left"], current_state, current_domain),
                right=self._lower_expr(node["right"], current_state, current_domain),
                bc_id=bc_id
            )

        if t == "UnaryOp":
            op = node["op"]
            child = node["child"]
            if op == "dt":
                child_ir = self._lower_expr(child, current_state, current_domain)
                if isinstance(child_ir, MathState):
                    return MathState(
                        name=child_ir.name,
                        domain_name=child_ir.domain_name,
                        offset=child_ir.offset,
                        size=child_ir.size,
                        is_ydot=True,
                        bc_id=bc_id
                    )
                return MathDt(child_ir, bc_id=bc_id)

            axis = node.get("axis")
            if not axis and current_domain:
                axes = self.topo.get_axes(current_domain)
                axis = axes[-1] if axes else current_domain

            if op == "grad":
                return MathGrad(child=self._lower_expr(child, current_state, axis), axis=axis, bc_id=bc_id)
            if op == "div":
                return MathDiv(child=self._lower_expr(child, current_state, axis), axis=axis, bc_id=bc_id)
            if op == "coords":
                return MathCoords(axis=axis, bc_id=bc_id)
            if op == "integral":
                over_dom = node.get("over")
                return MathIntegral(child=self._lower_expr(child, current_state, over_dom), over_domain=over_dom, bc_id=bc_id)

            return MathUnaryOp(op=op, child=self._lower_expr(child, current_state, current_domain), bc_id=bc_id)

        return MathScalar(0.0)
