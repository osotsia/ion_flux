"""
Normalization Pass: AST -> Math IR Lowering.

Transforms raw untyped AST dictionary payloads into strongly-typed Math IR representations,
resolving boundary conditions, piecewise regional domains, and Dirichlet boundary shrinkages.
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
    """
    Transforms AST dictionaries into a clean MathSystem IR, unrolling
    syntactic sugar such as piecewise domains and explicit boundary overrides.
    """
    def __init__(self, ast_payload: Dict[str, Any], topo: TopologyAnalyzer, 
                 semantic_ctx: SemanticContext, state_map: Dict[str, Any], layout: Optional[Any] = None):
        self.ast_payload = ast_payload
        self.topo = topo
        self.semantic_ctx = semantic_ctx
        self.state_map = state_map
        self.layout = layout

    def run(self) -> Dict[str, Any]:
        """
        Executes normalization without mutating ast_payload with non-serializable objects.
        """
        normalized = self.ast_payload.copy()
        normalized["equations"] = self._normalize_equations(self.ast_payload.get("equations", []))
        normalized["observables"] = self._normalize_equations(self.ast_payload.get("observables", []), is_obs=True)
        return normalized

    def lower_to_math_ir(self) -> MathSystem:
        """Translates the AST payload into typed Math IR structures."""
        equations: List[MathEquation] = []
        observables: List[MathObservable] = []
        dirichlet_overrides: List[MathDirichletOverride] = []

        # 1. Lower Equations
        for eq_data in self.ast_payload.get("equations", []):
            state_name = eq_data["state"]
            state_obj = self.state_map.get(state_name)
            target_domain = getattr(state_obj, "domain", None)
            d_name = target_domain.name if target_domain else None
            d_bcs = self.semantic_ctx.get_dirichlet_bc(state_name)
            last_axis = self.topo.get_axes(d_name)[-1] if d_name else None

            pw_info = eq_data.get("piecewise_info")
            if pw_info:
                # Pre-normalized piecewise equation
                regions_ast = pw_info["regions"]
                current_reg_ast = pw_info["current_region"]
                region_divs_ast = pw_info["region_divs"]

                lowered_regions = []
                for reg in regions_ast:
                    r_dom = reg["domain"]
                    r_start = reg["start_idx"]
                    r_end = reg["end_idx"]
                    reg_eq_ast = reg["eq"]
                    lowered_eq_expr = self._lower_expr(reg_eq_ast, state_name, d_name)
                    div_child_ast = region_divs_ast.get(r_dom)
                    lowered_div_child = self._lower_expr(div_child_ast, state_name, d_name) if div_child_ast else None
                    lowered_regions.append(MathPiecewiseRegion(
                        domain_name=r_dom,
                        start_idx=r_start,
                        end_idx=r_end,
                        expr=lowered_eq_expr,
                        div_flux=lowered_div_child
                    ))

                curr_dom = current_reg_ast["domain"]
                curr_div_ast = region_divs_ast.get(curr_dom)
                curr_div_expr = self._lower_expr(curr_div_ast, state_name, d_name) if curr_div_ast else None
                current_reg_ir = MathPiecewiseRegion(
                    domain_name=curr_dom,
                    start_idx=current_reg_ast["start_idx"],
                    end_idx=current_reg_ast["end_idx"],
                    expr=self._lower_expr(current_reg_ast["eq"], state_name, d_name),
                    div_flux=curr_div_expr
                )

                lhs_expr = self._lower_expr(eq_data["eq"].get("left"), state_name, d_name) if isinstance(eq_data["eq"], dict) and "left" in eq_data["eq"] else MathScalar(0.0)
                rhs_expr = self._lower_expr(eq_data["eq"].get("right"), state_name, d_name) if isinstance(eq_data["eq"], dict) and "right" in eq_data["eq"] else MathScalar(0.0)

                equations.append(MathEquation(
                    state_name=state_name,
                    target_domain=d_name,
                    lhs=lhs_expr,
                    rhs=rhs_expr,
                    bounds_override=eq_data.get("bounds_override"),
                    is_piecewise=True,
                    regions=lowered_regions,
                    current_region=current_reg_ir
                ))

            elif eq_data.get("type") == "piecewise":
                # Raw piecewise equation
                regions_ast = eq_data["regions"]
                region_divs = {r["domain"]: extract_div_child(r["eq"]) for r in regions_ast}
                lowered_regions = []
                for reg in regions_ast:
                    r_domain = reg["domain"]
                    r_start = reg["start_idx"]
                    r_end = reg["end_idx"]
                    reg_eq_ast = reg["eq"]
                    lowered_eq_expr = self._lower_expr(reg_eq_ast, state_name, d_name)
                    div_child_ast = region_divs.get(r_domain)
                    lowered_div_child = self._lower_expr(div_child_ast, state_name, d_name) if div_child_ast else None

                    lowered_regions.append(MathPiecewiseRegion(
                        domain_name=r_domain,
                        start_idx=r_start,
                        end_idx=r_end,
                        expr=lowered_eq_expr,
                        div_flux=lowered_div_child
                    ))

                for reg, reg_ir in zip(regions_ast, lowered_regions):
                    b_axis = self.topo.get_base_axis(reg["domain"])
                    r_start = reg["start_idx"]
                    r_res = reg["end_idx"] - reg["start_idx"]

                    if d_bcs and last_axis and self.topo.get_base_axis(last_axis) == b_axis:
                        domain_start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
                        domain_res = self.topo.domains.get(last_axis, {}).get("resolution", 1)

                        if "left" in d_bcs and r_start == domain_start:
                            r_start += 1
                            r_res -= 1
                        if "right" in d_bcs and r_start + r_res == domain_start + domain_res:
                            r_res -= 1

                    if r_res > 0:
                        lhs_expr = self._lower_expr(reg["eq"].get("left"), state_name, d_name) if isinstance(reg["eq"], dict) and "left" in reg["eq"] else MathScalar(0.0)
                        rhs_expr = self._lower_expr(reg["eq"].get("right"), state_name, d_name) if isinstance(reg["eq"], dict) and "right" in reg["eq"] else MathScalar(0.0)

                        equations.append(MathEquation(
                            state_name=state_name,
                            target_domain=d_name,
                            lhs=lhs_expr,
                            rhs=rhs_expr,
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

                is_valid = True
                if d_bcs and last_axis and bounds_override.get(self.topo.get_base_axis(last_axis), (0, 1))[1] <= 0:
                    is_valid = False

                if is_valid:
                    eq_dict = eq_data["eq"]
                    lhs_expr = self._lower_expr(eq_dict.get("left"), state_name, d_name) if isinstance(eq_dict, dict) and "left" in eq_dict else MathScalar(0.0)
                    rhs_expr = self._lower_expr(eq_dict.get("right"), state_name, d_name) if isinstance(eq_dict, dict) and "right" in eq_dict else MathScalar(0.0)

                    equations.append(MathEquation(
                        state_name=state_name,
                        target_domain=d_name,
                        lhs=lhs_expr,
                        rhs=rhs_expr,
                        bounds_override=bounds_override if bounds_override else None,
                        is_piecewise=False
                    ))

        # 2. Lower Dirichlet Overrides
        for bc_data in self.ast_payload.get("boundaries", []):
            if bc_data["type"] == "dirichlet":
                s_name = bc_data["state"]
                s_obj = self.state_map.get(s_name)
                s_dom = getattr(s_obj, "domain", None)
                last_ax = self.topo.get_axes(s_dom.name)[-1] if s_dom else None
                base_ax = self.topo.get_base_axis(last_ax) if last_ax else None

                for side, val_dict in bc_data["bcs"].items():
                    val_ir = self._lower_expr(val_dict, s_name, s_dom.name if s_dom else None)
                    dirichlet_overrides.append(MathDirichletOverride(
                        state_name=s_name,
                        side=side,
                        axis=base_ax,
                        value_expr=val_ir
                    ))

        # 3. Lower Observables
        for obs_data in self.ast_payload.get("observables", []):
            o_name = obs_data["state"]
            o_obj = self.state_map.get(o_name)
            o_dom = getattr(o_obj, "domain", None)
            d_name = o_dom.name if o_dom else None
            expr_ir = self._lower_expr(obs_data.get("eq"), o_name, d_name)
            observables.append(MathObservable(
                name=o_name,
                target_domain=d_name,
                expr=expr_ir,
                bounds_override=obs_data.get("bounds_override")
            ))

        # 4. Extract Moving Mesh Bindings
        dynamic_bindings = {}
        for d_name, binding in self.semantic_ctx.dynamic_domains.items():
            rhs_expr = self._lower_expr(binding["rhs"], "", d_name)
            dynamic_bindings[d_name] = {
                "side": binding["side"],
                "rhs_expr": rhs_expr,
                "rhs_ast": binding["rhs"]
            }

        return MathSystem(
            equations=equations,
            observables=observables,
            dirichlet_overrides=dirichlet_overrides,
            dynamic_domain_bindings=dynamic_bindings
        )

    def _lower_expr(self, node: Any, current_state: str, current_domain: Optional[str]) -> MathExpr:
        """Recursively parses an AST node into a typed MathExpr preserving _bc_id."""
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
            off, size = self.layout.state_offsets[s_name] if self.layout and s_name in self.layout.state_offsets else (0, 1)
            return MathState(name=s_name, domain_name=s_dom_name, offset=off, size=size, is_ydot=False, bc_id=bc_id)

        if t == "Boundary":
            child_ir = self._lower_expr(node["child"], current_state, current_domain)
            return MathBoundaryRef(child=child_ir, side=node["side"], domain=node.get("domain"), bc_id=bc_id)

        if t == "BinaryOp":
            left_ir = self._lower_expr(node["left"], current_state, current_domain)
            right_ir = self._lower_expr(node["right"], current_state, current_domain)
            return MathBinaryOp(op=node["op"], left=left_ir, right=right_ir, bc_id=bc_id)

        if t == "UnaryOp":
            op = node["op"]
            child = node["child"]
            if op == "dt":
                child_ir = self._lower_expr(child, current_state, current_domain)
                if isinstance(child_ir, MathState):
                    return MathState(
                        name=child_ir.name, domain_name=child_ir.domain_name,
                        offset=child_ir.offset, size=child_ir.size, is_ydot=True, bc_id=bc_id
                    )
                return MathDt(child_ir, bc_id=bc_id)
            if op == "grad":
                axis = node.get("axis")
                if not axis and current_domain:
                    axes = self.topo.get_axes(current_domain)
                    axis = axes[-1] if axes else current_domain
                return MathGrad(child=self._lower_expr(child, current_state, axis), axis=axis, bc_id=bc_id)
            if op == "div":
                axis = node.get("axis")
                if not axis and current_domain:
                    axes = self.topo.get_axes(current_domain)
                    axis = axes[-1] if axes else current_domain
                return MathDiv(child=self._lower_expr(child, current_state, axis), axis=axis, bc_id=bc_id)
            if op == "coords":
                axis = node.get("axis")
                if not axis and current_domain:
                    axes = self.topo.get_axes(current_domain)
                    axis = axes[-1] if axes else current_domain
                return MathCoords(axis=axis, bc_id=bc_id)
            if op == "integral":
                over_dom = node.get("over")
                return MathIntegral(child=self._lower_expr(child, current_state, over_dom), over_domain=over_dom, bc_id=bc_id)

            return MathUnaryOp(op=op, child=self._lower_expr(child, current_state, current_domain), bc_id=bc_id)

        return MathScalar(0.0)

    def _normalize_equations(self, equations: List[Dict[str, Any]], is_obs: bool = False) -> List[Dict[str, Any]]:
        flat_eqs = []
        for eq_data in equations:
            state_name = eq_data["state"]
            d_bcs = self.semantic_ctx.get_dirichlet_bc(state_name)
            d_name = getattr(self.state_map.get(state_name), "domain", None)
            last_axis = self.topo.get_axes(d_name.name)[-1] if d_name else None

            if eq_data["type"] == "piecewise":
                regions = eq_data["regions"]
                region_divs = {r["domain"]: extract_div_child(r["eq"]) for r in regions}

                for reg in regions:
                    b_axis = self.topo.get_base_axis(reg["domain"])
                    r_start = reg["start_idx"]
                    r_res = reg["end_idx"] - reg["start_idx"]

                    if d_bcs and last_axis and self.topo.get_base_axis(last_axis) == b_axis:
                        domain_start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
                        domain_res = self.topo.domains.get(last_axis, {}).get("resolution", 1)

                        if "left" in d_bcs and r_start == domain_start:
                            r_start += 1
                            r_res -= 1
                        if "right" in d_bcs and r_start + r_res == domain_start + domain_res:
                            r_res -= 1

                    if r_res > 0:
                        flat_eqs.append({
                            "state": state_name,
                            "type": "standard",
                            "eq": reg["eq"],
                            "bounds_override": {b_axis: (r_start, r_res)},
                            "piecewise_info": {
                                "regions": regions,
                                "region_divs": region_divs,
                                "current_region": reg
                            }
                        })
            else:
                bounds_override = {}
                if d_bcs and last_axis:
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

                is_valid = True
                if d_bcs and last_axis and bounds_override.get(self.topo.get_base_axis(last_axis), (0, 1))[1] <= 0:
                    is_valid = False

                if is_valid:
                    eq_out = {
                        "state": state_name,
                        "type": "standard",
                        "eq": eq_data["eq"]
                    }
                    if bounds_override:
                        eq_out["bounds_override"] = bounds_override
                    flat_eqs.append(eq_out)

        return flat_eqs