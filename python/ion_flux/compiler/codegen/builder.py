from typing import List, Dict, Any
from ion_flux.compiler.passes.semantic import SemanticContext
from ion_flux.compiler.passes.spatial import SpatialLoweringVisitor, IndexManager
from ion_flux.compiler.passes.ir import Loop, Assign, ArrayAccess, BinaryOp, Literal, Var, RawCpp
from ion_flux.compiler.passes.context import SpatialContext
from .templates import generate_cpp_skeleton
from .topology import TopologyAnalyzer

def emit_assignment(target_state: str, eq_dict: Any, layout, topo, visitor, ctx: SpatialContext,
                    bounds_override=None, is_obs=False) -> List[Any]:
    target_domain = getattr(visitor.state_map.get(target_state), "domain", None)
    axes = topo.get_axes(target_domain.name if target_domain else None)
    bounds_override = bounds_override or {}
    
    base_axis = topo.get_base_axis(axes[-1]) if axes else None
    eq_ctx = ctx.with_updates(axis=base_axis)
    
    idx_mgr = IndexManager(topo)
    for axis in axes:
        base = topo.get_base_axis(axis)
        start = topo.domains.get(axis, {}).get("start_idx", 0)
        res = topo.domains.get(axis, {}).get("resolution", 1)
        
        loop_start, _ = bounds_override.get(base, (start, res))
        loop_var = f"idx_{axis}"
        idx_mgr.register(base, BinaryOp("+", Var(loop_var), Literal(loop_start)))
    
    offset = layout.obs_offsets[target_state][0] if is_obs else layout.state_offsets[target_state][0]
    array_name = "obs" if is_obs else "res"
    flat_idx = idx_mgr.get_flat_index(target_domain.name if target_domain else None)
    res_ir = ArrayAccess(array_name, BinaryOp("+", Literal(offset), flat_idx))
    
    if isinstance(eq_dict, dict) and eq_dict.get("type") == "dirichlet_bnd":
        rhs_ir = visitor.lower(eq_dict["node"], idx_mgr, eq_ctx)
        y_ir = visitor._array_access("y", BinaryOp("+", Literal(offset), flat_idx))
        assign = Assign(res_ir, BinaryOp("-", y_ir, rhs_ir))
    elif not is_obs:
        lhs_ir = visitor.lower(eq_dict["left"], idx_mgr, eq_ctx)
        rhs_ir = visitor.lower(eq_dict["right"], idx_mgr, eq_ctx)
        
        for ale_ir in visitor.generate_ale_dilution(target_state, idx_mgr, eq_ctx):
            rhs_ir = BinaryOp("+", rhs_ir, ale_ir)
            
        assign = Assign(res_ir, BinaryOp("-", lhs_ir, rhs_ir))
    else:
        rhs_ir = visitor.lower(eq_dict, idx_mgr, eq_ctx)
        assign = Assign(res_ir, rhs_ir)
            
    curr_body = [assign]
    for i, axis in reversed(list(enumerate(axes))):
        base = topo.get_base_axis(axis)
        res = topo.domains.get(axis, {}).get("resolution", 1)
        
        _, loop_res = bounds_override.get(base, (0, res))
        loop_var = f"idx_{axis}"
        
        pragma = "#pragma omp parallel for" if i == 0 and loop_res > 50 and "omp" in visitor.target else ""
        curr_body = [Loop(loop_var, Literal(0), Literal(loop_res), curr_body, pragma)]
        
    return curr_body

def generate_cpp(ast_payload: Dict[str, Any], layout: Any, states: List[Any], observables: List[Any], target: str = "cpu") -> str:
    topo = TopologyAnalyzer(ast_payload.get("domains", {}))
    semantic_ctx = SemanticContext(ast_payload)
    
    visitor = SpatialLoweringVisitor(layout, {s.name: s for s in states}, semantic_ctx, topo, target)
    base_ctx = SpatialContext()
    
    eq_stmts = []
    obs_stmts = []

    l_phys_stmts = [RawCpp("double L_phys_default = 1.0;")]
    
    for d_name, d_info in ast_payload.get("domains", {}).items():
        if d_info.get("type") == "composite": continue
        if d_name in semantic_ctx.dynamic_domains:
            idx_mgr = IndexManager(topo)
            idx_mgr.register(topo.get_base_axis(d_name), Literal(0))
            
            rhs_ir = visitor.lower(semantic_ctx.dynamic_domains[d_name]["rhs"], idx_mgr, base_ctx)
            l_phys_stmts.append(RawCpp(f"double L_phys_{d_name} = std::max(1e-12, (double)({rhs_ir.to_cpp()}));"))
        else:
            bounds = d_info.get("bounds", (0.0, 1.0))
            l_phys_stmts.append(RawCpp(f"double L_phys_{d_name} = {float(bounds[1] - bounds[0])};"))

    def process_assignment(target_state, eq_dict, proc_ctx, bounds_override=None, is_obs=False):
        stmts = emit_assignment(target_state, eq_dict, layout, topo, visitor, proc_ctx, bounds_override, is_obs)
        if is_obs:
            obs_stmts.extend(stmts)
        else:
            eq_stmts.extend(stmts)

    from ion_flux.compiler.codegen.ast_analysis import extract_div_child
    for eq_data in ast_payload.get("equations", []):
        state_name = eq_data["state"]
        
        if eq_data["type"] == "piecewise":
            pw_ctx = SpatialContext(
                is_piecewise=True,
                piecewise_regions=eq_data["regions"],
                region_divs={r["domain"]: extract_div_child(r["eq"]) for r in eq_data["regions"]}
            )
            
            for reg in eq_data["regions"]:
                reg_ctx = pw_ctx.with_updates(current_region_data=reg)
                b_axis = topo.get_base_axis(reg["domain"])
                
                r_start = reg["start_idx"]
                r_res = reg["end_idx"] - reg["start_idx"]
                d_bcs = semantic_ctx.get_dirichlet_bc(state_name)
                d_name = getattr(visitor.state_map[state_name], "domain", None)
                last_axis = topo.get_axes(d_name.name)[-1] if d_name else None
                
                if d_bcs and last_axis and topo.get_base_axis(last_axis) == b_axis:
                    domain_start = topo.domains.get(last_axis, {}).get("start_idx", 0)
                    domain_res = topo.domains.get(last_axis, {}).get("resolution", 1)
                    
                    if "left" in d_bcs and r_start == domain_start:
                        r_start += 1
                        r_res -= 1
                    if "right" in d_bcs and r_start + r_res == domain_start + domain_res:
                        r_res -= 1
                        
                if r_res > 0:
                    override = {b_axis: (r_start, r_res)}
                    process_assignment(state_name, reg["eq"], reg_ctx, override)
        else:
            d_bcs = semantic_ctx.get_dirichlet_bc(state_name)
            d_name = getattr(visitor.state_map[state_name], "domain", None)
            last_axis = topo.get_axes(d_name.name)[-1] if d_name else None
            
            override = {}
            if d_bcs and last_axis:
                start = topo.domains.get(last_axis, {}).get("start_idx", 0)
                res = topo.domains.get(last_axis, {}).get("resolution", 1)
                if "left" in d_bcs:
                    start += 1
                    res -= 1
                if "right" in d_bcs:
                    res -= 1
                if res > 0:
                    override[topo.get_base_axis(last_axis)] = (start, res)
                    process_assignment(state_name, eq_data["eq"], SpatialContext(), override)
            else:
                process_assignment(state_name, eq_data["eq"], SpatialContext(), override)

    for bc_data in ast_payload.get("boundaries", []):
        if bc_data["type"] == "dirichlet":
            state_name = bc_data["state"]
            d_name = getattr(visitor.state_map[state_name], "domain", None)
            last_axis = topo.get_axes(d_name.name)[-1] if d_name else None
            base_axis = topo.get_base_axis(last_axis) if last_axis else None
            
            if last_axis:
                res = topo.domains.get(last_axis, {}).get("resolution", 1)
                start = topo.domains.get(last_axis, {}).get("start_idx", 0)
            else:
                res = 1
                start = 0
            
            for side, val_dict in bc_data["bcs"].items():
                dirichlet_node = {"type": "dirichlet_bnd", "node": val_dict}
                idx = start if side == "left" else start + res - 1
                override = {base_axis: (idx, 1)} if base_axis else {}
                process_assignment(state_name, dirichlet_node, SpatialContext(), override)

    visitor.state_map.update({o.name: o for o in observables})
    for eq_data in ast_payload.get("observables", []):
        obs_name = eq_data["state"]
        if eq_data["type"] == "piecewise":
            pw_ctx = SpatialContext(
                is_piecewise=True,
                piecewise_regions=eq_data["regions"],
                region_divs={r["domain"]: extract_div_child(r["eq"]) for r in eq_data["regions"]}
            )
            for reg in eq_data["regions"]:
                reg_ctx = pw_ctx.with_updates(current_region_data=reg)
                b_axis = topo.get_base_axis(reg["domain"])
                override = {b_axis: (reg["start_idx"], reg["end_idx"] - reg["start_idx"])}
                process_assignment(obs_name, reg["eq"], reg_ctx, override, is_obs=True)
        else:
            process_assignment(obs_name, eq_data["eq"], SpatialContext(), is_obs=True)

    body_str = "\n    ".join(stmt.to_cpp() for stmt in (l_phys_stmts + eq_stmts))
    obs_body_str = "\n    ".join(stmt.to_cpp() for stmt in (l_phys_stmts + obs_stmts))
    
    return generate_cpp_skeleton(layout.n_states, layout.n_params, layout.n_obs, body_str, obs_body_str)