from typing import List, Dict, Any
from ion_flux.compiler.passes.semantic import SemanticContext
from ion_flux.compiler.passes.spatial import SpatialLoweringVisitor
from ion_flux.compiler.passes.ir import Loop, Assign, ArrayAccess, BinaryOp, Literal, Var, RawCpp
from .templates import generate_cpp_skeleton
from .topology import TopologyAnalyzer

def emit_assignment(target_state: str, eq_dict: Any, layout, topo, visitor, 
                    bounds_override=None, is_obs=False) -> List[Any]:
    """Generates an N-dimensional nested loop assigning an AST node to a target array."""
    target_domain = getattr(visitor.state_map.get(target_state), "domain", None)
    axes = topo.get_axes(target_domain.name if target_domain else None)
    bounds_override = bounds_override or {}
    
    # Initialize default axis context for Neumann boundary interception
    visitor.current_axis = topo.get_base_axis(axes[-1]) if axes else None
    
    # 1. Build the active Index Environment
    env = {}
    for axis in axes:
        base = topo.get_base_axis(axis)
        start = topo.domains.get(axis, {}).get("start_idx", 0)
        res = topo.domains.get(axis, {}).get("resolution", 1)
        
        loop_start, _ = bounds_override.get(base, (start, res))
        loop_var = f"idx_{axis}"
        # The environment stores the ABSOLUTE topological index
        env[base] = BinaryOp("+", Var(loop_var), Literal(loop_start))
    
    offset = layout.obs_offsets[target_state][0] if is_obs else layout.state_offsets[target_state][0]
    array_name = "obs" if is_obs else "res"
    flat_idx = visitor._flat_index(target_domain.name if target_domain else None, env)
    res_ir = ArrayAccess(array_name, BinaryOp("+", Literal(offset), flat_idx))
    
    # 2. Lower the AST using the active environment
    if isinstance(eq_dict, dict) and eq_dict.get("type") == "dirichlet_bnd":
        rhs_ir = visitor.lower(eq_dict["node"], env)
        y_ir = ArrayAccess("y", BinaryOp("+", Literal(offset), flat_idx))
        assign = Assign(res_ir, BinaryOp("-", y_ir, rhs_ir))
    elif not is_obs:
        # Evaluate full PDE/DAE (lhs - rhs)
        lhs_ir = visitor.lower(eq_dict["left"], env)
        rhs_ir = visitor.lower(eq_dict["right"], env)
        
        for ale_ir in visitor.generate_ale_dilution(target_state, env):
            rhs_ir = BinaryOp("+", rhs_ir, ale_ir)
            
        assign = Assign(res_ir, BinaryOp("-", lhs_ir, rhs_ir))
    else:
        rhs_ir = visitor.lower(eq_dict, env)
        assign = Assign(res_ir, rhs_ir)
            
    # 3. Wrap assignment in N-Dimensional nested loops
    curr_body = [assign]
    for i, axis in reversed(list(enumerate(axes))):
        base = topo.get_base_axis(axis)
        res = topo.domains.get(axis, {}).get("resolution", 1)
        
        _, loop_res = bounds_override.get(base, (0, res))
        loop_var = f"idx_{axis}"
        
        pragma = "#pragma omp parallel for" if i == 0 and loop_res > 50 and "omp" in visitor.target else ""
        curr_body = [Loop(loop_var, Literal(0), Literal(loop_res), curr_body, pragma)]
        
    return curr_body

def generate_cpp(ast_payload: Dict[str, Any], layout: Any, states: List[Any], observables: List[Any], bandwidth: int = 0, target: str = "cpu") -> str:
    topo = TopologyAnalyzer(ast_payload.get("domains", {}))
    ctx = SemanticContext(ast_payload)
    visitor = SpatialLoweringVisitor(layout, {s.name: s for s in states}, ctx, topo, target)
    
    eq_stmts = []
    dx_stmts = [RawCpp("double dx_default = 1.0;")]
    
    # Emit dx spatial strides
    for d_name, d_info in ast_payload.get("domains", {}).items():
        if d_info.get("type") == "composite": continue
        res_val = max(d_info.get("resolution", 2) - 1, 1)
        if d_name in ctx.dynamic_domains:
            rhs_ir = visitor.lower(ctx.dynamic_domains[d_name]["rhs"], {topo.get_base_axis(d_name): Literal(0)})
            dx_stmts.append(RawCpp(f"double dx_{d_name} = std::max(1e-12, (double)({rhs_ir.to_cpp()})) / {res_val}.0;"))
        else:
            bounds = d_info.get("bounds", (0.0, 1.0))
            dx_stmts.append(RawCpp(f"double dx_{d_name} = {float(bounds[1] - bounds[0]) / res_val};"))

    # Process Equations
    from ion_flux.compiler.codegen.ast_analysis import extract_div_child
    for eq_data in ast_payload.get("equations", []):
        state_name = eq_data["state"]
        
        if eq_data["type"] == "piecewise":
            visitor.is_piecewise = True
            visitor.piecewise_regions = eq_data["regions"]
            visitor.region_divs = {r["domain"]: extract_div_child(r["eq"]) for r in eq_data["regions"]}
            
            for reg in eq_data["regions"]:
                visitor.current_region_data = reg
                b_axis = topo.get_base_axis(reg["domain"])
                # Limit the specific axis loop to the region's bounds
                override = {b_axis: (reg["start_idx"], reg["end_idx"] - reg["start_idx"])}
                eq_stmts.extend(emit_assignment(state_name, reg["eq"], layout, topo, visitor, override))
                
            visitor.is_piecewise = False
            visitor.current_region_data = None
        else:
            eq_stmts.extend(emit_assignment(state_name, eq_data["eq"], layout, topo, visitor))

    # Process Dirichlet Boundaries
    for bc_data in ast_payload.get("boundaries", []):
        if bc_data["type"] == "dirichlet":
            state_name = bc_data["state"]
            d_name = getattr(visitor.state_map[state_name], "domain", None)
            base_axis = topo.get_base_axis(topo.get_axes(d_name.name if d_name else None)[0]) if d_name else None
            res = topo.domains.get(base_axis, {}).get("resolution", 1)
            
            for side, val_dict in bc_data["bcs"].items():
                dirichlet_node = {"type": "dirichlet_bnd", "node": val_dict}
                idx = 0 if side == "left" else res - 1
                override = {base_axis: (idx, 1)} if base_axis else {}
                eq_stmts.extend(emit_assignment(state_name, dirichlet_node, layout, topo, visitor, override))

    # Process Observables
    obs_stmts = []
    visitor.state_map.update({o.name: o for o in observables})
    for eq_data in ast_payload.get("observables", []):
        obs_name = eq_data["state"]
        if eq_data["type"] == "piecewise":
            for reg in eq_data["regions"]:
                b_axis = topo.get_base_axis(reg["domain"])
                override = {b_axis: (reg["start_idx"], reg["end_idx"] - reg["start_idx"])}
                obs_stmts.extend(emit_assignment(obs_name, reg["eq"], layout, topo, visitor, override, is_obs=True))
        else:
            obs_stmts.extend(emit_assignment(obs_name, eq_data["eq"], layout, topo, visitor, is_obs=True))

    body_str = "\n    ".join(stmt.to_cpp() for stmt in (dx_stmts + eq_stmts))
    obs_body_str = "\n    ".join(stmt.to_cpp() for stmt in (dx_stmts + obs_stmts))
    
    return generate_cpp_skeleton(layout.n_states, layout.n_params, layout.n_obs, body_str, obs_body_str, bandwidth)