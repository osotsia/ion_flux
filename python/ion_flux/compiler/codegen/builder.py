from typing import List, Dict, Any
from ion_flux.compiler.passes.semantic import SemanticContext
from ion_flux.compiler.passes.spatial import SpatialLoweringVisitor, IndexManager
from ion_flux.compiler.passes.ir import Loop, Assign, ArrayAccess, BinaryOp, Literal, Var, RawCpp
from .templates import generate_cpp_skeleton
from .topology import TopologyAnalyzer

def emit_assignment(target_state: str, eq_dict: Any, layout, topo, visitor, 
                    bounds_override=None, is_obs=False) -> List[Any]:
    target_domain = getattr(visitor.state_map.get(target_state), "domain", None)
    axes = topo.get_axes(target_domain.name if target_domain else None)
    bounds_override = bounds_override or {}
    
    visitor.current_axis = topo.get_base_axis(axes[-1]) if axes else None
    
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
        rhs_ir = visitor.lower(eq_dict["node"], idx_mgr)
        y_ir = visitor._array_access("y", BinaryOp("+", Literal(offset), flat_idx))
        assign = Assign(res_ir, BinaryOp("-", y_ir, rhs_ir))
    elif not is_obs:
        lhs_ir = visitor.lower(eq_dict["left"], idx_mgr)
        rhs_ir = visitor.lower(eq_dict["right"], idx_mgr)
        
        for ale_ir in visitor.generate_ale_dilution(target_state, idx_mgr):
            rhs_ir = BinaryOp("+", rhs_ir, ale_ir)
            
        assign = Assign(res_ir, BinaryOp("-", lhs_ir, rhs_ir))
    else:
        rhs_ir = visitor.lower(eq_dict, idx_mgr)
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

def extract_loops_and_assign(stmts):
    loops = []
    curr = stmts
    while len(curr) == 1 and isinstance(curr[0], Loop):
        loops.append(curr[0])
        curr = curr[0].body
    assert len(curr) == 1 and isinstance(curr[0], Assign)
    return loops, curr[0]

def generate_cpp(ast_payload: Dict[str, Any], layout: Any, states: List[Any], observables: List[Any], bandwidth: int = 0, target: str = "cpu") -> str:
    topo = TopologyAnalyzer(ast_payload.get("domains", {}))
    ctx = SemanticContext(ast_payload)
    
    visitor = SpatialLoweringVisitor(layout, {s.name: s for s in states}, ctx, topo, target)
    
    eq_stmts = []
    obs_stmts = []
    
    jac_block_funcs = []
    jac_assembly_stmts = []
    block_counter = 0

    dx_stmts = [RawCpp("double dx_default = 1.0;")]
    
    for d_name, d_info in ast_payload.get("domains", {}).items():
        if d_info.get("type") == "composite": continue
        res_val = max(d_info.get("resolution", 2) - 1, 1)
        if d_name in ctx.dynamic_domains:
            idx_mgr = IndexManager(topo)
            idx_mgr.register(topo.get_base_axis(d_name), Literal(0))
            
            rhs_ir = visitor.lower(ctx.dynamic_domains[d_name]["rhs"], idx_mgr)
            dx_stmts.append(RawCpp(f"double dx_{d_name} = std::max(1e-12, (double)({rhs_ir.to_cpp()})) / {res_val}.0;"))
        else:
            bounds = d_info.get("bounds", (0.0, 1.0))
            dx_stmts.append(RawCpp(f"double dx_{d_name} = {float(bounds[1] - bounds[0]) / res_val};"))

    dx_stmts_str = "\n    ".join(s.to_cpp() for s in dx_stmts)

    def process_assignment(target_state, eq_dict, bounds_override=None, is_obs=False):
        nonlocal block_counter
        stmts = emit_assignment(target_state, eq_dict, layout, topo, visitor, bounds_override, is_obs)
        if is_obs:
            obs_stmts.extend(stmts)
            return
            
        eq_stmts.extend(stmts)
        
        loops, assign = extract_loops_and_assign(stmts)
        
        loop_var_defs = [f"int {l.var} = loop_idx[{i}];" for i, l in enumerate(loops)]
        if not loops: loop_var_defs.append("int dummy_idx = loop_idx[0]; (void)dummy_idx;")
        
        jac_block_funcs.append(f"""
static void eval_jac_block_{block_counter}(const double* y, const double* ydot, const double* p, const double* m, const int* loop_idx, double* res) {{
    {' '.join(loop_var_defs)}
    {dx_stmts_str}
    *res = {assign.rhs.to_cpp()};
}}""")
        
        loop_array_init = "{" + ", ".join(l.var for l in loops) + "}" if loops else "{0}"
        
        inner_body = f"""
        int loop_idx[{max(len(loops), 1)}] = {loop_array_init};
        
        double dres = 1.0;
        double res_val = 0.0;
        __enzyme_autodiff((void*)eval_jac_block_{block_counter},
            enzyme_dup, y, dy.data(),
            enzyme_dup, ydot, dydot.data(),
            enzyme_const, p,
            enzyme_const, m,
            enzyme_const, loop_idx,
            enzyme_dupnoneed, &res_val, &dres);
            
        int global_row = {assign.lhs.index.to_cpp()};
        for(int col = 0; col < N; ++col) {{
            if (dy[col] != 0.0 || dydot[col] != 0.0) {{
                double val = dy[col] + c_j * dydot[col];
                if (val != 0.0) {{
                    if (nnz < max_nnz) {{
                        out_rows[nnz] = global_row;
                        out_cols[nnz] = col;
                        out_vals[nnz] = val;
                        nnz++;
                    }}
                }}
                dy[col] = 0.0;
                dydot[col] = 0.0;
            }}
        }}"""
        
        assembly_code = inner_body
        for loop in reversed(loops):
            assembly_code = f"for(int {loop.var} = {loop.start.to_cpp()}; {loop.var} < {loop.end.to_cpp()}; ++{loop.var}) {{\n{assembly_code}\n}}"
            
        jac_assembly_stmts.append(f"{{\n{assembly_code}\n}}")
        block_counter += 1

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
                
                r_start = reg["start_idx"]
                r_res = reg["end_idx"] - reg["start_idx"]
                d_bcs = ctx.get_dirichlet_bc(state_name)
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
                    process_assignment(state_name, reg["eq"], override)
                
            visitor.is_piecewise = False
            visitor.current_region_data = None
        else:
            d_bcs = ctx.get_dirichlet_bc(state_name)
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
                    process_assignment(state_name, eq_data["eq"], override)
            else:
                process_assignment(state_name, eq_data["eq"], override)

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
                process_assignment(state_name, dirichlet_node, override)

    visitor.state_map.update({o.name: o for o in observables})
    for eq_data in ast_payload.get("observables", []):
        obs_name = eq_data["state"]
        if eq_data["type"] == "piecewise":
            for reg in eq_data["regions"]:
                b_axis = topo.get_base_axis(reg["domain"])
                override = {b_axis: (reg["start_idx"], reg["end_idx"] - reg["start_idx"])}
                process_assignment(obs_name, reg["eq"], override, is_obs=True)
        else:
            process_assignment(obs_name, eq_data["eq"], is_obs=True)

    body_str = "\n    ".join(stmt.to_cpp() for stmt in (dx_stmts + eq_stmts))
    obs_body_str = "\n    ".join(stmt.to_cpp() for stmt in (dx_stmts + obs_stmts))
    
    jac_block_funcs_str = "\n".join(jac_block_funcs)
    jac_assembly_body_str = "\n".join(jac_assembly_stmts)
    
    return generate_cpp_skeleton(layout.n_states, layout.n_params, layout.n_obs, body_str, obs_body_str, jac_block_funcs_str, jac_assembly_body_str)