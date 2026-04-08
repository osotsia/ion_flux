from typing import List, Dict, Any
from ion_flux.compiler.passes.semantic import SemanticContext
from ion_flux.compiler.passes.spatial import SpatialLoweringVisitor
from ion_flux.compiler.passes.ir import Loop, Assign, ArrayAccess, BinaryOp, Literal, Var, RawCpp
from .templates import generate_cpp_skeleton

def generate_cpp(ast_payload: Dict[str, Any], layout: Any, states: List[Any], bandwidth: int = 0, target: str = "cpu") -> str:
    state_map = {s.name: s for s in states}
    
    # --- Pass 1: Semantic Resolution ---
    ctx = SemanticContext(ast_payload)
    
    # --- Pass 2: Spatial Lowering ---
    visitor = SpatialLoweringVisitor(layout, state_map, ctx)
    
    eq_stmts = []
    dx_stmts = []
    
    for d_name, d_info in ast_payload.get("domains", {}).items():
        res_val = max(d_info["resolution"] - 1, 1)
        if d_name in ctx.dynamic_domains:
            rhs_ir = visitor.lower(ctx.dynamic_domains[d_name]["rhs"], Literal(0))
            dx_stmts.append(RawCpp(f"double dx_{d_name} = std::max(1e-12, (double)({rhs_ir.to_cpp()})) / {res_val}.0;"))
        else:
            dx_val = float(d_info["bounds"][1] - d_info["bounds"][0]) / res_val
            dx_stmts.append(RawCpp(f"double dx_{d_name} = {dx_val};"))
    dx_stmts.append(RawCpp("double dx_default = 1.0;"))
    
    for eq_data in ast_payload.get("equations", []):
        state_name = eq_data["state"]
        offset, size = layout.state_offsets[state_name]
        visitor.current_domain = getattr(state_map[state_name], "domain", None)
        
        if eq_data["type"] == "piecewise":
            visitor.is_piecewise = True
            from ion_flux.compiler.codegen.ast_analysis import extract_div_child
            region_divs = {}
            for reg in eq_data["regions"]:
                region_divs[reg["domain"]] = extract_div_child(reg["eq"])
                
            visitor.piecewise_regions = eq_data["regions"]
            visitor.region_divs = region_divs
            
            for reg in eq_data["regions"]:
                start = reg["start_idx"]
                end = reg["end_idx"]
                visitor.current_region_data = reg
                
                lhs_ir = visitor.lower(reg["eq"]["left"], Var("i"))
                rhs_ir = visitor.lower(reg["eq"]["right"], Var("i"))
                
                for ale_ir in visitor.generate_ale_dilution(state_name, offset, size, "i"):
                    rhs_ir = BinaryOp("+", rhs_ir, ale_ir)
                    
                # Check if this region shares a node with a previous region
                has_prev_overlap = False
                if start > 0:
                    for r in visitor.piecewise_regions:
                        if r["end_idx"] - 1 == start:
                            has_prev_overlap = True
                            break
                            
                if has_prev_overlap:
                    # Conditionally average the equations at the shared interface node
                    res_access = f"res[{offset} + i]"
                    body_cpp = (
                        f"if (i == {start}) {{ "
                        f"{res_access} = 0.5 * {res_access} + 0.5 * ({lhs_ir.to_cpp()} - ({rhs_ir.to_cpp()})); "
                        f"}} else {{ "
                        f"{res_access} = {lhs_ir.to_cpp()} - ({rhs_ir.to_cpp()}); "
                        f"}}"
                    )
                    assign = RawCpp(body_cpp)
                else:
                    # Standard isolated assignment
                    res_ir = ArrayAccess("res", BinaryOp("+", Literal(offset), Var("i")))
                    assign = Assign(res_ir, BinaryOp("-", lhs_ir, rhs_ir))
                
                eq_stmts.append(Loop("i", Literal(start), Literal(end), [assign]))
                
            visitor.piecewise_regions = None
            visitor.region_divs = None
            visitor.current_region_data = None

        elif eq_data["type"] == "standard":
            visitor.is_piecewise = False
            lhs_ir = visitor.lower(eq_data["eq"]["left"], Var("i"))
            rhs_ir = visitor.lower(eq_data["eq"]["right"], Var("i"))
            
            for ale_ir in visitor.generate_ale_dilution(state_name, offset, size, "i"):
                rhs_ir = BinaryOp("+", rhs_ir, ale_ir)
                
            res_ir = ArrayAccess("res", BinaryOp("+", Literal(offset), Var("i")))
            assign = Assign(res_ir, BinaryOp("-", lhs_ir, rhs_ir))
            
            pragma = "#pragma omp parallel for" if ("omp" in target and size > 50) else ""
            eq_stmts.append(Loop("i", Literal(0), Literal(size), [assign], pragma=pragma))

    for bc_data in ast_payload.get("boundaries", []):
        if bc_data["type"] == "dirichlet":
            state_name = bc_data["state"]
            offset, size = layout.state_offsets[state_name]
            visitor.current_domain = getattr(state_map[state_name], "domain", None)
            
            for side, val_dict in bc_data["bcs"].items():
                idx = 0 if side == "left" else size - 1
                val_ir = visitor.lower(val_dict, Literal(idx))
                res_ir = ArrayAccess("res", BinaryOp("+", Literal(offset), Literal(idx)))
                y_ir = ArrayAccess("y", BinaryOp("+", Literal(offset), Literal(idx)))
                
                eq_stmts.append(Assign(res_ir, BinaryOp("-", y_ir, val_ir)))

    # --- Pass 3: C++ Emission ---
    ir_stmts = dx_stmts + eq_stmts
    body_str = "\n    ".join(stmt.to_cpp() for stmt in ir_stmts)
    
    return generate_cpp_skeleton(layout.n_states, layout.n_params, body_str, bandwidth)