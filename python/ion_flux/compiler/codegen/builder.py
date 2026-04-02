from typing import List, Dict, Any
from . import ir
from .translator import DIRTranslator
from .templates import generate_cpp_skeleton

def generate_cpp(ast_payload: Dict[str, Any], layout: Any, states: List[Any], bandwidth: int = 0, target: str = "cpu") -> str:
    state_map = {s.name: s for s in states}
    
    neumann_bcs = {bc["node_id"]: bc["bcs"] for bc in ast_payload.get("boundaries", []) if bc["type"] == "neumann"}
    translator = DIRTranslator(layout, state_map, neumann_bcs)
    
    dx_stmts = []
    
    # 1. Emit Constants dynamically from all utilized physical domains
    for d_name, d_info in ast_payload.get("domains", {}).items():
        dx_val = float(d_info["bounds"][1] - d_info["bounds"][0]) / max(d_info["resolution"] - 1, 1)
        dx_stmts.append(ir.RawCpp(f"double dx_{d_name} = {dx_val};"))
    dx_stmts.append(ir.RawCpp("double dx_default = 1.0;"))
    
    all_domains = {d.name: d for s in states if s.domain for d in (s.domain.domains if hasattr(s.domain, "domains") else [s.domain])}
    
    eq_stmts = []
    # 2. Build Explicit Equation Loops
    for eq_data in ast_payload.get("equations", []):
        state_name = eq_data["state"]
        offset, size = layout.state_offsets[state_name]
        state_obj = state_map[state_name]
        
        if eq_data["type"] == "piecewise":
            for reg in eq_data["regions"]:
                start = reg["start_idx"]
                end = reg["end_idx"]
                translator.current_domain = next((d for d in all_domains.values() if d.name == reg["domain"]), None)
                
                # Equation is a BinaryOp. Extract "left" and "right" instead of legacy "lhs"/"rhs".
                lhs_ir = translator.translate(reg["eq"]["left"], ir.Var("i"))
                rhs_ir = translator.translate(reg["eq"]["right"], ir.Var("i"))
                
                # Natively handles Mass Matrix formulation: Res = LHS - RHS
                res_ir = ir.ArrayAccess("res", ir.BinaryOp("+", ir.Literal(offset), ir.Var("i")))
                assign = ir.Assign(res_ir, ir.BinaryOp("-", lhs_ir, rhs_ir))
                eq_stmts.append(ir.Loop("i", ir.Literal(start), ir.Literal(end), [assign]))
        else:
            translator.current_domain = getattr(state_obj, "domain", None)
            if size == 1:
                lhs_ir = translator.translate(eq_data["eq"]["left"], ir.Literal(0))
                rhs_ir = translator.translate(eq_data["eq"]["right"], ir.Literal(0))
                res_ir = ir.ArrayAccess("res", ir.Literal(offset))
                eq_stmts.append(ir.Assign(res_ir, ir.BinaryOp("-", lhs_ir, rhs_ir)))
            else:
                lhs_ir = translator.translate(eq_data["eq"]["left"], ir.Var("i"))
                rhs_ir = translator.translate(eq_data["eq"]["right"], ir.Var("i"))
                res_ir = ir.ArrayAccess("res", ir.BinaryOp("+", ir.Literal(offset), ir.Var("i")))
                pragma = "#pragma omp parallel for" if ("omp" in target and size > 50) else ""
                assign = ir.Assign(res_ir, ir.BinaryOp("-", lhs_ir, rhs_ir))
                eq_stmts.append(ir.Loop("i", ir.Literal(0), ir.Literal(size), [assign], pragma=pragma))

    # 3. Apply Explicit Dirichlet Boundary Overrides
    for bc_data in ast_payload.get("boundaries", []):
        if bc_data["type"] == "dirichlet":
            state_name = bc_data["state"]
            offset, size = layout.state_offsets[state_name]
            translator.current_domain = getattr(state_map[state_name], "domain", None)
            
            for side, val_dict in bc_data["bcs"].items():
                idx = 0 if side == "left" else size - 1
                val_ir = translator.translate(val_dict, ir.Literal(idx))
                res_ir = ir.ArrayAccess("res", ir.BinaryOp("+", ir.Literal(offset), ir.Literal(idx)))
                y_ir = ir.ArrayAccess("y", ir.BinaryOp("+", ir.Literal(offset), ir.Literal(idx)))
                
                # Natively overwrites the PDE bulk evaluation cleanly at the boundary node
                eq_stmts.append(ir.Assign(res_ir, ir.BinaryOp("-", y_ir, val_ir)))

    # 4. Assemble Final Block (Constants -> LICM Hoisted Preamble -> Equations)
    ir_stmts = dx_stmts + translator.preamble_stmts + eq_stmts

    body_str = "\n    ".join(stmt.to_cpp() for stmt in ir_stmts)
    return generate_cpp_skeleton(layout.n_states, layout.n_params, body_str, bandwidth)