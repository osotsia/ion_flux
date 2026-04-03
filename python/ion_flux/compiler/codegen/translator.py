from typing import Dict, Any, Optional
from . import ir
from .topology import get_stride, get_local_index, get_coord_sys, get_resolution
from .ast_analysis import extract_state_name

class DIRTranslator:
    _SIMPLE_MATH = {"neg": "-", "abs": "std::abs", "exp": "std::exp", "log": "std::log", "sin": "std::sin", "cos": "std::cos"}
    _BIN_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/", "gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="}

    def __init__(self, layout, state_map, neumann_bcs):
        self.layout = layout
        self.state_map = state_map
        self.neumann_bcs = neumann_bcs
        self.current_domain = None
        self.current_axis = None
        self.preamble_stmts = []          # Stores hoisted LICM pre-computations
        self._integral_counter = 0        # Guarantees unique hoisted variable names

    def translate(self, node: Dict[str, Any], idx: ir.Expr, face: Optional[str] = None) -> ir.Expr:
        # Dynamically inject Neumann boundary evaluations natively at faces
        bc_id = node.get("_bc_id")
        if bc_id and face in ("left", "right") and bc_id in self.neumann_bcs:
            bcs = self.neumann_bcs[bc_id]
            if face in bcs:
                bc_val_ir = self.translate(bcs[face], idx, face=None)
                
                # Use cached current_axis to prevent boundary injection from snapping 
                # to the outer macroscopic edges of a flattened macro-micro array.
                target_axis = self.current_axis if self.current_axis else (self.current_domain.name if self.current_domain else "")
                
                res_val = int(get_resolution(self.current_domain, target_axis))
                local_idx = get_local_index(idx.to_cpp(), self.current_domain, target_axis)
                
                cond_val = "0" if face == "left" else str(res_val - 1)
                cond_ir = ir.BinaryOp("==", ir.RawCpp(local_idx), ir.Literal(cond_val))
                base_eval_ir = self._dispatch(node, idx, face)
                
                return ir.Ternary(cond_ir, bc_val_ir, base_eval_ir)

        return self._dispatch(node, idx, face)

    def _dispatch(self, node: Dict[str, Any], idx: ir.Expr, face: Optional[str] = None) -> ir.Expr:
        node_type = node.get("type")
        if node_type == "Scalar": return ir.Literal(node["value"])
        if node_type == "Parameter": return ir.ArrayAccess("p", ir.Literal(self.layout.get_param_offset(node['name'])))
        if node_type == "State": return self._translate_State(node, idx, face)
        if node_type == "Boundary": return self._translate_Boundary(node, idx)
        if node_type == "BinaryOp": return self._translate_BinaryOp(node, idx, face)
        if node_type == "UnaryOp": return self._translate_UnaryOp(node, idx, face)
        raise ValueError(f"Unknown AST node: {node_type}")

    def _safe_offset(self, base_idx: ir.Expr, offset: int, domain: Any, axis: str) -> ir.Expr:
        if offset == 0: return base_idx
        if domain and type(domain).__name__ == "CompositeDomain" and len(domain.domains) == 2:
            d_mac, d_mic = domain.domains
            if axis == d_mic.name:
                cond = ir.BinaryOp("==", ir.BinaryOp("%", base_idx, ir.Literal(d_mic.resolution)), ir.Literal(d_mic.resolution - 1 if offset > 0 else 0))
                return ir.BinaryOp("+", base_idx, ir.Ternary(cond, ir.Literal(0), ir.Literal(offset)))
            elif axis == d_mac.name:
                stride = d_mic.resolution
                macro_idx = ir.BinaryOp("/", base_idx, ir.Literal(stride))
                cond = ir.BinaryOp("==", macro_idx, ir.Literal(d_mac.resolution - 1 if offset > 0 else 0))
                return ir.BinaryOp("+", base_idx, ir.Ternary(cond, ir.Literal(0), ir.Literal(offset)))
        
        return ir.BinaryOp("+", base_idx, ir.Literal(offset))

    def _translate_State(self, node: Dict[str, Any], idx: ir.Expr, face: Optional[str] = None) -> ir.Expr:
        offset, size = self.layout.state_offsets[node["name"]]
        
        # Index translation for nested hierarchies
        state_obj = self.state_map.get(node["name"])
        target_domain = getattr(state_obj, "domain", None)
        eval_idx = idx
        
        if self.current_domain and target_domain and self.current_domain != target_domain:
            if type(self.current_domain).__name__ == "CompositeDomain" and len(self.current_domain.domains) == 2:
                d_mac, d_mic = self.current_domain.domains
                if target_domain.name == d_mac.name:
                    eval_idx = ir.BinaryOp("/", idx, ir.Literal(d_mic.resolution))
                elif target_domain.name == d_mic.name:
                    eval_idx = ir.BinaryOp("%", idx, ir.Literal(d_mic.resolution))
                    
        # States unconditionally self-clamp to enforce internal topology integrity
        clamped_idx = ir.FuncCall("CLAMP", [eval_idx, ir.Literal(size)])
        array_name = "ydot" if getattr(self, "use_ydot", False) else "y"
        base_access = ir.ArrayAccess(array_name, ir.BinaryOp("+", ir.Literal(offset), clamped_idx))
        
        if face == "right":
            axis = self.current_axis if self.current_axis else (self.current_domain.name if self.current_domain else "")
            stride = int(get_stride(self.current_domain, axis)) if self.current_domain else 1
            next_idx = self._safe_offset(eval_idx, stride, self.current_domain, axis)
            next_idx = ir.FuncCall("CLAMP", [next_idx, ir.Literal(size)])
            next_access = ir.ArrayAccess(array_name, ir.BinaryOp("+", ir.Literal(offset), next_idx))
            return ir.BinaryOp("*", ir.Literal(0.5), ir.BinaryOp("+", base_access, next_access))
        if face == "left":
            axis = self.current_axis if self.current_axis else (self.current_domain.name if self.current_domain else "")
            stride = int(get_stride(self.current_domain, axis)) if self.current_domain else 1
            prev_idx = self._safe_offset(eval_idx, -stride, self.current_domain, axis)
            prev_idx = ir.FuncCall("CLAMP", [prev_idx, ir.Literal(size)])
            prev_access = ir.ArrayAccess(array_name, ir.BinaryOp("+", ir.Literal(offset), prev_idx))
            return ir.BinaryOp("*", ir.Literal(0.5), ir.BinaryOp("+", base_access, prev_access))
            
        return base_access

    def _translate_Boundary(self, node: Dict[str, Any], idx: ir.Expr) -> ir.Expr:
        side = node["side"]
        if node["child"].get("type") == "State":
            state_name = node["child"]["name"]
            _, size = self.layout.state_offsets[state_name]
            
            state_obj = self.state_map.get(state_name)
            if state_obj and hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
                d_mac, d_mic = state_obj.domain.domains
                if node.get("domain") == d_mic.name:
                    if self.current_domain and self.current_domain.name == d_mac.name:
                        base = ir.BinaryOp("*", idx, ir.Literal(d_mic.resolution))
                    else:
                        base = ir.BinaryOp("*", ir.BinaryOp("/", idx, ir.Literal(d_mic.resolution)), ir.Literal(d_mic.resolution))
                    b_idx = base if side == "left" else ir.BinaryOp("+", base, ir.Literal(d_mic.resolution - 1))
                    return self.translate(node["child"], b_idx, face=None)

            b_idx = ir.Literal(0) if side == "left" else ir.Literal(size - 1)
            return self.translate(node["child"], b_idx, face=None)
        return self.translate(node["child"], idx, face=None)

    def _translate_BinaryOp(self, node: Dict[str, Any], idx: ir.Expr, face: Optional[str] = None) -> ir.Expr:
        l = self.translate(node["left"], idx, face)
        r = self.translate(node["right"], idx, face)
        op = node["op"]
        
        if op in self._BIN_SYM:
            bop = ir.BinaryOp(self._BIN_SYM[op], l, r)
            if op in ("gt", "lt", "ge", "le", "eq", "ne"): return ir.Ternary(bop, ir.Literal(1.0), ir.Literal(0.0))
            return bop
        if op == "pow": return ir.FuncCall("std::pow", [l, r])
        if op == "max": return ir.FuncCall("std::max", [ir.RawCpp(f"(double)({l.to_cpp()})"), ir.RawCpp(f"(double)({r.to_cpp()})")])
        if op == "min": return ir.FuncCall("std::min", [ir.RawCpp(f"(double)({l.to_cpp()})"), ir.RawCpp(f"(double)({r.to_cpp()})")])
        raise ValueError(f"Unknown BinaryOp: {op}")

    def _translate_UnaryOp(self, node: Dict[str, Any], idx: ir.Expr, face: Optional[str] = None) -> ir.Expr:
        op = node["op"]
        
        # --- Volume Integration (Loop Invariant Code Motion) ---
        if op == "integral":
            target_domain_name = node.get("over")
            target_domain = None
            
            if target_domain_name:
                for s in self.state_map.values():
                    if getattr(s, "domain", None):
                        domains = s.domain.domains if hasattr(s.domain, "domains") else [s.domain]
                        for d in domains:
                            if d.name == target_domain_name:
                                target_domain = d
                                break
                    if target_domain: break
                    
            if not target_domain:
                raise ValueError(f"Integral target domain '{target_domain_name}' not found.")
                
            res = int(target_domain.resolution)
            dx_ir = ir.Var(f"dx_{target_domain_name}")
            
            # Check if this is a nested macro-micro integral (varies per macro node)
            is_nested = self.current_domain and self.current_domain.name != target_domain_name
            
            int_idx_name = f"idx_int_{self._integral_counter}"
            self._integral_counter += 1
            
            if is_nested:
                # Nested Field Integration (IIFE - Optimal O(N) for macro-micro)
                if type(self.current_domain).__name__ == "CompositeDomain":
                    macro_idx = ir.BinaryOp("/", idx, ir.Literal(res))
                    macro_offset = ir.BinaryOp("*", macro_idx, ir.Literal(res))
                else:
                    macro_offset = ir.BinaryOp("*", idx, ir.Literal(res))
                    
                eval_idx = ir.BinaryOp("+", macro_offset, ir.Var(int_idx_name))
                child_expr = self.translate(node["child"], eval_idx, face=None)
                
                # Evaluates immediately within the current macro loop
                cpp_code = (
                    f"[&]() {{\n"
                    f"    double sum = 0.0;\n"
                    f"    for(int {int_idx_name} = 0; {int_idx_name} < {res}; ++{int_idx_name}) {{\n"
                    f"        double w = ({int_idx_name} == 0 || {int_idx_name} == {res - 1}) ? 0.5 : 1.0;\n"
                    f"        sum += {child_expr.to_cpp()} * {dx_ir.to_cpp()} * w;\n"
                    f"    }}\n"
                    f"    return sum;\n"
                    f"}}()"
                )
                return ir.RawCpp(cpp_code)
            else:
                # Global Invariant Integration (LICM Hoisting to prevent O(N^2) redundancy)
                var_name = f"hoisted_int_{self._integral_counter}"
                
                # Temporarily bind the target domain so the child AST nodes map correctly
                prev_domain = self.current_domain
                self.current_domain = target_domain
                child_expr = self.translate(node["child"], ir.Var(int_idx_name), face=None)
                self.current_domain = prev_domain
                
                # Appended to preamble to run EXACTLY ONCE at the top of the function
                cpp_code = (
                    f"double {var_name} = 0.0;\n"
                    f"for(int {int_idx_name} = 0; {int_idx_name} < {res}; ++{int_idx_name}) {{\n"
                    f"    double w = ({int_idx_name} == 0 || {int_idx_name} == {res - 1}) ? 0.5 : 1.0;\n"
                    f"    {var_name} += {child_expr.to_cpp()} * {dx_ir.to_cpp()} * w;\n"
                    f"}}"
                )
                self.preamble_stmts.append(ir.RawCpp(cpp_code))
                
                # Main equation simply references the O(1) static variable
                return ir.Var(var_name)
            
        if op == "dt":
            state_name = node["child"]["name"]
            offset, size = self.layout.state_offsets[state_name]
            clamped = ir.FuncCall("CLAMP", [idx, ir.Literal(size)])
            return ir.ArrayAccess("ydot", ir.BinaryOp("+", ir.Literal(offset), clamped))
            
        if op == "coords":
            dx_ir = ir.Var("dx_default")
            if self.current_domain: dx_ir = ir.Var(f"dx_{self.current_domain.name}")
            base = ir.BinaryOp("*", idx, dx_ir)
            if face == "right": return ir.BinaryOp("+", base, ir.BinaryOp("*", ir.Literal(0.5), dx_ir))
            if face == "left": return ir.BinaryOp("-", base, ir.BinaryOp("*", ir.Literal(0.5), dx_ir))
            return base
            
        if op == "grad":
            axis = node.get("axis")
            if not axis and self.current_domain:
                axis = self.current_axis if self.current_axis else getattr(self.current_domain, "name", None)
            dx_ir = ir.Var(f"dx_{axis}" if axis else "dx_default")
            stride = int(get_stride(self.current_domain, axis)) if self.current_domain else 1
            
            if face == "right":
                right_idx = self._safe_offset(idx, stride, self.current_domain, axis)
                right = self.translate(node["child"], right_idx, face=None)
                curr = self.translate(node["child"], idx, face=None)
                return ir.BinaryOp("/", ir.BinaryOp("-", right, curr), dx_ir)
            elif face == "left":
                curr = self.translate(node["child"], idx, face=None)
                left_idx = self._safe_offset(idx, -stride, self.current_domain, axis)
                left = self.translate(node["child"], left_idx, face=None)
                return ir.BinaryOp("/", ir.BinaryOp("-", curr, left), dx_ir)
            else:
                right_idx = self._safe_offset(idx, stride, self.current_domain, axis)
                left_idx = self._safe_offset(idx, -stride, self.current_domain, axis)
                right = self.translate(node["child"], right_idx, face=None)
                left = self.translate(node["child"], left_idx, face=None)
                return ir.BinaryOp("/", ir.BinaryOp("-", right, left), ir.BinaryOp("*", ir.Literal(2.0), dx_ir))
                
        if op == "div":
            axis = node.get("axis")
            if not axis and self.current_domain:
                axis = self.current_axis if self.current_axis else getattr(self.current_domain, "name", None)
            dx_ir = ir.Var(f"dx_{axis}" if axis else "dx_default")
            
            coord_sys = get_coord_sys(self.current_domain, axis)
            
            if coord_sys == "unstructured":
                mesh_name = self.current_domain.name
                offsets = self.layout.mesh_offsets[mesh_name]
                rp = offsets["row_ptr"]
                ci = offsets["col_ind"]
                w = offsets["weights"]
                
                state_name = extract_state_name(node["child"])
                s_off, _ = self.layout.state_offsets[state_name]
                
                cpp_code = (
                    f"[&]() {{\n"
                    f"    double sum = 0.0;\n"
                    f"    int start = (int)m[{rp} + (int)({idx.to_cpp()})];\n"
                    f"    int end = (int)m[{rp} + (int)({idx.to_cpp()}) + 1];\n"
                    f"    for(int k = start; k < end; ++k) {{\n"
                    f"        int c_idx = (int)m[{ci} + k];\n"
                    f"        sum += m[{w} + k] * y[{s_off} + c_idx];\n"
                    f"    }}\n"
                    f"    return sum;\n"
                    f"}}()"
                )
                
                bulk_div = ir.RawCpp(cpp_code)
                
                flux_bc_id = node["child"].get("_bc_id")
                bc_terms = []
                if flux_bc_id and flux_bc_id in self.neumann_bcs:
                    bcs = self.neumann_bcs[flux_bc_id]
                    for surf_tag, bc_node in bcs.items():
                        if surf_tag in offsets.get("surfaces", {}):
                            surf_off = offsets["surfaces"][surf_tag]
                            bc_val = self.translate(bc_node, idx, face=None).to_cpp()
                            mask_val = f"m[{surf_off} + (int)({idx.to_cpp()})]"
                            bc_terms.append(f"({bc_val}) * {mask_val}")
                            
                if bc_terms:
                    bc_expr = " + ".join(bc_terms)
                    # Subtract boundary mass flux out to conserve integration orientation
                    return ir.RawCpp(f"({bulk_div.to_cpp()} - ({bc_expr}))")
                
                return bulk_div

            prev_axis = self.current_axis
            self.current_axis = axis
            
            right = self.translate(node["child"], idx, face="right")
            left = self.translate(node["child"], idx, face="left")
            
            # Spherical terms evaluate directly at the node center to prevent 
            # severe geometrical interpolation failures near the origin (r=0).
            center = self.translate(node["child"], idx, face=None)
            
            self.current_axis = prev_axis
            
            std_div = ir.BinaryOp("/", ir.BinaryOp("-", right, left), dx_ir)
            
            # Apply finite volume half-cell mass conservation correction at boundaries
            if self.current_domain:
                axis_to_use = axis if axis else self.current_domain.name
                res_val = get_resolution(self.current_domain, axis_to_use)
                local_idx = get_local_index(idx.to_cpp(), self.current_domain, axis_to_use)
                
                cond_left = ir.BinaryOp("==", ir.RawCpp(local_idx), ir.Literal("0"))
                cond_right = ir.BinaryOp("==", ir.RawCpp(local_idx), ir.Literal(f"{res_val} - 1"))
                is_boundary = ir.BinaryOp("||", cond_left, cond_right)
                
                std_div = ir.Ternary(is_boundary, ir.BinaryOp("*", ir.Literal(2.0), std_div), std_div)
            
            if coord_sys == "spherical":
                local_idx = get_local_index(idx.to_cpp(), self.current_domain, axis)
                r_coord = f"(std::max(1e-12, (double)({local_idx}) * {dx_ir.to_cpp()}))"
                
                # Evaluate the cell-center flux safely by averaging the face fluxes to prevent mass leaks!
                center = ir.BinaryOp("*", ir.Literal(0.5), ir.BinaryOp("+", right, left))
                
                spherical_term = ir.BinaryOp("*", ir.RawCpp(f"(2.0 / {r_coord})"), center)
                combined = ir.BinaryOp("+", std_div, spherical_term)
                
                return ir.Ternary(ir.BinaryOp("==", ir.RawCpp(local_idx), ir.Literal("0")), 
                                  ir.BinaryOp("/", ir.BinaryOp("*", ir.Literal(6.0), right), dx_ir), 
                                  combined)
            return std_div

        if op in self._SIMPLE_MATH:
            child_expr = self.translate(node["child"], idx, face)
            func = self._SIMPLE_MATH[op]
            return ir.RawCpp(f"(-{child_expr.to_cpp()})") if op == "neg" else ir.FuncCall(func, [child_expr])

        raise ValueError(f"Unknown UnaryOp: {op}")