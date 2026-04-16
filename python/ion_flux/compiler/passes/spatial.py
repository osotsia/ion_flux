from typing import Dict, Any, Optional, List
from .ir import Expr, Literal, Var, ArrayAccess, BinaryOp, FuncCall, Ternary, RawCpp
from .semantic import SemanticContext

class SpatialLoweringVisitor:
    _BIN_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "std::pow",
                "gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="}
    _UNARY_SYM = {"neg": "-", "abs": "std::abs", "exp": "std::exp", "log": "std::log", 
                  "sin": "std::sin", "cos": "std::cos"}

    def __init__(self, layout, state_map, semantic_ctx, topo, target):
        self.layout = layout
        self.state_map = state_map
        self.ctx = semantic_ctx
        self.topo = topo
        self.target = target
        self.current_axis = None
        self.use_ydot = False

    def _flat_index(self, domain_name: Optional[str], env: Dict[str, Expr]) -> Expr:
        """Converts an N-Dimensional environment into a flat C-array memory offset."""
        if not domain_name: return Literal(0)
        axes = self.topo.get_axes(domain_name)
        strides = self.topo.get_strides(domain_name)
        
        terms = []
        for axis in axes:
            base = self.topo.get_base_axis(axis)
            abs_idx = env.get(base, Literal(0))
            
            start_idx = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            
            local_idx = BinaryOp("-", abs_idx, Literal(start_idx))
            clamped = FuncCall("CLAMP", [local_idx, Literal(res)])
            
            stride = strides[axis]
            if stride > 1: terms.append(BinaryOp("*", clamped, Literal(stride)))
            else: terms.append(clamped)
            
        if not terms: return Literal(0)
        flat = terms[0]
        for t in terms[1:]: flat = BinaryOp("+", flat, t)
        return flat

    def lower(self, node: Dict[str, Any], env: Dict[str, Expr], face: Optional[str] = None) -> Expr:
        bc_info = self.ctx.get_neumann_bc(node.get("_bc_id"), face)
        if bc_info:
            bc_ir = self.lower(bc_info["ast"], env, face=None)
            
            b_axis = self.topo.get_base_axis(self.current_axis)
            res = self.topo.domains.get(b_axis, {}).get("resolution", 1)
            start = self.topo.domains.get(b_axis, {}).get("start_idx", 0)
            
            edge_val = start if face == "left" else start + res - 1
            is_edge = BinaryOp("==", env.get(b_axis, Literal(0)), Literal(edge_val))
            
            return Ternary(is_edge, bc_ir, self._dispatch(node, env, face))

        return self._dispatch(node, env, face)

    def _dispatch(self, node: Dict[str, Any], env: Dict[str, Expr], face: Optional[str]) -> Expr:
        t = node.get("type")
        if t == "Scalar": return Literal(node["value"])
        if t == "Parameter": return ArrayAccess("p", Literal(self.layout.get_param_offset(node['name'])))
        if t == "State": return self._lower_state(node, env, face)
        if t == "Boundary": return self._lower_boundary(node, env)
        if t == "BinaryOp": return self._lower_binary_op(node, env, face)
        if t == "UnaryOp": return self._lower_unary_op(node, env, face)
        if t == "dirichlet_bnd": return self.lower(node["node"], env) # Safely unwrap injection dictionary
        raise ValueError(f"Unknown IR Node: {t}")

    def _lower_state(self, node: Dict[str, Any], env: Dict[str, Expr], face: Optional[str]) -> Expr:
        offset = self.layout.state_offsets[node["name"]][0]
        target_domain = getattr(self.state_map.get(node["name"]), "domain", None)
        d_name = target_domain.name if target_domain else None
        
        flat_idx = self._flat_index(d_name, env)
        arr = "ydot" if self.use_ydot else "y"
        base_access = ArrayAccess(arr, BinaryOp("+", Literal(offset), flat_idx))
        
        if face and self.current_axis:
            b_axis = self.topo.get_base_axis(self.current_axis)
            env_shifted = env.copy()
            shift = 1 if face == "right" else -1
            env_shifted[b_axis] = BinaryOp("+", env.get(b_axis, Literal(0)), Literal(shift))
            
            neighbor_idx = self._flat_index(d_name, env_shifted)
            neighbor_access = ArrayAccess(arr, BinaryOp("+", Literal(offset), neighbor_idx))
            return BinaryOp("*", Literal(0.5), BinaryOp("+", base_access, neighbor_access))
            
        return base_access

    def _lower_boundary(self, node: Dict[str, Any], env: Dict[str, Expr]) -> Expr:
        env_bnd = env.copy()
        
        from ion_flux.compiler.codegen.ast_analysis import extract_state_names
        state_names = extract_state_names(node["child"])
        
        if state_names:
            state_domain = getattr(self.state_map.get(state_names[0]), "domain", None)
            d_name = node.get("domain") or (state_domain.name if state_domain else None)
            
            if d_name:
                b_axis = self.topo.get_base_axis(self.topo.get_axes(d_name)[-1])
                start = self.topo.domains.get(b_axis, {}).get("start_idx", 0)
                res = self.topo.domains.get(b_axis, {}).get("resolution", 1)
                
                b_idx = start if node["side"] == "left" else start + res - 1
                env_bnd[b_axis] = Literal(b_idx)
                
        return self.lower(node["child"], env_bnd, face=None)

    def _lower_binary_op(self, node: Dict[str, Any], env: Dict[str, Expr], face: Optional[str]) -> Expr:
        l = self.lower(node["left"], env, face)
        r = self.lower(node["right"], env, face)
        op = node["op"]
        if op in ("max", "min"): return FuncCall(f"std::{op}", [l, r])
        bop = BinaryOp(self._BIN_SYM[op], l, r) if op != "pow" else FuncCall("std::pow", [l, r])
        if op in ("gt", "lt", "ge", "le", "eq", "ne"): return Ternary(bop, Literal(1.0), Literal(0.0))
        return bop

    def _lower_unary_op(self, node: Dict[str, Any], env: Dict[str, Expr], face: Optional[str]) -> Expr:
        op, child = node["op"], node["child"]
        if op == "dt": 
            self.use_ydot = True
            res = self.lower(child, env, face)
            self.use_ydot = False
            return res
        if op == "integral": return self._lower_integral(node, child, env)
        if op == "coords": return self._lower_coords(node, env)
        
        prev_axis = self.current_axis
        if op in ("grad", "div"): self.current_axis = node.get("axis") or prev_axis

        if op == "grad": res = self._lower_grad(child, env, face)
        elif op == "div": res = self._lower_div(child, env)
        else:
            c_ir = self.lower(child, env, face)
            res = RawCpp(f"(-{c_ir.to_cpp()})") if op == "neg" else FuncCall(self._UNARY_SYM[op], [c_ir])
            
        self.current_axis = prev_axis
        return res

    def _lower_coords(self, node: Dict[str, Any], env: Dict[str, Expr]) -> Expr:
        b_axis = self.topo.get_base_axis(node.get("axis") or self.current_axis)
        if not b_axis: return Literal(0.0)
        start = self.topo.domains.get(b_axis, {}).get("start_idx", 0)
        local_idx = BinaryOp("-", env.get(b_axis, Literal(0)), Literal(start))
        return BinaryOp("*", local_idx, Var(f"dx_{b_axis}"))

    def _lower_grad(self, child: Dict[str, Any], env: Dict[str, Expr], face: Optional[str]) -> Expr:
        b_axis = self.topo.get_base_axis(self.current_axis)
        dx_ir = Var(f"dx_{b_axis}") if b_axis else Var("dx_default")
        
        if face == "right" or face == "left":
            env_shift = env.copy()
            shift = 1 if face == "right" else -1
            env_shift[b_axis] = BinaryOp("+", env.get(b_axis, Literal(0)), Literal(shift))
            
            c_shift = self.lower(child, env_shift, face=None)
            c_curr = self.lower(child, env, face=None)
            
            if face == "right": return BinaryOp("/", BinaryOp("-", c_shift, c_curr), dx_ir)
            else: return BinaryOp("/", BinaryOp("-", c_curr, c_shift), dx_ir)
            
        # Central Difference
        env_r, env_l = env.copy(), env.copy()
        env_r[b_axis] = BinaryOp("+", env.get(b_axis, Literal(0)), Literal(1))
        env_l[b_axis] = BinaryOp("-", env.get(b_axis, Literal(0)), Literal(1))
        
        r_val = self.lower(child, env_r, face=None)
        l_val = self.lower(child, env_l, face=None)
        return BinaryOp("/", BinaryOp("-", r_val, l_val), BinaryOp("*", Literal(2.0), dx_ir))

    def _lower_integral(self, node: Dict[str, Any], child: Dict[str, Any], env: Dict[str, Expr]) -> Expr:
        target_domain = node.get("over")
        axes = self.topo.get_axes(target_domain)
        
        new_env = env.copy()
        int_id = id(node)
        geom_code = ""
        
        for axis in axes:
            b_axis = self.topo.get_base_axis(axis)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            coord_sys = self.topo.domains.get(axis, {}).get("coord_sys", "cartesian")
            
            int_var = f"i_{int_id}_{axis}"
            new_env[b_axis] = BinaryOp("+", Var(int_var), Literal(start))
            
            if coord_sys == "spherical":
                geom_code += (
                    f"        double r_R = ({int_var} == {res}-1) ? ({int_var} * dx_{b_axis}) : ({int_var} * dx_{b_axis} + 0.5 * dx_{b_axis});\n"
                    f"        double r_L = ({int_var} == 0) ? 0.0 : ({int_var} * dx_{b_axis} - 0.5 * dx_{b_axis});\n"
                    f"        vol *= (4.0/3.0) * 3.141592653589793 * (std::pow(r_R, 3.0) - std::pow(r_L, 3.0));\n"
                )
            elif coord_sys == "unstructured":
                if b_axis in self.layout.mesh_offsets and "volumes" in self.layout.mesh_offsets[b_axis]:
                    vol_off = self.layout.mesh_offsets[b_axis]["volumes"]
                    geom_code += f"        vol *= m[{vol_off} + {int_var}];\n"
                else:
                    geom_code += f"        vol *= 1.0;\n"
            else:
                geom_code += f"        vol *= ({int_var} == 0 || {int_var} == {res}-1) ? 0.5 * dx_{b_axis} : dx_{b_axis};\n"
        
        child_expr = self.lower(child, new_env, face=None)
        
        cpp_code = "[&]() {\n    double sum = 0.0;\n"
        for axis in axes:
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            cpp_code += f"    for(int i_{int_id}_{axis} = 0; i_{int_id}_{axis} < {res}; ++i_{int_id}_{axis}) {{\n"
            
        cpp_code += "        double vol = 1.0;\n" + geom_code
        cpp_code += f"        sum += {child_expr.to_cpp()} * vol;\n"
        
        for _ in axes: cpp_code += "    }\n"
        cpp_code += "    return sum;\n}()"
        
        return RawCpp(cpp_code)

    def _lower_div(self, child: Dict[str, Any], env: Dict[str, Expr]) -> Expr:
        b_axis = self.topo.get_base_axis(self.current_axis)
        if not b_axis: return Literal(0.0)
        
        coord_sys = self.topo.domains.get(self.current_axis, {}).get("coord_sys", "cartesian")
        if coord_sys == "unstructured":
            return self._lower_div_unstructured(child, env, b_axis)

        r_flux = self.lower(child, env, face="right")
        l_flux = self.lower(child, env, face="left")

        # Piecewise Auto-Stitching
        if getattr(self, "is_piecewise", False) and getattr(self, "current_region_data", None):
            reg = self.current_region_data
            start, end = reg["start_idx"], reg["end_idx"]
            
            c_right = BinaryOp("==", env[b_axis], Literal(end - 1))
            c_left = BinaryOp("==", env[b_axis], Literal(start))
            
            for r in self.piecewise_regions:
                if r["start_idx"] == end and r["domain"] in self.region_divs:
                    n_flux = self.lower(self.region_divs[r["domain"]], env, face="right")
                    r_flux = Ternary(c_right, BinaryOp("*", Literal(0.5), BinaryOp("+", r_flux, n_flux)), r_flux)
                if r["end_idx"] == start and r["domain"] in self.region_divs:
                    p_flux = self.lower(self.region_divs[r["domain"]], env, face="left")
                    l_flux = Ternary(c_left, BinaryOp("*", Literal(0.5), BinaryOp("+", l_flux, p_flux)), l_flux)

        dx_ir = Var(f"dx_{b_axis}")
        idx_expr = BinaryOp("-", env[b_axis], Literal(self.topo.domains.get(b_axis, {}).get("start_idx", 0)))
        res = self.topo.domains.get(b_axis, {}).get("resolution", 1)
        
        c_left_bnd = BinaryOp("==", idx_expr, Literal(0))
        c_right_bnd = BinaryOp("==", idx_expr, Literal(res - 1))
        
        i_R = Ternary(c_right_bnd, idx_expr, BinaryOp("+", idx_expr, Literal(0.5)))
        i_L = Ternary(c_left_bnd, idx_expr, BinaryOp("-", idx_expr, Literal(0.5)))
        
        if coord_sys == "spherical":
            A_R, A_L = BinaryOp("*", i_R, BinaryOp("*", i_R, BinaryOp("*", dx_ir, dx_ir))), BinaryOp("*", i_L, BinaryOp("*", i_L, BinaryOp("*", dx_ir, dx_ir)))
            V = BinaryOp("/", BinaryOp("*", BinaryOp("-", BinaryOp("*", A_R, i_R), BinaryOp("*", A_L, i_L)), dx_ir), Literal(3.0))
        elif coord_sys == "cylindrical":
            A_R, A_L = BinaryOp("*", i_R, dx_ir), BinaryOp("*", i_L, dx_ir)
            V = BinaryOp("/", BinaryOp("*", BinaryOp("-", BinaryOp("*", A_R, i_R), BinaryOp("*", A_L, i_L)), dx_ir), Literal(2.0))
        else:
            A_R, A_L = Literal(1.0), Literal(1.0)
            V = BinaryOp("*", BinaryOp("-", i_R, i_L), dx_ir)

        V_safe = FuncCall("std::max", [Literal("1e-30"), V])
        return BinaryOp("/", BinaryOp("-", BinaryOp("*", A_R, r_flux), BinaryOp("*", A_L, l_flux)), V_safe)

    def _lower_div_unstructured(self, child: Dict[str, Any], env: Dict[str, Expr], b_axis: str) -> Expr:
        mesh_name = self.current_axis
        offsets = self.layout.mesh_offsets[mesh_name]
        rp, ci, w = offsets["row_ptr"], offsets["col_ind"], offsets["weights"]
        
        from ion_flux.compiler.codegen.ast_analysis import extract_state_name
        s_off = self.layout.state_offsets[extract_state_name(child)][0]
        
        idx_cpp = env[b_axis].to_cpp()
        
        cpp_code = (
            f"[&]() {{\n    double sum = 0.0;\n"
            f"    for(int k = (int)m[{rp} + {idx_cpp}]; k < (int)m[{rp} + {idx_cpp} + 1]; ++k) {{\n"
            f"        sum += m[{w} + k] * (y[{s_off} + (int)m[{ci} + k]] - y[{s_off} + {idx_cpp}]);\n"
            f"    }}\n    return sum;\n}}()"
        )
        bulk_div = RawCpp(cpp_code)
        
        # Determine the multiplier of grad(c)
        def replace_grad(n):
            if not isinstance(n, dict): return n
            if n.get("type") == "UnaryOp" and n.get("op") == "grad":
                return {"type": "Scalar", "value": 1.0}
            new_n = {}
            for k, v in n.items():
                if isinstance(v, dict): new_n[k] = replace_grad(v)
                elif isinstance(v, list): new_n[k] = [replace_grad(x) for x in v]
                else: new_n[k] = v
            return new_n
            
        modified_child = replace_grad(child)
        multiplier_expr = self.lower(modified_child, env)
        
        bulk_div = BinaryOp("*", multiplier_expr, bulk_div)

        bc_id = child.get("_bc_id")
        bc_terms = []
        if bc_id:
            for s_face in ["left", "right", "top", "bottom"]:
                if s_face in offsets.get("surfaces", {}) and self.ctx.get_neumann_bc(bc_id, s_face):
                    bc_val = self.lower(self.ctx.get_neumann_bc(bc_id, s_face)["ast"], env).to_cpp()
                    mask = f"m[{offsets['surfaces'][s_face]} + {idx_cpp}]"
                    
                    if "volumes" in offsets:
                        bc_terms.append(f"(({bc_val}) * {mask} / std::max(1e-30, m[{offsets['volumes']} + {idx_cpp}]))")
                    else:
                        bc_terms.append(f"({bc_val}) * {mask}")
                        
        if bc_terms: return RawCpp(f"({bulk_div.to_cpp()} + {' + '.join(bc_terms)})")
        return bulk_div

    def generate_ale_dilution(self, state_name: str, env: Dict[str, Expr]) -> List[Expr]:
        ale = []
        domain = getattr(self.state_map.get(state_name), "domain", None)
        if not domain: return ale

        for d_name, binding in self.ctx.dynamic_domains.items():
            if domain.name == d_name:
                L = self.lower(binding["rhs"], env)
                self.use_ydot = True
                L_dot = self.lower(binding["rhs"], env)
                self.use_ydot = False
                
                y_curr = ArrayAccess("y", BinaryOp("+", Literal(self.layout.state_offsets[state_name][0]), self._flat_index(d_name, env)))
                dim_mult = 3.0 if getattr(domain, "coord_sys", "") == "spherical" else (2.0 if getattr(domain, "coord_sys", "") == "cylindrical" else 1.0)
                
                div_v = BinaryOp("*", Literal(dim_mult), BinaryOp("/", L_dot, FuncCall("std::max", [Literal(1e-12), L])))
                ale.append(BinaryOp("*", RawCpp(f"(-{y_curr.to_cpp()})"), div_v))
        return ale