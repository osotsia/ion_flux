from typing import Dict, Any, Optional, List
from .ir import Expr, Literal, Var, ArrayAccess, BinaryOp, FuncCall, Ternary, RawCpp
from .semantic import SemanticContext
from .discretization import Discretizer

class IndexManager:
    """Safely manages multi-dimensional broadcasting and C-array flattening."""
    def __init__(self, topo):
        self.topo = topo
        self.active_indices: Dict[str, Expr] = {}

    def register(self, axis: str, expr: Expr):
        self.active_indices[axis] = expr

    def get_local(self, axis: str) -> Expr:
        base = self.topo.get_base_axis(axis)
        return self.active_indices.get(base, Literal(0))

    def get_flat_index(self, domain_name: Optional[str]) -> Expr:
        if not domain_name: return Literal(0)
        axes = self.topo.get_axes(domain_name)
        strides = self.topo.get_strides(domain_name)
        
        terms = []
        for axis in axes:
            base = self.topo.get_base_axis(axis)
            abs_idx = self.active_indices.get(base, Literal(0))
            
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
        
    def clone(self) -> 'IndexManager':
        new_mgr = IndexManager(self.topo)
        new_mgr.active_indices = self.active_indices.copy()
        return new_mgr

class SpatialLoweringVisitor:
    _BIN_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "std::pow",
                "gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="}
    _UNARY_SYM = {"neg": "-", "abs": "std::abs", "exp": "std::exp", "log": "std::log", 
                  "sin": "std::sin", "cos": "std::cos", "sqrt": "std::sqrt"}

    def __init__(self, layout, state_map, semantic_ctx, topo, target):
        self.layout = layout
        self.state_map = state_map
        self.ctx = semantic_ctx
        self.topo = topo
        self.target = target
        self.current_axis = None
        self.use_ydot = False

    def lower(self, node: Dict[str, Any], idx_mgr: IndexManager, face: Optional[str] = None) -> Expr:
        bc_info = self.ctx.get_neumann_bc(node.get("_bc_id"), face)
        if bc_info:
            bc_ir = self.lower(bc_info["ast"], idx_mgr, face=None)
            
            axis = self.current_axis
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            b_axis = self.topo.get_base_axis(axis)
            
            edge_val = start if face == "left" else start + res - 1
            is_edge = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(edge_val))
            
            return Ternary(is_edge, bc_ir, self._dispatch(node, idx_mgr, face))

        return self._dispatch(node, idx_mgr, face)

    def _dispatch(self, node: Dict[str, Any], idx_mgr: IndexManager, face: Optional[str]) -> Expr:
        t = node.get("type")
        if t == "Scalar": return Literal(node["value"])
        if t == "Parameter": return ArrayAccess("p", Literal(self.layout.get_param_offset(node['name'])))
        if t == "State": return self._lower_state(node, idx_mgr, face)
        if t == "Boundary": return self._lower_boundary(node, idx_mgr)
        if t == "BinaryOp": return self._lower_binary_op(node, idx_mgr, face)
        if t == "UnaryOp": return self._lower_unary_op(node, idx_mgr, face)
        if t == "dirichlet_bnd": return self.lower(node["node"], idx_mgr) 
        raise ValueError(f"Unknown IR Node: {t}")

    def _array_access(self, arr: str, index: Expr) -> Expr:
        return ArrayAccess(arr, index)

    def _lower_state(self, node: Dict[str, Any], idx_mgr: IndexManager, face: Optional[str]) -> Expr:
        state_name = node["name"]
        offset = self.layout.state_offsets[state_name][0]
        target_domain = getattr(self.state_map.get(state_name), "domain", None)
        d_name = target_domain.name if target_domain else None
        
        flat_idx = idx_mgr.get_flat_index(d_name)
        arr = "ydot" if self.use_ydot else "y"
        base_access = self._array_access(arr, BinaryOp("+", Literal(offset), flat_idx))
        
        if face and self.current_axis:
            b_axis = self.topo.get_base_axis(self.current_axis)
            res = self.topo.domains.get(self.current_axis, {}).get("resolution", 1)
            start = self.topo.domains.get(self.current_axis, {}).get("start_idx", 0)
            
            idx_shifted = idx_mgr.clone()
            shift = 1 if face == "right" else -1
            idx_shifted.register(b_axis, BinaryOp("+", idx_mgr.get_local(b_axis), Literal(shift)))
            
            neighbor_idx = idx_shifted.get_flat_index(d_name)
            neighbor_access = self._array_access(arr, BinaryOp("+", Literal(offset), neighbor_idx))
            interpolated_access = BinaryOp("*", Literal(0.5), BinaryOp("+", base_access, neighbor_access))
            
            dirichlet_bcs = self.ctx.get_dirichlet_bc(state_name)
            if dirichlet_bcs:
                local_idx = BinaryOp("-", idx_mgr.get_local(b_axis), Literal(start))
                if face == "left" and "left" in dirichlet_bcs:
                    is_edge = BinaryOp("==", local_idx, Literal(0))
                    val_ir = self.lower(dirichlet_bcs["left"], idx_mgr, face=None)
                    return Ternary(is_edge, val_ir, interpolated_access)
                if face == "right" and "right" in dirichlet_bcs:
                    is_edge = BinaryOp("==", local_idx, Literal(res - 1))
                    val_ir = self.lower(dirichlet_bcs["right"], idx_mgr, face=None)
                    return Ternary(is_edge, val_ir, interpolated_access)

            return interpolated_access
            
        return base_access

    def _lower_boundary(self, node: Dict[str, Any], idx_mgr: IndexManager) -> Expr:
        idx_bnd = idx_mgr.clone()
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
                idx_bnd.register(b_axis, Literal(b_idx))
                
        return self.lower(node["child"], idx_bnd, face=None)

    def _lower_binary_op(self, node: Dict[str, Any], idx_mgr: IndexManager, face: Optional[str]) -> Expr:
        l = self.lower(node["left"], idx_mgr, face)
        r = self.lower(node["right"], idx_mgr, face)
        op = node["op"]
        if op in ("max", "min"): return FuncCall(f"std::{op}", [l, r])
        bop = BinaryOp(self._BIN_SYM[op], l, r) if op != "pow" else FuncCall("std::pow", [l, r])
        if op in ("gt", "lt", "ge", "le", "eq", "ne"): return Ternary(bop, Literal(1.0), Literal(0.0))
        return bop

    def _lower_unary_op(self, node: Dict[str, Any], idx_mgr: IndexManager, face: Optional[str]) -> Expr:
        op, child = node["op"], node["child"]
        if op == "dt": 
            self.use_ydot = True
            res = self.lower(child, idx_mgr, face)
            self.use_ydot = False
            return res
        if op == "integral": return self._lower_integral(node, child, idx_mgr)
        if op == "coords": return self._lower_coords(node, idx_mgr)
        
        prev_axis = self.current_axis
        if op in ("grad", "div"): self.current_axis = node.get("axis") or prev_axis

        if op == "grad": res = self._lower_grad(child, idx_mgr, face)
        elif op == "div": res = self._lower_div(child, idx_mgr)
        else:
            c_ir = self.lower(child, idx_mgr, face)
            res = RawCpp(f"(-{c_ir.to_cpp()})") if op == "neg" else FuncCall(self._UNARY_SYM[op], [c_ir])
            
        self.current_axis = prev_axis
        return res

    def _lower_coords(self, node: Dict[str, Any], idx_mgr: IndexManager) -> Expr:
        axis = node.get("axis") or self.current_axis
        b_axis = self.topo.get_base_axis(axis)
        if not b_axis: return Literal(0.0)
        
        if self.topo.domains.get(b_axis, {}).get("coord_sys") == "unstructured":
            return Literal(0.0)
            
        idx_expr = idx_mgr.get_local(b_axis)
        
        off_centers = self.layout.mesh_offsets[b_axis]["w_centers"]
        w_center = ArrayAccess("m", BinaryOp("+", Literal(off_centers), idx_expr))
        
        bounds = self.topo.domains.get(b_axis, {}).get("bounds", (0.0, 1.0))
        l_phys_ir = Var(f"L_phys_{b_axis}")
        
        return BinaryOp("+", Literal(bounds[0]), BinaryOp("*", l_phys_ir, w_center))

    def _lower_grad(self, child: Dict[str, Any], idx_mgr: IndexManager, face: Optional[str]) -> Expr:
        b_axis = self.topo.get_base_axis(self.current_axis)
        l_phys_ir = Var(f"L_phys_{b_axis}") if b_axis else Var("L_phys_default")
        
        if not b_axis or self.topo.domains.get(b_axis, {}).get("coord_sys") == "unstructured":
            return Literal(0.0)
            
        res = self.topo.domains.get(b_axis, {}).get("resolution", 1)
        idx_expr = idx_mgr.get_local(b_axis)
        off_w_dx = self.layout.mesh_offsets[b_axis]["w_dx_faces"]
        
        if face == "right" or face == "left":
            idx_shift = idx_mgr.clone()
            shift = 1 if face == "right" else -1
            idx_shift.register(b_axis, BinaryOp("+", idx_expr, Literal(shift)))
            
            c_shift = self.lower(child, idx_shift, face=None)
            c_curr = self.lower(child, idx_mgr, face=None)
            
            face_idx = idx_expr if face == "right" else BinaryOp("-", idx_expr, Literal(1))
            clamped_face = FuncCall("CLAMP", [face_idx, Literal(max(res - 1, 1))])
            w_dx = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_face))
            
            dist_ir = BinaryOp("*", l_phys_ir, w_dx)
            dist_safe = FuncCall("std::max", [Literal("1e-30"), dist_ir])
            
            if face == "right": return BinaryOp("/", BinaryOp("-", c_shift, c_curr), dist_safe)
            else: return BinaryOp("/", BinaryOp("-", c_curr, c_shift), dist_safe)
            
        idx_r, idx_l = idx_mgr.clone(), idx_mgr.clone()
        idx_r.register(b_axis, BinaryOp("+", idx_expr, Literal(1)))
        idx_l.register(b_axis, BinaryOp("-", idx_expr, Literal(1)))
        
        r_val = self.lower(child, idx_r, face=None)
        l_val = self.lower(child, idx_l, face=None)
        
        clamped_r = FuncCall("CLAMP", [idx_expr, Literal(max(res - 1, 1))])
        clamped_l = FuncCall("CLAMP", [BinaryOp("-", idx_expr, Literal(1)), Literal(max(res - 1, 1))])
        
        w_dx_r = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_r))
        w_dx_l = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_l))
        
        w_dist_total = BinaryOp("+", w_dx_r, w_dx_l)
        dist_ir = BinaryOp("*", l_phys_ir, w_dist_total)
        dist_safe = FuncCall("std::max", [Literal("1e-30"), dist_ir])
        
        return BinaryOp("/", BinaryOp("-", r_val, l_val), dist_safe)

    def _lower_integral(self, node: Dict[str, Any], child: Dict[str, Any], idx_mgr: IndexManager) -> Expr:
        target_domain = node.get("over")
        axes = self.topo.get_axes(target_domain)
        
        idx_new = idx_mgr.clone()
        int_id = id(node)
        geom_code = ""
        
        for axis in axes:
            b_axis = self.topo.get_base_axis(axis)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            coord_sys = self.topo.domains.get(axis, {}).get("coord_sys", "cartesian")
            
            int_var = f"i_{int_id}_{axis}"
            idx_new.register(b_axis, BinaryOp("+", Var(int_var), Literal(start)))
            
            geom_code += Discretizer.integral_volume_code_normalized(coord_sys, int_var, start, b_axis, self.layout)
        
        child_expr = self.lower(child, idx_new, face=None)
        
        cpp_code = "[&]() {\n    double sum = 0.0;\n"
        for axis in axes:
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            # Note: Added back `#pragma clang loop unroll(full)`. 
            # Enzyme has a hard time with complex integrals (Brosa2021)
            # Tradeoff: Increased compile times
            # cpp_code += f"    for(int i_{int_id}_{axis} = 0; i_{int_id}_{axis} < {res}; ++i_{int_id}_{axis}) {{\n"
            cpp_code += f"    #pragma clang loop unroll(full)\n    for(int i_{int_id}_{axis} = 0; i_{int_id}_{axis} < {res}; ++i_{int_id}_{axis}) {{\n"
            
        cpp_code += "        double vol = 1.0;\n" + geom_code
        cpp_code += f"        sum += {child_expr.to_cpp()} * vol;\n"
        
        for _ in axes: cpp_code += "    }\n"
        cpp_code += "    return sum;\n}()"
        
        return RawCpp(cpp_code)

    def _harmonic_mean(self, a: Expr, b: Expr) -> Expr:
        """
        Computes the harmonic mean (2AB / (A+B)) to properly conserve mass 
        across material discontinuities (e.g. stepping from high to low diffusivity).
        
        We use a robust absolute-value formulation:
        J_eff = (A * |B| + B * |A| + 0.5 * eps * (A + B)) / (|A| + |B| + eps)
        
        This perfectly matches the harmonic mean when A and B have the same sign,
        and smoothly evaluates to 0.0 when they have opposite signs.
        
        Crucially, the `0.5 * eps * (A + B)` regularizer prevents the Enzyme AD 
        gradient from vanishing to 0.0 at the origin (A=0, B=0), ensuring the 
        implicit Jacobian remains structurally coupled during flat initializations.
        """
        abs_a = FuncCall("std::abs", [a])
        abs_b = FuncCall("std::abs", [b])
        
        # Primary harmonic terms
        term1 = BinaryOp("*", a, abs_b)
        term2 = BinaryOp("*", b, abs_a)
        base_num = BinaryOp("+", term1, term2)
        
        # Regularization to provide an exact arithmetic mean subgradient at the origin
        sum_ab = BinaryOp("+", a, b)
        reg_num = BinaryOp("*", Literal("5e-31"), sum_ab)
        
        num = BinaryOp("+", base_num, reg_num)
        den = BinaryOp("+", BinaryOp("+", abs_a, abs_b), Literal("1e-30"))
        
        return BinaryOp("/", num, den)

    def _lower_div(self, child: Dict[str, Any], idx_mgr: IndexManager) -> Expr:
        b_axis = self.topo.get_base_axis(self.current_axis)
        if not b_axis: return Literal(0.0)
        
        coord_sys = self.topo.domains.get(self.current_axis, {}).get("coord_sys", "cartesian")
        if coord_sys == "unstructured":
            return self._lower_div_unstructured(child, idx_mgr, b_axis)

        r_flux = self.lower(child, idx_mgr, face="right")
        l_flux = self.lower(child, idx_mgr, face="left")

        if getattr(self, "is_piecewise", False) and getattr(self, "current_region_data", None):
            reg = self.current_region_data
            start, end = reg["start_idx"], reg["end_idx"]
            
            c_right = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(end - 1))
            c_left = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(start))
            
            for r in self.piecewise_regions:
                if r["start_idx"] == end and r["domain"] in self.region_divs:
                    n_flux = self.lower(self.region_divs[r["domain"]], idx_mgr, face="right")
                    r_flux = Ternary(c_right, self._harmonic_mean(r_flux, n_flux), r_flux)
                if r["end_idx"] == start and r["domain"] in self.region_divs:
                    p_flux = self.lower(self.region_divs[r["domain"]], idx_mgr, face="left")
                    l_flux = Ternary(c_left, self._harmonic_mean(l_flux, p_flux), l_flux)

        l_phys_ir = Var(f"L_phys_{b_axis}")
        idx_expr = idx_mgr.get_local(b_axis)
        
        off_A = self.layout.mesh_offsets[b_axis]["w_A_faces"]
        off_V = self.layout.mesh_offsets[b_axis]["w_V_nodes"]
        
        A_L = ArrayAccess("m", BinaryOp("+", Literal(off_A), idx_expr))
        A_R = ArrayAccess("m", BinaryOp("+", Literal(off_A), BinaryOp("+", idx_expr, Literal(1))))
        V_i = ArrayAccess("m", BinaryOp("+", Literal(off_V), idx_expr))
        
        return Discretizer.divergence_normalized(r_flux, l_flux, A_R, A_L, V_i, l_phys_ir)

    def _lower_div_unstructured(self, child: Dict[str, Any], idx_mgr: IndexManager, b_axis: str) -> Expr:
        mesh_name = self.current_axis
        offsets = self.layout.mesh_offsets[mesh_name]
        rp, ci, w = offsets["row_ptr"], offsets["col_ind"], offsets["weights"]
        
        from ion_flux.compiler.codegen.ast_analysis import extract_state_name
        s_off = self.layout.state_offsets[extract_state_name(child)][0]
        idx_cpp = idx_mgr.get_local(b_axis).to_cpp()
        
        cpp_code = Discretizer.unstructured_divergence_code(rp, ci, w, s_off, idx_cpp)
        bulk_div = RawCpp(cpp_code)
        
        def replace_grad(n):
            if not isinstance(n, dict): return n
            if n.get("type") == "UnaryOp" and n.get("op") == "grad": return {"type": "Scalar", "value": 1.0}
            new_n = {}
            for k, v in n.items():
                if isinstance(v, dict): new_n[k] = replace_grad(v)
                elif isinstance(v, list): new_n[k] = [replace_grad(x) for x in v]
                else: new_n[k] = v
            return new_n
            
        multiplier_expr = self.lower(replace_grad(child), idx_mgr)
        bulk_div = BinaryOp("*", multiplier_expr, bulk_div)

        bc_id = child.get("_bc_id")
        bc_terms = []
        if bc_id:
            for s_face in ["left", "right", "top", "bottom"]:
                if s_face in offsets.get("surfaces", {}) and self.ctx.get_neumann_bc(bc_id, s_face):
                    bc_val = self.lower(self.ctx.get_neumann_bc(bc_id, s_face)["ast"], idx_mgr).to_cpp()
                    mask = f"m[{offsets['surfaces'][s_face]} + {idx_cpp}]"
                    
                    if "volumes" in offsets:
                        bc_terms.append(f"(({bc_val}) * {mask} / std::max(1e-30, m[{offsets['volumes']} + {idx_cpp}]))")
                    else:
                        bc_terms.append(f"({bc_val}) * {mask}")
                        
        if bc_terms: return RawCpp(f"({bulk_div.to_cpp()} + {' + '.join(bc_terms)})")
        return bulk_div

    def generate_ale_dilution(self, state_name: str, idx_mgr: IndexManager) -> List[Expr]:
        ale = []
        domain = getattr(self.state_map.get(state_name), "domain", None)
        if not domain: return ale

        for d_name, binding in self.ctx.dynamic_domains.items():
            if domain.name == d_name:
                L = self.lower(binding["rhs"], idx_mgr)
                self.use_ydot = True
                L_dot = self.lower(binding["rhs"], idx_mgr)
                self.use_ydot = False
                
                y_curr = self._array_access("y", BinaryOp("+", Literal(self.layout.state_offsets[state_name][0]), idx_mgr.get_flat_index(d_name)))
                
                dim_mult = Discretizer.ale_dimension_multiplier(getattr(domain, "coord_sys", ""))
                div_v = BinaryOp("*", Literal(dim_mult), BinaryOp("/", L_dot, FuncCall("std::max", [Literal(1e-12), L])))
                ale.append(BinaryOp("*", RawCpp(f"(-{y_curr.to_cpp()})"), div_v))
        return ale