from typing import Dict, Any, Optional, List, Tuple
from ion_flux.stage2_compiler._2_lowering.ir import Expr, Literal, Var, ArrayAccess, BinaryOp, FuncCall, Ternary, RawCpp, UnaryMinus, Reduction
from ion_flux.stage2_compiler._1_analysis.semantics import SemanticContext
from ion_flux.stage2_compiler._2_lowering.context import SpatialContext
from ion_flux.stage2_compiler._2_lowering.dialects import get_dialect

class IndexManager:
    """
    Safely manages multi-dimensional broadcasting and C-array flattening.
    Maps high-level topological axes to their current loop evaluation variables.
    """
    def __init__(self, topo):
        self.topo = topo
        self.active_indices: Dict[str, Expr] = {}

    def register(self, axis: str, expr: Expr) -> None:
        """Binds a topological base axis to a specific loop index expression."""
        self.active_indices[axis] = expr

    def get_local(self, axis: str) -> Expr:
        """Retrieves the active loop index for a given axis."""
        base = self.topo.get_base_axis(axis)
        return self.active_indices.get(base, Literal(0))

    def get_flat_index(self, domain_name: Optional[str]) -> Expr:
        """
        Calculates the flat 1D C-array memory offset for a multi-dimensional domain
        based on the currently registered loop indices and static domain strides.
        """
        if not domain_name: 
            return Literal(0)
            
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
    """
    Transforms topology-agnostic Python AST math into deterministic Loop-Level MIR.
    Strictly stateless: Relies entirely on the immutable SpatialContext for tracking evaluation state.
    """
    
    _BIN_SYM = {
        "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "std::pow",
        "gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="
    }
    
    _UNARY_SYM = {
        "abs": "std::abs", "exp": "std::exp", "log": "std::log", 
        "sin": "std::sin", "cos": "std::cos", "sqrt": "std::sqrt"
    }

    def __init__(self, layout, state_map, semantic_ctx: SemanticContext, topo, target: str):
        self.layout = layout
        self.state_map = state_map
        self.semantic_ctx = semantic_ctx
        self.topo = topo
        self.target = target

    def lower(self, node: Dict[str, Any], idx_mgr: IndexManager, ctx: SpatialContext, face: Optional[str] = None) -> Expr:
        bc_info = self.semantic_ctx.get_neumann_bc(node.get("_bc_id"), face)
        if bc_info:
            bc_ir = self.lower(bc_info["ast"], idx_mgr, ctx, face=None)
            axis = ctx.axis
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            b_axis = self.topo.get_base_axis(axis)
            
            edge_val = start if face == "left" else start + res - 1
            is_edge = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(edge_val))
            
            return Ternary(is_edge, bc_ir, self._dispatch(node, idx_mgr, ctx, face))

        return self._dispatch(node, idx_mgr, ctx, face)

    def _dispatch(self, node: Dict[str, Any], idx_mgr: IndexManager, ctx: SpatialContext, face: Optional[str]) -> Expr:
        t = node.get("type")
        if t == "Scalar": return Literal(node["value"])
        if t == "Parameter": return ArrayAccess("p", Literal(self.layout.get_param_offset(node['name'])))
        if t == "State": return self._lower_state(node, idx_mgr, ctx, face)
        if t == "Boundary": return self._lower_boundary(node, idx_mgr, ctx)
        if t == "BinaryOp": return self._lower_binary_op(node, idx_mgr, ctx, face)
        if t == "UnaryOp": return self._lower_unary_op(node, idx_mgr, ctx, face)
        if t == "dirichlet_bnd": return self.lower(node["node"], idx_mgr, ctx) 
        raise ValueError(f"Unknown IR Node: {t}")

    def _array_access(self, arr: str, index: Expr) -> Expr:
        return ArrayAccess(arr, index)

    def _lower_state(self, node: Dict[str, Any], idx_mgr: IndexManager, ctx: SpatialContext, face: Optional[str]) -> Expr:
        state_name = node["name"]
        offset = self.layout.state_offsets[state_name][0]
        target_domain = getattr(self.state_map.get(state_name), "domain", None)
        d_name = target_domain.name if target_domain else None
        
        flat_idx = idx_mgr.get_flat_index(d_name)
        arr = "ydot" if ctx.use_ydot else "y"
        base_access = self._array_access(arr, BinaryOp("+", Literal(offset), flat_idx))
        
        if face and ctx.axis:
            b_axis = self.topo.get_base_axis(ctx.axis)
            res = self.topo.domains.get(ctx.axis, {}).get("resolution", 1)
            start = self.topo.domains.get(ctx.axis, {}).get("start_idx", 0)
            
            idx_shifted = idx_mgr.clone()
            shift = 1 if face == "right" else -1
            idx_shifted.register(b_axis, BinaryOp("+", idx_mgr.get_local(b_axis), Literal(shift)))
            
            neighbor_idx = idx_shifted.get_flat_index(d_name)
            neighbor_access = self._array_access(arr, BinaryOp("+", Literal(offset), neighbor_idx))
            
            interpolated_access = BinaryOp("*", Literal(0.5), BinaryOp("+", base_access, neighbor_access))
            
            dirichlet_bcs = self.semantic_ctx.get_dirichlet_bc(state_name)
            if dirichlet_bcs:
                local_idx = BinaryOp("-", idx_mgr.get_local(b_axis), Literal(start))
                if face == "left" and "left" in dirichlet_bcs:
                    is_edge = BinaryOp("==", local_idx, Literal(0))
                    val_ir = self.lower(dirichlet_bcs["left"], idx_mgr, ctx, face=None)
                    return Ternary(is_edge, val_ir, interpolated_access)
                if face == "right" and "right" in dirichlet_bcs:
                    is_edge = BinaryOp("==", local_idx, Literal(res - 1))
                    val_ir = self.lower(dirichlet_bcs["right"], idx_mgr, ctx, face=None)
                    return Ternary(is_edge, val_ir, interpolated_access)

            return interpolated_access
            
        return base_access

    def _lower_boundary(self, node: Dict[str, Any], idx_mgr: IndexManager, ctx: SpatialContext) -> Expr:
        idx_bnd = idx_mgr.clone()
        from ion_flux.stage2_compiler._1_analysis.ast_utils import extract_state_names
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
                
        return self.lower(node["child"], idx_bnd, ctx, face=None)

    def _lower_binary_op(self, node: Dict[str, Any], idx_mgr: IndexManager, ctx: SpatialContext, face: Optional[str]) -> Expr:
        l = self.lower(node["left"], idx_mgr, ctx, face)
        r = self.lower(node["right"], idx_mgr, ctx, face)
        op = node["op"]
        
        if op in ("max", "min"): return FuncCall(f"std::{op}", [l, r])
        bop = BinaryOp(self._BIN_SYM[op], l, r) if op != "pow" else FuncCall("std::pow", [l, r])
        if op in ("gt", "lt", "ge", "le", "eq", "ne"): return Ternary(bop, Literal(1.0), Literal(0.0))
            
        return bop

    def _lower_unary_op(self, node: Dict[str, Any], idx_mgr: IndexManager, ctx: SpatialContext, face: Optional[str]) -> Expr:
        op, child = node["op"], node["child"]
        if op == "dt": return self.lower(child, idx_mgr, ctx.with_updates(use_ydot=True), face)
        if op == "integral": return self._lower_integral(node, child, idx_mgr, ctx)
        if op == "coords": return self._lower_coords(node, idx_mgr, ctx)
        
        child_ctx = ctx
        if op in ("grad", "div"): 
            child_ctx = ctx.with_updates(axis=node.get("axis") or ctx.axis)

        if op == "grad": 
            dialect = get_dialect(self.topo, self.layout, child_ctx.axis)
            return dialect.gradient(self, child, idx_mgr, child_ctx, face)
        elif op == "div": 
            dialect = get_dialect(self.topo, self.layout, child_ctx.axis)
            return dialect.divergence(self, child, idx_mgr, child_ctx)
            
        c_ir = self.lower(child, idx_mgr, child_ctx, face)
        if op == "neg": return UnaryMinus(c_ir)
            
        return FuncCall(self._UNARY_SYM[op], [c_ir])

    def _lower_coords(self, node: Dict[str, Any], idx_mgr: IndexManager, ctx: SpatialContext) -> Expr:
        axis = node.get("axis") or ctx.axis
        b_axis = self.topo.get_base_axis(axis)
        
        if not b_axis or self.topo.domains.get(b_axis, {}).get("coord_sys") == "unstructured": 
            return Literal(0.0)
            
        idx_expr = idx_mgr.get_local(b_axis)
        off_centers = self.layout.mesh_offsets[b_axis]["w_centers"]
        w_center = ArrayAccess("m", BinaryOp("+", Literal(off_centers), idx_expr))
        
        bounds = self.topo.domains.get(b_axis, {}).get("bounds", (0.0, 1.0))
        l_phys_ir = Var(f"L_phys_{b_axis}")
        
        return BinaryOp("+", Literal(bounds[0]), BinaryOp("*", l_phys_ir, w_center))

    def _lower_integral(self, node: Dict[str, Any], child: Dict[str, Any], idx_mgr: IndexManager, ctx: SpatialContext) -> Expr:
        from ion_flux.stage2_compiler._4_codegen.emitter import CppEmitter
        target_domain = node.get("over")
        axes = self.topo.get_axes(target_domain)
        
        idx_new = idx_mgr.clone()
        int_id = id(node)
        geom_code = ""
        loop_vars = []
        loop_ends = []
        
        for axis in axes:
            b_axis = self.topo.get_base_axis(axis)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            int_var = f"i_{int_id}_{axis}"
            
            loop_vars.append(int_var)
            loop_ends.append(Literal(res))
            idx_new.register(b_axis, BinaryOp("+", Var(int_var), Literal(start)))
            
            dialect = get_dialect(self.topo, self.layout, axis)
            geom_code += dialect.integral_volume_weight(int_var, start)
        
        int_ctx = ctx
        if axes:
            int_ctx = ctx.with_updates(axis=axes[-1])
            
        child_expr = self.lower(child, idx_new, int_ctx, face=None)
        child_cpp = CppEmitter().emit(child_expr)
        
        cpp_code = "[&]() {\n    double sum = 0.0;\n"
        for axis in axes:
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            cpp_code += f"    #pragma clang loop unroll(full)\n    for(int i_{int_id}_{axis} = 0; i_{int_id}_{axis} < {res}; ++i_{int_id}_{axis}) {{\n"
            
        cpp_code += "        double vol = 1.0;\n" + geom_code
        cpp_code += f"        sum += {child_cpp} * vol;\n"
        
        for _ in axes: cpp_code += "    }\n"
        cpp_code += "    return sum;\n}()"
        
        return Reduction(loop_vars, loop_ends, child_expr, cpp_code)

    def _harmonic_mean(self, a: Expr, b: Expr) -> Expr:
        abs_a = FuncCall("std::abs", [a])
        abs_b = FuncCall("std::abs", [b])
        
        term1 = BinaryOp("*", a, abs_b)
        term2 = BinaryOp("*", b, abs_a)
        base_num = BinaryOp("+", term1, term2)
        
        sum_ab = BinaryOp("+", a, b)
        reg_num = BinaryOp("*", Literal("5e-31"), sum_ab)
        
        num = BinaryOp("+", base_num, reg_num)
        den = BinaryOp("+", BinaryOp("+", abs_a, abs_b), Literal("1e-30"))
        
        return BinaryOp("/", num, den)

    def stitch_piecewise_fluxes(self, r_flux: Expr, l_flux: Expr, idx_mgr: IndexManager, ctx: SpatialContext, b_axis: str) -> Tuple[Expr, Expr]:
        """Sutures independent regional fluxes across macro material interfaces."""
        if ctx.is_piecewise and ctx.current_region_data:
            reg = ctx.current_region_data
            start, end = reg["start_idx"], reg["end_idx"]
            
            c_right = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(end - 1))
            c_left = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(start))
            
            for r in (ctx.piecewise_regions or []):
                if r["start_idx"] == end and r["domain"] in (ctx.region_divs or {}):
                    n_flux = self.lower(ctx.region_divs[r["domain"]], idx_mgr, ctx, face="right")
                    r_flux = Ternary(c_right, self._harmonic_mean(r_flux, n_flux), r_flux)
                if r["end_idx"] == start and r["domain"] in (ctx.region_divs or {}):
                    p_flux = self.lower(ctx.region_divs[r["domain"]], idx_mgr, ctx, face="left")
                    l_flux = Ternary(c_left, self._harmonic_mean(l_flux, p_flux), l_flux)
                    
        return r_flux, l_flux

    def generate_ale_dilution(self, state_name: str, idx_mgr: IndexManager, ctx: SpatialContext) -> List[Expr]:
        """
        Dynamically applies Arbitrary Lagrangian-Eulerian kinematics (-ydot * v_mesh) 
        to states operating on moving physical boundaries.
        """
        ale = []
        domain = getattr(self.state_map.get(state_name), "domain", None)
        if not domain: 
            return ale

        dialect = get_dialect(self.topo, self.layout, domain.name)

        for d_name, binding in self.semantic_ctx.dynamic_domains.items():
            if domain.name == d_name:
                L = self.lower(binding["rhs"], idx_mgr, ctx)
                L_dot = self.lower(binding["rhs"], idx_mgr, ctx.with_updates(use_ydot=True))
                
                y_curr = self._array_access("y", BinaryOp("+", Literal(self.layout.state_offsets[state_name][0]), idx_mgr.get_flat_index(d_name)))
                
                dim_mult = dialect.ale_dimension_multiplier()
                div_v = BinaryOp("*", Literal(dim_mult), BinaryOp("/", L_dot, FuncCall("std::max", [Literal(1e-12), L])))
                
                ale.append(BinaryOp("*", UnaryMinus(y_curr), div_v))
                
        return ale