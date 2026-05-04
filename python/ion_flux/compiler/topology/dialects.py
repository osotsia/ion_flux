from typing import Dict, Any, Optional
from ion_flux.compiler.passes.ir import Expr, Literal, Var, ArrayAccess, BinaryOp, FuncCall, RawCpp

class TopologyDialect:
    """Base interface for abstracting spatial geometries."""
    def __init__(self, topo, layout, axis_name: str):
        self.topo = topo
        self.layout = layout
        self.axis_name = axis_name
        self.b_axis = topo.get_base_axis(axis_name) if axis_name else None

    def divergence(self, visitor, child: Dict[str, Any], idx_mgr, ctx) -> Expr:
        raise NotImplementedError

    def gradient(self, visitor, child: Dict[str, Any], idx_mgr, ctx, face: Optional[str]) -> Expr:
        raise NotImplementedError

    def integral_volume_weight(self, int_var: str, start: int) -> str:
        raise NotImplementedError

    def ale_dimension_multiplier(self) -> float:
        return 1.0


class StructuredDialect(TopologyDialect):
    """Emits Finite Volume Method geometric operators for uniform Cartesian/Cylindrical/Spherical grids."""
    def __init__(self, topo, layout, axis_name: str, coord_sys: str):
        super().__init__(topo, layout, axis_name)
        self.coord_sys = coord_sys

    def divergence(self, visitor, child: Dict[str, Any], idx_mgr, ctx) -> Expr:
        r_flux = visitor.lower(child, idx_mgr, ctx, face="right")
        l_flux = visitor.lower(child, idx_mgr, ctx, face="left")
        
        # Suture regional fluxes using harmonic means to conserve mass across material interfaces
        r_flux, l_flux = visitor.stitch_piecewise_fluxes(r_flux, l_flux, idx_mgr, ctx, self.b_axis)
        
        l_phys_ir = Var(f"L_phys_{self.b_axis}")
        idx_expr = idx_mgr.get_local(self.b_axis)
        
        off_A = self.layout.mesh_offsets[self.b_axis]["w_A_faces"]
        off_V = self.layout.mesh_offsets[self.b_axis]["w_V_nodes"]
        
        A_L = ArrayAccess("m", BinaryOp("+", Literal(off_A), idx_expr))
        A_R = ArrayAccess("m", BinaryOp("+", Literal(off_A), BinaryOp("+", idx_expr, Literal(1))))
        V_i = ArrayAccess("m", BinaryOp("+", Literal(off_V), idx_expr))
        
        V_scaled = BinaryOp("*", V_i, l_phys_ir)
        V_safe = FuncCall("std::max", [Literal("1e-30"), V_scaled])
        net_flux = BinaryOp("-", BinaryOp("*", A_R, r_flux), BinaryOp("*", A_L, l_flux))
        
        return BinaryOp("/", net_flux, V_safe)

    def gradient(self, visitor, child: Dict[str, Any], idx_mgr, ctx, face: Optional[str]) -> Expr:
        l_phys_ir = Var(f"L_phys_{self.b_axis}") if self.b_axis else Var("L_phys_default")
        
        res = self.topo.domains.get(self.b_axis, {}).get("resolution", 1)
        idx_expr = idx_mgr.get_local(self.b_axis)
        off_w_dx = self.layout.mesh_offsets[self.b_axis]["w_dx_faces"]
        
        if face == "right" or face == "left":
            idx_shift = idx_mgr.clone()
            shift = 1 if face == "right" else -1
            idx_shift.register(self.b_axis, BinaryOp("+", idx_expr, Literal(shift)))
            
            c_shift = visitor.lower(child, idx_shift, ctx, face=None)
            c_curr = visitor.lower(child, idx_mgr, ctx, face=None)
            
            face_idx = idx_expr if face == "right" else BinaryOp("-", idx_expr, Literal(1))
            clamped_face = FuncCall("CLAMP", [face_idx, Literal(max(res - 1, 1))])
            w_dx = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_face))
            
            dist_ir = BinaryOp("*", l_phys_ir, w_dx)
            dist_safe = FuncCall("std::max", [Literal("1e-30"), dist_ir])
            
            if face == "right": 
                return BinaryOp("/", BinaryOp("-", c_shift, c_curr), dist_safe)
            else: 
                return BinaryOp("/", BinaryOp("-", c_curr, c_shift), dist_safe)
            
        idx_r, idx_l = idx_mgr.clone(), idx_mgr.clone()
        idx_r.register(self.b_axis, BinaryOp("+", idx_expr, Literal(1)))
        idx_l.register(self.b_axis, BinaryOp("-", idx_expr, Literal(1)))
        
        r_val = visitor.lower(child, idx_r, ctx, face=None)
        l_val = visitor.lower(child, idx_l, ctx, face=None)
        
        clamped_r = FuncCall("CLAMP", [idx_expr, Literal(max(res - 1, 1))])
        clamped_l = FuncCall("CLAMP", [BinaryOp("-", idx_expr, Literal(1)), Literal(max(res - 1, 1))])
        
        w_dx_r = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_r))
        w_dx_l = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_l))
        
        w_dist_total = BinaryOp("+", w_dx_r, w_dx_l)
        dist_ir = BinaryOp("*", l_phys_ir, w_dist_total)
        dist_safe = FuncCall("std::max", [Literal("1e-30"), dist_ir])
        
        return BinaryOp("/", BinaryOp("-", r_val, l_val), dist_safe)

    def integral_volume_weight(self, int_var: str, start: int) -> str:
        dim_exp = 3.0 if self.coord_sys == "spherical" else (2.0 if self.coord_sys == "cylindrical" else 1.0)
        vol_off = self.layout.mesh_offsets[self.b_axis]["w_V_nodes"]
        return (
            f"        double L_scale_{self.b_axis} = std::pow(L_phys_{self.b_axis}, {dim_exp});\n"
            f"        vol *= m[{vol_off} + {start} + {int_var}] * L_scale_{self.b_axis};\n"
        )

    def ale_dimension_multiplier(self) -> float:
        if self.coord_sys == "spherical": return 3.0
        if self.coord_sys == "cylindrical": return 2.0
        return 1.0


class UnstructuredDialect(TopologyDialect):
    """Emits explicit CSR pointers for 3D unstructured meshes."""
    def divergence(self, visitor, child: Dict[str, Any], idx_mgr, ctx) -> Expr:
        from ion_flux.compiler.codegen.emitter import CppEmitter
        from ion_flux.compiler.codegen.ast_analysis import extract_state_name
        
        offsets = self.layout.mesh_offsets[self.axis_name]
        rp, ci, w = offsets["row_ptr"], offsets["col_ind"], offsets["weights"]
        s_off = self.layout.state_offsets[extract_state_name(child)][0]
        
        emitter = CppEmitter()
        idx_cpp = emitter.emit(idx_mgr.get_local(self.b_axis))
        
        cpp_code = (
            f"[&]() {{\n    double sum = 0.0;\n"
            f"    for(int k = (int)m[{rp} + {idx_cpp}]; k < (int)m[{rp} + {idx_cpp} + 1]; ++k) {{\n"
            f"        sum += m[{w} + k] * (y[{s_off} + (int)m[{ci} + k]] - y[{s_off} + {idx_cpp}]);\n"
            f"    }}\n    return sum;\n}}()"
        )
        bulk_div = RawCpp(cpp_code)
        
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
            
        multiplier_expr = visitor.lower(replace_grad(child), idx_mgr, ctx)
        bulk_div = BinaryOp("*", multiplier_expr, bulk_div)

        bc_id = child.get("_bc_id")
        bc_terms = []
        if bc_id:
            for s_face in ["left", "right", "top", "bottom"]:
                if s_face in offsets.get("surfaces", {}) and visitor.semantic_ctx.get_neumann_bc(bc_id, s_face):
                    bc_val = emitter.emit(visitor.lower(visitor.semantic_ctx.get_neumann_bc(bc_id, s_face)["ast"], idx_mgr, ctx))
                    mask = f"m[{offsets['surfaces'][s_face]} + {idx_cpp}]"
                    
                    if "volumes" in offsets:
                        bc_terms.append(f"(({bc_val}) * {mask} / std::max(1e-30, m[{offsets['volumes']} + {idx_cpp}]))")
                    else:
                        bc_terms.append(f"({bc_val}) * {mask}")
                        
        if bc_terms: 
            return RawCpp(f"({emitter.emit(bulk_div)} + {' + '.join(bc_terms)})")
            
        return bulk_div

    def gradient(self, visitor, child: Dict[str, Any], idx_mgr, ctx, face: Optional[str]) -> Expr:
        return Literal(0.0)

    def integral_volume_weight(self, int_var: str, start: int) -> str:
        if self.b_axis in self.layout.mesh_offsets and "volumes" in self.layout.mesh_offsets[self.b_axis]:
            vol_off = self.layout.mesh_offsets[self.b_axis]["volumes"]
            return f"        vol *= m[{vol_off} + {int_var}];\n"
        return f"        vol *= 1.0;\n"


def get_dialect(topo, layout, axis_name: Optional[str]) -> TopologyDialect:
    if not axis_name:
        return StructuredDialect(topo, layout, None, "cartesian")
        
    coord_sys = topo.domains.get(axis_name, {}).get("coord_sys", "cartesian")
    if coord_sys == "unstructured":
        return UnstructuredDialect(topo, layout, axis_name)
        
    return StructuredDialect(topo, layout, axis_name, coord_sys)