"""
================================================================================
Middle-End Compiler: Spatial Lowering Pass (spatial.py)
================================================================================

MENTAL MAP:
-----------
This module bridges the gap between Abstract Math and execution. It takes a purely 
semantic Abstract Syntax Tree (AST) and lowers it into an Intermediate Representation 
(IR) that directly translates to C++ arrays and loops.

Key responsibilities:
1. Discretization: Turns `fx.grad(c)` and `fx.div(N)` into Finite Volume Method 
   (FVM) numerical stencils (e.g., `(y[i+1] - y[i-1]) / (2*dx)`).
2. Memory Striding: Flattens hierarchical composite meshes (e.g., 1D Macro * 1D Micro) 
   into flat 1D C-arrays, handling the modulo indexing automatically.
3. Coordinate Geometry: Automatically applies Cartesian, Cylindrical, or Spherical 
   volumetric scalings (e.g., `r^2`) to divergence operators.
4. Boundary Injection: Intercepts specific mathematical nodes and overwrites their 
   evaluation with user-defined Neumann flux bounds or Piecewise interface stitching.
================================================================================
"""

from typing import Dict, Any, Optional, List
from .ir import Expr, Literal, Var, ArrayAccess, BinaryOp, FuncCall, Ternary, RawCpp
from .semantic import SemanticContext
from ion_flux.compiler.codegen.topology import get_stride, get_local_index, get_coord_sys, get_resolution

class SpatialLoweringVisitor:
    """Recursively walks the AST, emitting discrete C++ IR nodes."""
    
    _BIN_SYM = {
        "add": "+", "sub": "-", "mul": "*", "div": "/", 
        "gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="
    }
    _UNARY_SYM = {
        "neg": "-", "abs": "std::abs", "exp": "std::exp", 
        "log": "std::log", "sin": "std::sin", "cos": "std::cos"
    }

    # ==========================================================================
    # 1. INITIALIZATION & STATE
    # ==========================================================================

    def __init__(self, layout, state_map, semantic_ctx: SemanticContext):
        self.layout = layout
        self.state_map = state_map
        self.ctx = semantic_ctx
        
        # Stateful tracking during tree traversal
        self.current_domain = None
        self.current_axis = None
        self.use_ydot = False


    def _get_topological_idx(self, idx: Expr) -> Expr:
        """Returns the absolute topological index, regardless of Piecewise or Standard loop bounds."""
        start_idx = getattr(self.current_domain, "start_idx", 0)
        if getattr(self, "is_piecewise", False):
            # In Piecewise, the loop variable already spans the global domain
            return idx
        else:
            if start_idx > 0:
                return BinaryOp("+", idx, Literal(start_idx))
            return idx

    # ==========================================================================
    # 2. MAIN ENTRY POINT & DISPATCH
    # ==========================================================================

    def lower(self, node: Dict[str, Any], idx: Expr, face: Optional[str] = None) -> Expr:
        """
        Primary entry point for lowering a node.
        Checks if the node is intercepted by a Boundary Condition (Neumann flux) 
        before dispatching to standard math evaluation.
        """
        # 1. Boundary Interception
        bc_info = self.ctx.get_neumann_bc(node.get("_bc_id"), face)
        if bc_info:
            bc_ast = bc_info["ast"]
            bc_ir = self.lower(bc_ast, idx, face=None)
            
            axis = self.current_axis if self.current_axis else getattr(self.current_domain, "name", "")
            res_val = int(get_resolution(self.current_domain, axis)) if self.current_domain else 1
            
            if hasattr(self.current_domain, "domains") and len(self.current_domain.domains) == 2:
                # Micro-particles use repeating relative modulo geometries
                local_idx = get_local_index(idx.to_cpp(), self.current_domain, axis)
                edge_val_str = "0" if face == "left" else str(res_val - 1)
                is_edge = BinaryOp("==", RawCpp(local_idx), Literal(edge_val_str))
            else:
                # Flat macro grids and sub-regions strictly use absolute global indices
                top_idx = self._get_topological_idx(idx).to_cpp()
                start_idx = getattr(self.current_domain, "start_idx", 0)
                edge_val_str = str(start_idx) if face == "left" else str(start_idx + res_val - 1)
                is_edge = BinaryOp("==", RawCpp(top_idx), Literal(edge_val_str))
            
            base_eval = self._dispatch(node, idx, face)
            
            # C++ Output: (is_edge ? boundary_value : normal_evaluation)
            return Ternary(is_edge, bc_ir, base_eval)

        # 2. Standard Dispatch
        return self._dispatch(node, idx, face)

    def _dispatch(self, node: Dict[str, Any], idx: Expr, face: Optional[str] = None) -> Expr:
        """Routes the AST node to the correct specific lowering method."""
        node_type = node.get("type")
        
        if node_type == "Scalar": 
            return Literal(node["value"])
        if node_type == "Parameter": 
            return ArrayAccess("p", Literal(self.layout.get_param_offset(node['name'])))
        if node_type == "State": 
            return self._lower_state(node, idx, face)
        if node_type == "Boundary": 
            return self._lower_boundary(node, idx)
        if node_type == "BinaryOp": 
            return self._lower_binary_op(node, idx, face)
        if node_type == "UnaryOp": 
            return self._lower_unary_op(node, idx, face)
            
        raise ValueError(f"SpatialLoweringVisitor encountered unknown node: {node_type}")

    # ==========================================================================
    # 3. BASE AST NODE HANDLERS
    # ==========================================================================

    def _lower_state(self, node: Dict[str, Any], idx: Expr, face: Optional[str]) -> Expr:
        """Maps an abstract `State` variable to a direct C-array memory fetch: y[idx]"""
        offset, size = self.layout.state_offsets[node["name"]]
        state_obj = self.state_map.get(node["name"])
        target_domain = getattr(state_obj, "domain", None)
        eval_idx = idx
        
        # Handle hierarchical cross-mesh access (e.g. macro variable accessed by a micro equation)
        if self.current_domain and target_domain and self.current_domain != target_domain:
            eval_idx = self._resolve_cross_domain_index(idx, target_domain)
                    
        # Array bounds safety mechanism
        clamped_idx = FuncCall("CLAMP", [eval_idx, Literal(size)])
        array_name = "ydot" if self.use_ydot else "y"
        base_access = ArrayAccess(array_name, BinaryOp("+", Literal(offset), clamped_idx))
        
        # If evaluating exactly on a cell face (between two nodes), perform a linear interpolation.
        # This is the bedrock of staggered-grid Finite Volume Methods.
        if face:
            axis_to_use = self.current_axis if self.current_axis else getattr(self.current_domain, "name", "")
            stride = int(get_stride(self.current_domain, axis_to_use)) if self.current_domain else 1
            offset_val = stride if face == "right" else -stride
            
            neighbor_idx = self._safe_offset(eval_idx, offset_val, self.current_domain, axis_to_use)
            neighbor_idx_clamped = FuncCall("CLAMP", [neighbor_idx, Literal(size)])
            neighbor_access = ArrayAccess(array_name, BinaryOp("+", Literal(offset), neighbor_idx_clamped))
            
            return BinaryOp("*", Literal(0.5), BinaryOp("+", base_access, neighbor_access))
            
        return base_access

    def _lower_boundary(self, node: Dict[str, Any], idx: Expr) -> Expr:
        """Translates `state.right` or `state.left` into a locked index evaluation."""
        side = node["side"]
        
        if node["child"].get("type") == "State":
            state_name = node["child"]["name"]
            _, size = self.layout.state_offsets[state_name]
            
            # Default behavior: grab absolute end of the array
            b_idx = Literal(0) if side == "left" else Literal(size - 1)
            
            # Complex behavior: If the state is a 2D composite domain (Macro * Micro),
            # grabbing the "right" boundary means grabbing the right surface of the micro
            # particle at the *current* Macro slice being iterated over.
            state_obj = self.state_map.get(state_name)
            if state_obj and hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
                macro_domain, micro_domain = state_obj.domain.domains
                
                if node.get("domain") == micro_domain.name:
                    if self.current_domain and getattr(self.current_domain, "name", "") == macro_domain.name:
                        base = BinaryOp("*", idx, Literal(micro_domain.resolution))
                    else:
                        base = BinaryOp("*", BinaryOp("/", idx, Literal(micro_domain.resolution)), Literal(micro_domain.resolution))
                        
                    b_idx = base if side == "left" else BinaryOp("+", base, Literal(micro_domain.resolution - 1))
                    
            return self.lower(node["child"], b_idx, face=None)
            
        return self.lower(node["child"], idx, face=None)

    def _lower_binary_op(self, node: Dict[str, Any], idx: Expr, face: Optional[str]) -> Expr:
        l = self.lower(node["left"], idx, face)
        r = self.lower(node["right"], idx, face)
        op = node["op"]
        
        if op in self._BIN_SYM:
            bop = BinaryOp(self._BIN_SYM[op], l, r)
            # Boolean triggers translate to 1.0 (True) or 0.0 (False) for C-math compatibility
            if op in ("gt", "lt", "ge", "le", "eq", "ne"): 
                return Ternary(bop, Literal(1.0), Literal(0.0))
            return bop
            
        if op == "max": return FuncCall("std::max", [l, r])
        if op == "min": return FuncCall("std::min", [l, r])
        if op == "pow": return FuncCall("std::pow", [l, r])
        
        raise ValueError(f"Unknown BinaryOp: {op}")

    # ==========================================================================
    # 4. CALCULUS OPERATORS (grad, div, integral, dt)
    # ==========================================================================

    def _lower_unary_op(self, node: Dict[str, Any], idx: Expr, face: Optional[str]) -> Expr:
        """Central hub for calculus operations and standard single-argument math."""
        op = node["op"]
        child = node["child"]
        
        if op == "dt":       return self._lower_dt(child, idx)
        if op == "integral": return self._lower_integral(node, child, idx)
        if op == "coords":   return self._lower_coords(node, idx)
        if op == "grad":     return self._lower_grad(node, child, idx, face)
        if op == "div":      return self._lower_div(node, child, idx)
            
        if op in self._UNARY_SYM:
            child_expr = self.lower(child, idx, face)
            func = self._UNARY_SYM[op]
            return RawCpp(f"(-{child_expr.to_cpp()})") if op == "neg" else FuncCall(func, [child_expr])

        raise ValueError(f"Unknown UnaryOp: {op}")

    def _lower_dt(self, child: Dict[str, Any], idx: Expr) -> Expr:
        """Maps `fx.dt(state)` to `ydot[idx]`."""
        state_name = child["name"]
        offset, size = self.layout.state_offsets[state_name]
        clamped = FuncCall("CLAMP", [idx, Literal(size)])
        return ArrayAccess("ydot", BinaryOp("+", Literal(offset), clamped))

    def _lower_integral(self, node: Dict[str, Any], child: Dict[str, Any], idx: Expr) -> Expr:
        """Emits an inline C++ lambda using exact FVM volume integration."""
        target_domain_name = node.get("over")
        res = self.ctx.payload["domains"][target_domain_name]["resolution"]
        coord_sys = self.ctx.payload["domains"][target_domain_name].get("coord_sys", "cartesian")
        dx_ir = Var(f"dx_{target_domain_name}")
        int_idx = f"idx_int_{id(node)}"
        
        # Context switch: We are summing over a DIFFERENT domain than the current loop
        prev_domain = self.current_domain
        prev_is_piecewise = getattr(self, "is_piecewise", False)
        
        domain_obj = None
        for s in self.state_map.values():
            if getattr(s, "domain", None) and getattr(s.domain, "name", "") == target_domain_name:
                domain_obj = s.domain
                break
                
        if domain_obj is None:
            # Fallback: create a mock domain to satisfy coordinate and spatial lookups
            domain_info = self.ctx.payload["domains"].get(target_domain_name, {})
            start_idx = domain_info.get("start_idx", 0)
            parent_name = domain_info.get("parent")
            
            parent_obj = type("MockDomain", (), {"name": parent_name})() if parent_name else None
            
            domain_obj = type("MockDomain", (), {
                "name": target_domain_name, 
                "start_idx": start_idx, 
                "coord_sys": coord_sys,
                "parent": parent_obj
            })()
            
        self.current_domain = domain_obj
        self.is_piecewise = False  # The integral loop evaluates as a standard 0..res local loop
        
        child_expr = self.lower(child, Var(int_idx), face=None)
        
        self.current_domain = prev_domain # Restore context
        self.is_piecewise = prev_is_piecewise
        
        pi_val = "3.14159265358979323846"
        if coord_sys == "spherical":
            geom_code = (
                f"        double dx = {dx_ir.to_cpp()};\n"
                f"        double r_right = ({int_idx} == {res - 1}) ? ({int_idx} * dx) : ({int_idx} * dx + 0.5 * dx);\n"
                f"        double r_left = ({int_idx} == 0) ? 0.0 : ({int_idx} * dx - 0.5 * dx);\n"
                f"        double vol = (4.0/3.0) * {pi_val} * (std::pow(r_right, 3.0) - std::pow(r_left, 3.0));\n"
            )
        elif coord_sys == "cylindrical":
            geom_code = (
                f"        double dx = {dx_ir.to_cpp()};\n"
                f"        double r_right = ({int_idx} == {res - 1}) ? ({int_idx} * dx) : ({int_idx} * dx + 0.5 * dx);\n"
                f"        double r_left = ({int_idx} == 0) ? 0.0 : ({int_idx} * dx - 0.5 * dx);\n"
                f"        double vol = {pi_val} * (std::pow(r_right, 2.0) - std::pow(r_left, 2.0));\n"
            )
        else: # cartesian
            geom_code = (
                f"        double dx = {dx_ir.to_cpp()};\n"
                f"        double vol = ({int_idx} == 0 || {int_idx} == {res - 1}) ? 0.5 * dx : dx;\n"
            )

        cpp_code = (
            f"[&]() {{\n"
            f"    double sum = 0.0;\n"
            f"    for(int {int_idx} = 0; {int_idx} < {res}; ++{int_idx}) {{\n"
            f"{geom_code}"
            f"        sum += {child_expr.to_cpp()} * vol;\n"
            f"    }}\n"
            f"    return sum;\n"
            f"}}()"
        )
        return RawCpp(cpp_code)

    def _lower_coords(self, node: Dict[str, Any], idx: Expr) -> Expr:
        """Maps `domain.coords` to absolute spatial positions (idx * dx)."""
        axis_name = node.get("axis")
        axis_to_use = axis_name or self.current_axis or getattr(self.current_domain, "name", "")
        dx_ir = Var("dx_default") if not axis_to_use else Var(f"dx_{axis_to_use}")
        top_idx = self._get_topological_idx(idx)
        return BinaryOp("*", top_idx, dx_ir)

    def _lower_grad(self, node: Dict[str, Any], child: Dict[str, Any], idx: Expr, face: Optional[str]) -> Expr:
        """Finite difference central gradient. Adjusts stencil direction if queried on a specific face."""
        axis_name = node.get("axis")
        axis_to_use = axis_name or self.current_axis or getattr(self.current_domain, "name", "")
        dx_ir = Var("dx_default") if not axis_to_use else Var(f"dx_{axis_to_use}")
        
        if face == "right":
            right_idx = self._safe_offset(idx, 1, self.current_domain, axis_to_use)
            right = self.lower(child, right_idx, face=None)
            curr = self.lower(child, idx, face=None)
            return BinaryOp("/", BinaryOp("-", right, curr), dx_ir)
        elif face == "left":
            curr = self.lower(child, idx, face=None)
            left_idx = self._safe_offset(idx, -1, self.current_domain, axis_to_use)
            left = self.lower(child, left_idx, face=None)
            return BinaryOp("/", BinaryOp("-", curr, left), dx_ir)
        else: # Central Difference (spanning 2 dx intervals)
            right_idx = self._safe_offset(idx, 1, self.current_domain, axis_to_use)
            left_idx = self._safe_offset(idx, -1, self.current_domain, axis_to_use)
            right = self.lower(child, right_idx, face=None)
            left = self.lower(child, left_idx, face=None)
            return BinaryOp("/", BinaryOp("-", right, left), BinaryOp("*", Literal(2.0), dx_ir))

    def _lower_div(self, node: Dict[str, Any], child: Dict[str, Any], idx: Expr) -> Expr:
        """
        Calculates the spatial Divergence operator. 
        Routes to Unstructured (Graph Traversal) or Structured (FVM Stencils) depending on domain type.
        """
        axis_name = node.get("axis")
        axis_to_use = axis_name or self.current_axis or getattr(self.current_domain, "name", "")
        coord_sys = get_coord_sys(self.current_domain, axis_to_use) if self.current_domain else "cartesian"
        
        if coord_sys == "unstructured":
            return self._lower_div_unstructured(child, idx)
        else:
            return self._lower_div_structured(child, idx, axis_to_use, coord_sys)

    def _lower_div_unstructured(self, child: Dict[str, Any], idx: Expr) -> Expr:
        """Evaluates divergence over an arbitrary 3D mesh utilizing a pre-computed CSR format graph."""
        mesh_name = self.current_domain.name
        offsets = self.layout.mesh_offsets[mesh_name]
        
        # Pointers to the static constant array containing the mesh geometry
        rp, ci, w = offsets["row_ptr"], offsets["col_ind"], offsets["weights"]
        
        from ion_flux.compiler.codegen.ast_analysis import extract_state_name
        state_name = extract_state_name(child)
        s_off, _ = self.layout.state_offsets[state_name]
        
        # Emit an inline C++ lambda to traverse the compressed sparse row (CSR) arrays
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
        bulk_div = RawCpp(cpp_code)
        
        # Neumann boundaries are handled uniquely for unstructured meshes via Surface Mask Arrays
        flux_bc_id = child.get("_bc_id")
        bc_terms = []
        if flux_bc_id:
            for s_face in ["left", "right", "top", "bottom"]:
                bc_info = self.ctx.get_neumann_bc(flux_bc_id, s_face)
                if bc_info and s_face in offsets.get("surfaces", {}):
                    bc_ast = bc_info["ast"]
                    surf_off = offsets["surfaces"][s_face]
                    bc_val = self.lower(bc_ast, idx, face=None).to_cpp()
                    mask_val = f"m[{surf_off} + (int)({idx.to_cpp()})]"
                    bc_terms.append(f"({bc_val}) * {mask_val}")
                    
        if bc_terms:
            return RawCpp(f"({bulk_div.to_cpp()} - ({' + '.join(bc_terms)}))")
        return bulk_div

    def _lower_div_structured(self, child: Dict[str, Any], idx: Expr, axis_to_use: str, coord_sys: str) -> Expr:
        """
        Evaluates divergence on 1D regular grids.
        Enforces Mass Conservation by evaluating outward/inward fluxes at exact cell boundaries,
        incorporating geometric volume scaling (e.g., spherical particle limits).
        """
        dx_ir = Var("dx_default") if not axis_to_use else Var(f"dx_{axis_to_use}")
        res_val = get_resolution(self.current_domain, axis_to_use)
        
        # 1. Evaluate specific fluxes on the cell edges
        prev_axis = getattr(self, "current_axis", None)
        self.current_axis = axis_to_use
        right_flux = self.lower(child, idx, face="right")
        left_flux = self.lower(child, idx, face="left")
        self.current_axis = prev_axis

        # 2. Stitch together Piecewise regions to prevent mass leakage
        if getattr(self, "piecewise_regions", None) and getattr(self, "current_region_data", None):
            right_flux, left_flux = self._stitch_piecewise_fluxes(right_flux, left_flux, idx)

        # 3. Geometric Area and Volume evaluations based on Coordinate System
        top_idx_str = self._get_topological_idx(idx).to_cpp()
        global_idx_str = idx.to_cpp()
        
        if hasattr(self.current_domain, "domains") and len(self.current_domain.domains) == 2:
            local_idx_str = get_local_index(global_idx_str, self.current_domain, axis_to_use)
            cond_left_bnd = BinaryOp("==", RawCpp(local_idx_str), Literal("0"))
            cond_right_bnd = BinaryOp("==", RawCpp(local_idx_str), Literal(f"{res_val} - 1"))
            phys_idx_str = local_idx_str
        else:
            start_idx = getattr(self.current_domain, "start_idx", 0)
            is_abs_left = (start_idx == 0)
            
            parent = getattr(self.current_domain, "parent", None)
            if parent is not None:
                parent_res = int(get_resolution(parent, axis_to_use))
                is_abs_right = (start_idx + int(res_val) >= parent_res)
            else:
                is_abs_right = True

            abs_left_val = start_idx
            abs_right_val = start_idx + int(res_val) - 1
            
            cond_left_bnd = BinaryOp("==", RawCpp(top_idx_str), Literal(str(abs_left_val))) if is_abs_left else Literal("0")
            cond_right_bnd = BinaryOp("==", RawCpp(top_idx_str), Literal(str(abs_right_val))) if is_abs_right else Literal("0")
            phys_idx_str = top_idx_str

        # FVM faces sit EXACTLY on the boundary coordinate, and 0.5 step inward for bulk cells
        idx_right_face = Ternary(cond_right_bnd, RawCpp(phys_idx_str), BinaryOp("+", RawCpp(phys_idx_str), Literal("0.5")))
        idx_left_face = Ternary(cond_left_bnd, RawCpp(phys_idx_str), BinaryOp("-", RawCpp(phys_idx_str), Literal("0.5")))
        
        r_right = BinaryOp("*", idx_right_face, dx_ir)
        r_left = BinaryOp("*", idx_left_face, dx_ir)
        
        if coord_sys == "spherical":
            A_right = BinaryOp("*", r_right, r_right)
            A_left = BinaryOp("*", r_left, r_left)
            V_cell = BinaryOp("/", BinaryOp("-", BinaryOp("*", A_right, r_right), BinaryOp("*", A_left, r_left)), Literal(3.0))
        elif coord_sys == "cylindrical":
            A_right = r_right
            A_left = r_left
            V_cell = BinaryOp("/", BinaryOp("-", BinaryOp("*", A_right, r_right), BinaryOp("*", A_left, r_left)), Literal(2.0))
        else: # Cartesian
            A_right = Literal(1.0)
            A_left = Literal(1.0)
            V_cell = BinaryOp("-", r_right, r_left)
            
        flux_out = BinaryOp("*", A_right, right_flux)
        flux_in = BinaryOp("*", A_left, left_flux)
        
        # 4. Final Mass Conservation Equation (Outflow - Inflow) / Volume
        # Max limit prevents 0/0 NaNs at the origin of spherical geometries
        V_safe = FuncCall("std::max", [Literal("1e-30"), V_cell])
        return BinaryOp("/", BinaryOp("-", flux_out, flux_in), V_safe)

    # ==========================================================================
    # 5. MESH & INDEXING UTILITIES
    # ==========================================================================

    def _safe_offset(self, base_idx: Expr, offset: int, domain: Any, axis: str) -> Expr:
        """
        Calculates neighboring indices in flat memory. 
        Highly complex because composite meshes (e.g. Macro * Micro) interleave data.
        If a macro-node seeks its neighbor, it must jump forward by the length of the entire micro mesh.
        """
        if offset == 0: 
            return base_idx
            
        if domain and type(domain).__name__ == "CompositeDomain" and len(domain.domains) == 2:
            macro_domain, micro_domain = domain.domains
            
            if axis == micro_domain.name:
                # Stepping in the micro mesh (e.g. inside a particle)
                # Ensure we don't step into the adjacent macro particle's memory!
                cond = BinaryOp("==", BinaryOp("%", base_idx, Literal(micro_domain.resolution)), Literal(micro_domain.resolution - 1 if offset > 0 else 0))
                return BinaryOp("+", base_idx, Ternary(cond, Literal(0), Literal(offset)))
                
            elif axis == macro_domain.name:
                # Stepping in the macro mesh (e.g. across the electrode)
                # We must step forward by the entire stride of the micro-mesh
                stride = micro_domain.resolution
                macro_idx = BinaryOp("/", base_idx, Literal(stride))
                cond = BinaryOp("==", macro_idx, Literal(macro_domain.resolution - 1 if offset > 0 else 0))
                return BinaryOp("+", base_idx, Ternary(cond, Literal(0), Literal(offset)))
                
        # Standard 1D memory array jump
        return BinaryOp("+", base_idx, Literal(offset))

    def _resolve_cross_domain_index(self, idx: Expr, target_domain: Any) -> Expr:
        """Translates indices when evaluating equations spanning different spatial domains."""
        if type(self.current_domain).__name__ == "CompositeDomain" and len(self.current_domain.domains) == 2:
            macro_domain, micro_domain = self.current_domain.domains
            if target_domain.name == macro_domain.name:
                return BinaryOp("/", idx, Literal(micro_domain.resolution))
            elif target_domain.name == micro_domain.name:
                return BinaryOp("%", idx, Literal(micro_domain.resolution))
        else:
            target_parent = getattr(target_domain, "parent", None)
            current_parent = getattr(self.current_domain, "parent", None)
            
            # Topological Sub-meshing offsets (e.g. Anode Sub-Mesh starts at index 0, Separator starts at index 40)
            if target_parent and target_parent.name == self.current_domain.name:
                return BinaryOp("-", idx, Literal(target_domain.start_idx))
            elif current_parent and current_parent.name == target_domain.name:
                return BinaryOp("+", idx, Literal(self.current_domain.start_idx))
                
        return idx

    def _stitch_piecewise_fluxes(self, right_flux: Expr, left_flux: Expr, idx: Expr):
        """
        If we are at the interface boundary of two adjacent Piecewise regions, 
        evaluates the correct mathematical fluxes from both sides to guarantee global mass conservation.
        Handles both node-sharing (overlap) and face-sharing meshes natively.
        """
        reg_data = self.current_region_data
        end_idx = reg_data["end_idx"]
        start_idx = reg_data["start_idx"]
        
        # Check for overlapping (node-sharing) meshes
        next_reg_overlap = next((r for r in self.piecewise_regions if r["start_idx"] == end_idx - 1), None)
        prev_reg_overlap = next((r for r in self.piecewise_regions if r["end_idx"] - 1 == start_idx), None)
        
        # Check for non-overlapping (face-sharing) meshes
        next_reg_face = next((r for r in self.piecewise_regions if r["start_idx"] == end_idx), None)
        prev_reg_face = next((r for r in self.piecewise_regions if r["end_idx"] == start_idx), None)
        
        if next_reg_overlap and next_reg_overlap["domain"] in self.region_divs:
            next_flux_node = self.region_divs[next_reg_overlap["domain"]]
            if next_flux_node:
                next_flux_ir = self.lower(next_flux_node, idx, face="right")
                cond = BinaryOp("==", idx, Literal(f"{end_idx - 1}"))
                # Right face of the shared node is fully in the next region
                right_flux = Ternary(cond, next_flux_ir, right_flux)
                
        elif next_reg_face and next_reg_face["domain"] in self.region_divs:
            next_flux_node = self.region_divs[next_reg_face["domain"]]
            if next_flux_node:
                next_flux_ir = self.lower(next_flux_node, idx, face="right")
                cond = BinaryOp("==", idx, Literal(f"{end_idx - 1}"))
                avg_flux = BinaryOp("*", Literal(0.5), BinaryOp("+", right_flux, next_flux_ir))
                right_flux = Ternary(cond, avg_flux, right_flux)

        if prev_reg_overlap and prev_reg_overlap["domain"] in self.region_divs:
            prev_flux_node = self.region_divs[prev_reg_overlap["domain"]]
            if prev_flux_node:
                prev_flux_ir = self.lower(prev_flux_node, idx, face="left")
                cond = BinaryOp("==", idx, Literal(f"{start_idx}"))
                # Left face of the shared node is fully in the previous region
                left_flux = Ternary(cond, prev_flux_ir, left_flux)
                
        elif prev_reg_face and prev_reg_face["domain"] in self.region_divs:
            prev_flux_node = self.region_divs[prev_reg_face["domain"]]
            if prev_flux_node:
                prev_flux_ir = self.lower(prev_flux_node, idx, face="left")
                cond = BinaryOp("==", idx, Literal(f"{start_idx}"))
                avg_flux = BinaryOp("*", Literal(0.5), BinaryOp("+", left_flux, prev_flux_ir))
                left_flux = Ternary(cond, avg_flux, left_flux)
                
        return right_flux, left_flux

    def generate_ale_dilution(self, state_name: str, offset: int, size: int, idx_str: str) -> List[Expr]:
        """
        Injects ALE Moving Mesh kinematics into the state evaluation.
        If the domain physically stretches, local volumetric concentration naturally dilutes.
        Formula: Dilution_Rate = - y_current * (dim_multiplier * (L_dot / L_current))
        """
        ale_terms = []
        state_obj = self.state_map.get(state_name)
        if not state_obj or not getattr(state_obj, "domain", None): 
            return ale_terms

        for d_name, binding in self.ctx.dynamic_domains.items():
            if state_obj.domain.name == d_name:
                idx_expr = Var(idx_str)
                L_expr = self.lower(binding["rhs"], idx_expr)
                
                # Fetch time derivative of the mesh boundary position
                self.use_ydot = True
                L_dot_expr = self.lower(binding["rhs"], idx_expr)
                self.use_ydot = False
                
                y_curr = ArrayAccess("y", BinaryOp("+", Literal(offset), FuncCall("CLAMP", [idx_expr, Literal(size)])))
                
                # Spherical domains stretch 3x faster volumetrically than linear Cartesian domains
                coord_sys = getattr(state_obj.domain, "coord_sys", "cartesian")
                dim_mult = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)
                
                div_v_mesh = BinaryOp("*", Literal(dim_mult), BinaryOp("/", L_dot_expr, FuncCall("std::max", [Literal(1e-12), L_expr])))
                dilution_term = BinaryOp("*", RawCpp(f"(-{y_curr.to_cpp()})"), div_v_mesh)
                
                ale_terms.append(dilution_term)
                
        return ale_terms