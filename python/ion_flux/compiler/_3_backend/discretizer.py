"""
FVM Discretizer.

Transforms N-Dimensional Math IR into strict 1D Compute IR.
Handles memory stride linearization, Finite Volume Method geometric scaling,
flux upwinding, piecewise interface harmonic averaging, and ALE dynamic mesh dilution.
"""

from typing import Dict, Any, Optional, List, Tuple
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._4_codegen.compute_ir import (
    Expr, Stmt, Literal, Var, ArrayAccess, BinaryOp, FuncCall, Ternary,
    UnaryMinus, Loop, Assign, RawCpp, UnstructuredRead, Reduction
)
from ion_flux.compiler._3_backend.math_ir import (
    MathExpr, MathScalar, MathParameter, MathState, MathBinaryOp,
    MathUnaryOp, MathGrad, MathDiv, MathDt, MathCoords, MathIntegral,
    MathBoundaryRef, MathEquation, MathObservable, MathDirichletOverride, MathSystem
)


def extract_math_state_names(expr: MathExpr) -> List[str]:
    """Recursively extracts all State variable names referenced in a MathExpr."""
    if isinstance(expr, MathState):
        return [expr.name]
    names = []
    if isinstance(expr, (MathUnaryOp, MathGrad, MathDiv, MathDt, MathIntegral, MathBoundaryRef)):
        names.extend(extract_math_state_names(expr.child))
    elif isinstance(expr, MathBinaryOp):
        names.extend(extract_math_state_names(expr.left))
        names.extend(extract_math_state_names(expr.right))
    seen = set()
    return [x for x in names if not (x in seen or seen.add(x))]


def extract_domain_name(expr: MathExpr) -> Optional[str]:
    """Recursively determines the spatial domain of an expression."""
    if isinstance(expr, MathState):
        return expr.domain_name
    if isinstance(expr, MathBoundaryRef):
        return expr.domain or extract_domain_name(expr.child)
    if isinstance(expr, (MathGrad, MathDiv, MathCoords)):
        if expr.axis:
            return expr.axis
        return extract_domain_name(expr.child) if hasattr(expr, "child") else None
    if isinstance(expr, (MathUnaryOp, MathDt)):
        return extract_domain_name(expr.child)
    if isinstance(expr, MathBinaryOp):
        return extract_domain_name(expr.left) or extract_domain_name(expr.right)
    return None


class IndexManager:
    """
    Tracks and maps N-dimensional loop variables to 1D contiguous memory strides.
    """
    def __init__(self, topo: TopologyAnalyzer):
        self.topo = topo
        self.active_indices: Dict[str, Expr] = {}

    def register(self, axis: str, expr: Expr) -> None:
        base = self.topo.get_base_axis(axis)
        self.active_indices[base] = expr

    def get_local(self, axis: str) -> Expr:
        base = self.topo.get_base_axis(axis)
        return self.active_indices.get(base, Literal(0))

    def get_flat_index(self, domain_name: Optional[str]) -> Expr:
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
            if stride > 1:
                terms.append(BinaryOp("*", clamped, Literal(stride)))
            else:
                terms.append(clamped)

        if not terms:
            return Literal(0)
        flat = terms[0]
        for t in terms[1:]:
            flat = BinaryOp("+", flat, t)
        return flat

    def clone(self) -> 'IndexManager':
        clone_mgr = IndexManager(self.topo)
        clone_mgr.active_indices = self.active_indices.copy()
        return clone_mgr


class FVMDiscretizer:
    """
    Discretizes continuum Math IR tensors into 1D array operations and loops.
    """
    _BIN_SYM = {
        "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "std::pow",
        "gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="
    }

    _UNARY_SYM = {
        "abs": "std::abs", "exp": "std::exp", "log": "std::log",
        "sin": "std::sin", "cos": "std::cos", "sqrt": "std::sqrt"
    }

    def __init__(self, layout: Any, topo: TopologyAnalyzer, semantic_ctx: SemanticContext,
                 state_map: Dict[str, Any], target: str = "cpu"):
        self.layout = layout
        self.topo = topo
        self.semantic_ctx = semantic_ctx
        self.state_map = state_map
        self.target = target

    def _resolve_axis(self, axis_name: Optional[str]) -> Optional[str]:
        """Resolves an axis name (including composite domains) to its foundational 1D base axis."""
        if not axis_name:
            return None
        axes = self.topo.get_axes(axis_name)
        if axes:
            return self.topo.get_base_axis(axes[-1])
        return self.topo.get_base_axis(axis_name)

    def discretize_system(self, math_sys: MathSystem) -> Tuple[List[Stmt], List[Stmt], List[Stmt]]:
        """
        Lowers the full MathSystem into C++ compute IR statements.
        Returns: (l_phys_stmts, equation_stmts, observable_stmts)
        """
        l_phys_stmts: List[Stmt] = [RawCpp("double L_phys_default = 1.0;")]

        # Physical domain length parameters
        for d_name, d_info in self.topo.domains.items():
            if d_info.get("type") == "composite":
                continue
            if d_name in math_sys.dynamic_domain_bindings:
                binding = math_sys.dynamic_domain_bindings[d_name]
                idx_mgr = IndexManager(self.topo)
                idx_mgr.register(self.topo.get_base_axis(d_name), Literal(0))
                rhs_ir = self.lower_expr(binding["rhs_expr"], idx_mgr, current_axis=d_name)
                from ion_flux.compiler._4_codegen.cpp_emitter import CppEmitter
                emitter = CppEmitter()
                l_phys_stmts.append(RawCpp(f"double L_phys_{d_name} = std::max(1e-12, (double)({emitter.emit(rhs_ir)}));"))
            else:
                bounds = d_info.get("bounds", (0.0, 1.0))
                l_phys_stmts.append(RawCpp(f"double L_phys_{d_name} = {float(bounds[1] - bounds[0])};"))

        eq_stmts: List[Stmt] = []
        for eq in math_sys.equations:
            eq_stmts.extend(self.discretize_equation(eq))

        for d_override in math_sys.dirichlet_overrides:
            eq_stmts.extend(self.discretize_dirichlet_override(d_override))

        obs_stmts: List[Stmt] = []
        for obs in math_sys.observables:
            obs_stmts.extend(self.discretize_observable(obs))

        return l_phys_stmts, eq_stmts, obs_stmts

    def discretize_equation(self, eq: MathEquation) -> List[Stmt]:
        """Lowers an individual MathEquation into nested loops and residual assignments."""
        axes = self.topo.get_axes(eq.target_domain)
        bounds_override = eq.bounds_override or {}
        base_axis = self._resolve_axis(axes[-1]) if axes else None

        idx_mgr = IndexManager(self.topo)
        for axis in axes:
            base = self.topo.get_base_axis(axis)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            loop_start, _ = bounds_override.get(base, (start, res))
            loop_var = f"idx_{axis}"
            idx_mgr.register(base, BinaryOp("+", Var(loop_var), Literal(loop_start)))

        offset = self.layout.state_offsets[eq.state_name][0]
        flat_idx = idx_mgr.get_flat_index(eq.target_domain)
        res_access = ArrayAccess("res", BinaryOp("+", Literal(offset), flat_idx))

        lhs_ir = self.lower_expr(eq.lhs, idx_mgr, current_axis=base_axis, current_eq=eq)
        rhs_ir = self.lower_expr(eq.rhs, idx_mgr, current_axis=base_axis, current_eq=eq)

        # Dynamic ALE kinematic dilution
        for ale_term in self._generate_ale_dilution(eq.state_name, idx_mgr, current_axis=base_axis):
            rhs_ir = BinaryOp("+", rhs_ir, ale_term)

        assign = Assign(res_access, BinaryOp("-", lhs_ir, rhs_ir))

        curr_body: List[Stmt] = [assign]
        for i, axis in reversed(list(enumerate(axes))):
            base = self.topo.get_base_axis(axis)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            _, loop_res = bounds_override.get(base, (0, res))
            loop_var = f"idx_{axis}"
            pragma = "#pragma omp parallel for" if i == 0 and loop_res > 50 and "omp" in self.target else ""
            curr_body = [Loop(loop_var, Literal(0), Literal(loop_res), curr_body, pragma)]

        return curr_body

    def discretize_dirichlet_override(self, override: MathDirichletOverride) -> List[Stmt]:
        """Lowers an explicit Dirichlet boundary node override."""
        state_obj = self.state_map.get(override.state_name)
        d_name = getattr(state_obj, "domain", None)
        target_dom_name = d_name.name if d_name else None
        last_axis = self.topo.get_axes(target_dom_name)[-1] if target_dom_name else None
        base_axis = self._resolve_axis(last_axis) if last_axis else None

        if last_axis:
            res = self.topo.domains.get(last_axis, {}).get("resolution", 1)
            start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
        else:
            res = 1
            start = 0

        idx = start if override.side == "left" else start + res - 1
        idx_mgr = IndexManager(self.topo)
        if base_axis:
            idx_mgr.register(base_axis, Literal(idx))

        offset = self.layout.state_offsets[override.state_name][0]
        flat_idx = idx_mgr.get_flat_index(target_dom_name)
        res_access = ArrayAccess("res", BinaryOp("+", Literal(offset), flat_idx))
        y_access = ArrayAccess("y", BinaryOp("+", Literal(offset), flat_idx))

        rhs_ir = self.lower_expr(override.value_expr, idx_mgr, current_axis=base_axis)
        assign = Assign(res_access, BinaryOp("-", y_access, rhs_ir))
        return [assign]

    def discretize_observable(self, obs: MathObservable) -> List[Stmt]:
        """Lowers an algebraic observable equation."""
        axes = self.topo.get_axes(obs.target_domain)
        bounds_override = obs.bounds_override or {}
        base_axis = self._resolve_axis(axes[-1]) if axes else None

        idx_mgr = IndexManager(self.topo)
        for axis in axes:
            base = self.topo.get_base_axis(axis)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            loop_start, _ = bounds_override.get(base, (start, res))
            loop_var = f"idx_{axis}"
            idx_mgr.register(base, BinaryOp("+", Var(loop_var), Literal(loop_start)))

        offset = self.layout.obs_offsets[obs.name][0]
        flat_idx = idx_mgr.get_flat_index(obs.target_domain)
        obs_access = ArrayAccess("obs", BinaryOp("+", Literal(offset), flat_idx))

        rhs_ir = self.lower_expr(obs.expr, idx_mgr, current_axis=base_axis)
        assign = Assign(obs_access, rhs_ir)

        curr_body: List[Stmt] = [assign]
        for i, axis in reversed(list(enumerate(axes))):
            base = self.topo.get_base_axis(axis)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            _, loop_res = bounds_override.get(base, (0, res))
            loop_var = f"idx_{axis}"
            pragma = "#pragma omp parallel for" if i == 0 and loop_res > 50 and "omp" in self.target else ""
            curr_body = [Loop(loop_var, Literal(0), Literal(loop_res), curr_body, pragma)]

        return curr_body

    # =========================================================================
    # Expression Lowering
    # =========================================================================

    def lower_expr(self, node: MathExpr, idx_mgr: IndexManager, current_axis: Optional[str] = None,
                   face: Optional[str] = None, current_eq: Optional[MathEquation] = None) -> Expr:
        """Lowers a MathExpr into a Compute IR Expr, applying Neumann boundaries at faces."""
        if face and getattr(node, "bc_id", None):
            bc_info = self.semantic_ctx.get_neumann_bc(node.bc_id, face)
            if bc_info:
                bc_expr = self._lower_ast_dict(bc_info["ast"])
                bc_ir = self.lower_expr(bc_expr, idx_mgr, current_axis, face=None, current_eq=current_eq)

                axis = self._resolve_axis(bc_info.get("domain")) or current_axis
                res = self.topo.domains.get(axis, {}).get("resolution", 1)
                start = self.topo.domains.get(axis, {}).get("start_idx", 0)
                b_axis = self.topo.get_base_axis(axis)

                edge_val = start if face == "left" else start + res - 1
                is_edge = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(edge_val))

                return Ternary(is_edge, bc_ir, self._dispatch(node, idx_mgr, current_axis, face, current_eq))

        return self._dispatch(node, idx_mgr, current_axis, face, current_eq)

    def _dispatch(self, node: MathExpr, idx_mgr: IndexManager, current_axis: Optional[str],
                  face: Optional[str], current_eq: Optional[MathEquation]) -> Expr:
        """Internal dispatcher for node types."""
        if isinstance(node, MathScalar):
            return Literal(node.value)

        if isinstance(node, MathParameter):
            return ArrayAccess("p", Literal(node.offset))

        if isinstance(node, MathState):
            return self._lower_state(node, idx_mgr, current_axis, face)

        if isinstance(node, MathBoundaryRef):
            return self._lower_boundary_ref(node, idx_mgr, current_axis)

        if isinstance(node, MathBinaryOp):
            l = self.lower_expr(node.left, idx_mgr, current_axis, face, current_eq)
            r = self.lower_expr(node.right, idx_mgr, current_axis, face, current_eq)
            op = node.op
            if op in ("max", "min"):
                return FuncCall(f"std::{op}", [l, r])
            bop = BinaryOp(self._BIN_SYM[op], l, r) if op != "pow" else FuncCall("std::pow", [l, r])
            if op in ("gt", "lt", "ge", "le", "eq", "ne"):
                return Ternary(bop, Literal(1.0), Literal(0.0))
            return bop

        if isinstance(node, MathUnaryOp):
            c_ir = self.lower_expr(node.child, idx_mgr, current_axis, face, current_eq)
            if node.op == "neg":
                return UnaryMinus(c_ir)
            if node.op == "coords":
                return self._lower_coords(current_axis, idx_mgr)
            return FuncCall(self._UNARY_SYM.get(node.op, node.op), [c_ir])

        if isinstance(node, MathDt):
            if isinstance(node.child, MathState):
                return self._lower_state(node.child, idx_mgr, current_axis, face, force_ydot=True)
            return self.lower_expr(node.child, idx_mgr, current_axis, face, current_eq)

        if isinstance(node, MathCoords):
            axis = node.axis or current_axis
            return self._lower_coords(axis, idx_mgr)

        if isinstance(node, MathGrad):
            axis = node.axis or current_axis
            return self._lower_gradient(node.child, axis, idx_mgr, face, current_eq)

        if isinstance(node, MathDiv):
            axis = node.axis or current_axis
            return self._lower_divergence(node.child, axis, idx_mgr, current_eq)

        if isinstance(node, MathIntegral):
            return self._lower_integral(node.child, node.over_domain, idx_mgr)

        return Literal(0.0)

    def _lower_state(self, node: MathState, idx_mgr: IndexManager, current_axis: Optional[str],
                     face: Optional[str] = None, force_ydot: bool = False) -> Expr:
        flat_idx = idx_mgr.get_flat_index(node.domain_name)
        arr = "ydot" if (node.is_ydot or force_ydot) else "y"
        base_access = ArrayAccess(arr, BinaryOp("+", Literal(node.offset), flat_idx))

        if face and current_axis:
            b_axis = self._resolve_axis(current_axis)
            res = self.topo.domains.get(current_axis, {}).get("resolution", 1)
            start = self.topo.domains.get(current_axis, {}).get("start_idx", 0)

            idx_shifted = idx_mgr.clone()
            shift = 1 if face == "right" else -1
            idx_shifted.register(b_axis, BinaryOp("+", idx_mgr.get_local(b_axis), Literal(shift)))

            neighbor_idx = idx_shifted.get_flat_index(node.domain_name)
            neighbor_access = ArrayAccess(arr, BinaryOp("+", Literal(node.offset), neighbor_idx))
            interpolated_access = BinaryOp("*", Literal(0.5), BinaryOp("+", base_access, neighbor_access))

            dirichlet_bcs = self.semantic_ctx.get_dirichlet_bc(node.name)
            if dirichlet_bcs:
                local_idx = BinaryOp("-", idx_mgr.get_local(b_axis), Literal(start))
                if face == "left" and "left" in dirichlet_bcs:
                    is_edge = BinaryOp("==", local_idx, Literal(0))
                    val_ir = self.lower_expr(self._lower_ast_dict(dirichlet_bcs["left"]), idx_mgr, current_axis)
                    return Ternary(is_edge, val_ir, interpolated_access)
                if face == "right" and "right" in dirichlet_bcs:
                    is_edge = BinaryOp("==", local_idx, Literal(res - 1))
                    val_ir = self.lower_expr(self._lower_ast_dict(dirichlet_bcs["right"]), idx_mgr, current_axis)
                    return Ternary(is_edge, val_ir, interpolated_access)

            return interpolated_access

        return base_access

    def _lower_boundary_ref(self, node: MathBoundaryRef, idx_mgr: IndexManager, current_axis: Optional[str]) -> Expr:
        idx_bnd = idx_mgr.clone()
        d_name = node.domain or extract_domain_name(node.child)

        if d_name:
            b_axis = self._resolve_axis(d_name)
            start = self.topo.domains.get(b_axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(b_axis, {}).get("resolution", 1)
            b_idx = start if node.side == "left" else start + res - 1
            idx_bnd.register(b_axis, Literal(b_idx))

        return self.lower_expr(node.child, idx_bnd, current_axis=d_name or current_axis)

    def _lower_coords(self, axis: Optional[str], idx_mgr: IndexManager) -> Expr:
        b_axis = self._resolve_axis(axis)
        if not b_axis or self.topo.domains.get(b_axis, {}).get("coord_sys") == "unstructured":
            return Literal(0.0)

        idx_expr = idx_mgr.get_local(b_axis)
        off_centers = self.layout.mesh_offsets[b_axis]["w_centers"]
        w_center = ArrayAccess("m", BinaryOp("+", Literal(off_centers), idx_expr))
        bounds = self.topo.domains.get(b_axis, {}).get("bounds", (0.0, 1.0))
        l_phys_ir = Var(f"L_phys_{b_axis}")
        return BinaryOp("+", Literal(bounds[0]), BinaryOp("*", l_phys_ir, w_center))

    def _lower_gradient(self, child: MathExpr, axis_name: Optional[str], idx_mgr: IndexManager,
                        face: Optional[str], current_eq: Optional[MathEquation]) -> Expr:
        b_axis = self._resolve_axis(axis_name)
        coord_sys = self.topo.domains.get(b_axis, {}).get("coord_sys", "cartesian") if b_axis else "cartesian"

        if coord_sys == "unstructured":
            return Literal(0.0)

        l_phys_ir = Var(f"L_phys_{b_axis}") if b_axis else Var("L_phys_default")
        res = self.topo.domains.get(b_axis, {}).get("resolution", 1) if b_axis else 1
        idx_expr = idx_mgr.get_local(b_axis) if b_axis else Literal(0)
        off_w_dx = self.layout.mesh_offsets[b_axis]["w_dx_faces"] if b_axis else 0

        if face in ("left", "right"):
            idx_shift = idx_mgr.clone()
            shift = 1 if face == "right" else -1
            idx_shift.register(b_axis, BinaryOp("+", idx_expr, Literal(shift)))

            c_shift = self.lower_expr(child, idx_shift, axis_name, face=None, current_eq=current_eq)
            c_curr = self.lower_expr(child, idx_mgr, axis_name, face=None, current_eq=current_eq)

            face_idx = idx_expr if face == "right" else BinaryOp("-", idx_expr, Literal(1))
            clamped_face = FuncCall("CLAMP", [face_idx, Literal(max(res - 1, 1))])
            w_dx = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_face))
            dist_safe = FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", l_phys_ir, w_dx)])

            if face == "right":
                return BinaryOp("/", BinaryOp("-", c_shift, c_curr), dist_safe)
            else:
                return BinaryOp("/", BinaryOp("-", c_curr, c_shift), dist_safe)

        idx_r, idx_l = idx_mgr.clone(), idx_mgr.clone()
        idx_r.register(b_axis, BinaryOp("+", idx_expr, Literal(1)))
        idx_l.register(b_axis, BinaryOp("-", idx_expr, Literal(1)))

        r_val = self.lower_expr(child, idx_r, axis_name, face=None, current_eq=current_eq)
        l_val = self.lower_expr(child, idx_l, axis_name, face=None, current_eq=current_eq)

        clamped_r = FuncCall("CLAMP", [idx_expr, Literal(max(res - 1, 1))])
        clamped_l = FuncCall("CLAMP", [BinaryOp("-", idx_expr, Literal(1)), Literal(max(res - 1, 1))])
        w_dx_r = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_r))
        w_dx_l = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_l))

        dist_safe = FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", l_phys_ir, BinaryOp("+", w_dx_r, w_dx_l))])
        return BinaryOp("/", BinaryOp("-", r_val, l_val), dist_safe)

    def _lower_divergence(self, child: MathExpr, axis_name: Optional[str], idx_mgr: IndexManager,
                          current_eq: Optional[MathEquation]) -> Expr:
        b_axis = self._resolve_axis(axis_name)
        coord_sys = self.topo.domains.get(b_axis, {}).get("coord_sys", "cartesian") if b_axis else "cartesian"

        if coord_sys == "unstructured":
            return self._lower_unstructured_divergence(child, axis_name, idx_mgr, current_eq)

        r_flux = self.lower_expr(child, idx_mgr, axis_name, face="right", current_eq=current_eq)
        l_flux = self.lower_expr(child, idx_mgr, axis_name, face="left", current_eq=current_eq)

        # Apply harmonic mean auto-stitching across piecewise domain regions
        r_flux, l_flux = self._stitch_piecewise_fluxes(r_flux, l_flux, idx_mgr, axis_name, current_eq)

        l_phys_ir = Var(f"L_phys_{b_axis}")
        idx_expr = idx_mgr.get_local(b_axis)
        off_A = self.layout.mesh_offsets[b_axis]["w_A_faces"]
        off_V = self.layout.mesh_offsets[b_axis]["w_V_nodes"]

        A_L = ArrayAccess("m", BinaryOp("+", Literal(off_A), idx_expr))
        A_R = ArrayAccess("m", BinaryOp("+", Literal(off_A), BinaryOp("+", idx_expr, Literal(1))))
        V_i = ArrayAccess("m", BinaryOp("+", Literal(off_V), idx_expr))

        V_safe = FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", V_i, l_phys_ir)])
        net_flux = BinaryOp("-", BinaryOp("*", A_R, r_flux), BinaryOp("*", A_L, l_flux))
        return BinaryOp("/", net_flux, V_safe)

    def _lower_unstructured_divergence(self, child: MathExpr, axis_name: str, idx_mgr: IndexManager,
                                       current_eq: Optional[MathEquation]) -> Expr:
        offsets = self.layout.mesh_offsets[axis_name]
        rp_off = Literal(offsets["row_ptr"])
        ci_off = Literal(offsets["col_ind"])
        w_off = Literal(offsets["weights"])

        state_names = extract_math_state_names(child)
        if not state_names:
            raise ValueError(f"Could not resolve a primary State target from MathExpr: {child}")
        s_off = Literal(self.layout.state_offsets[state_names[0]][0])
        idx_expr = idx_mgr.get_local(self.topo.get_base_axis(axis_name))

        bulk_div = UnstructuredRead(s_off, rp_off, ci_off, w_off, idx_expr)

        # Strip grad to extract conductivity multiplier
        multiplier_expr = self.lower_expr(self._strip_grad(child), idx_mgr, axis_name, current_eq=current_eq)
        res_ir = BinaryOp("*", multiplier_expr, bulk_div)

        bc_id = getattr(child, "bc_id", None)
        if bc_id:
            for s_face in ["left", "right", "top", "bottom"]:
                if s_face in offsets.get("surfaces", {}) and self.semantic_ctx.get_neumann_bc(bc_id, s_face):
                    bc_ast = self.semantic_ctx.get_neumann_bc(bc_id, s_face)["ast"]
                    bc_expr = self._lower_ast_dict(bc_ast)
                    bc_val_ir = self.lower_expr(bc_expr, idx_mgr, axis_name)
                    mask_ir = ArrayAccess("m", BinaryOp("+", Literal(offsets["surfaces"][s_face]), idx_expr))

                    if "volumes" in offsets:
                        vol_ir = FuncCall("std::max", [Literal("1e-30"), ArrayAccess("m", BinaryOp("+", Literal(offsets["volumes"]), idx_expr))])
                        term_ir = BinaryOp("/", BinaryOp("*", bc_val_ir, mask_ir), vol_ir)
                    else:
                        term_ir = BinaryOp("*", bc_val_ir, mask_ir)
                    res_ir = BinaryOp("+", res_ir, term_ir)

        return res_ir

    def _strip_grad(self, expr: MathExpr) -> MathExpr:
        """Strips MathGrad from an expression to isolate multiplier coefficients."""
        if isinstance(expr, MathGrad):
            return MathScalar(1.0)
        if isinstance(expr, MathBinaryOp):
            return MathBinaryOp(expr.op, self._strip_grad(expr.left), self._strip_grad(expr.right))
        if isinstance(expr, MathUnaryOp):
            return MathUnaryOp(expr.op, self._strip_grad(expr.child))
        return expr

    def _harmonic_mean(self, a: Expr, b: Expr) -> Expr:
        abs_a = FuncCall("std::abs", [a])
        abs_b = FuncCall("std::abs", [b])
        base_num = BinaryOp("+", BinaryOp("*", a, abs_b), BinaryOp("*", b, abs_a))
        num = BinaryOp("+", base_num, BinaryOp("*", Literal("5e-31"), BinaryOp("+", a, b)))
        den = BinaryOp("+", BinaryOp("+", abs_a, abs_b), Literal("1e-30"))
        return BinaryOp("/", num, den)

    def _stitch_piecewise_fluxes(self, r_flux: Expr, l_flux: Expr, idx_mgr: IndexManager,
                                 axis_name: str, current_eq: Optional[MathEquation]) -> Tuple[Expr, Expr]:
        if current_eq and current_eq.is_piecewise and current_eq.current_region:
            reg = current_eq.current_region
            start, end = reg.start_idx, reg.end_idx
            b_axis = self._resolve_axis(axis_name)

            c_right = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(end - 1))
            c_left = BinaryOp("==", idx_mgr.get_local(b_axis), Literal(start))

            for r in (current_eq.regions or []):
                if r.start_idx == end and r.div_flux:
                    n_flux = self.lower_expr(r.div_flux, idx_mgr, axis_name, face="right", current_eq=current_eq)
                    r_flux = Ternary(c_right, self._harmonic_mean(r_flux, n_flux), r_flux)
                if r.end_idx == start and r.div_flux:
                    p_flux = self.lower_expr(r.div_flux, idx_mgr, axis_name, face="left", current_eq=current_eq)
                    l_flux = Ternary(c_left, self._harmonic_mean(l_flux, p_flux), l_flux)

        return r_flux, l_flux

    def _lower_integral(self, child: MathExpr, over_domain: str, idx_mgr: IndexManager) -> Expr:
        axes = self.topo.get_axes(over_domain)
        idx_new = idx_mgr.clone()
        int_id = id(child)

        loops = []
        vol_exprs = []
        for axis in axes:
            b_axis = self.topo.get_base_axis(axis)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            int_var = f"i_{int_id}_{axis}"

            loops.append((int_var, Literal(res)))
            idx_new.register(b_axis, BinaryOp("+", Var(int_var), Literal(start)))

            vol_exprs.append(self._get_integral_volume_weight(axis, b_axis, int_var, start))

        child_expr = self.lower_expr(child, idx_new, current_axis=axes[-1] if axes else None)
        return Reduction(loops, child_expr, vol_exprs)

    def _get_integral_volume_weight(self, axis: str, b_axis: str, int_var: str, start: int) -> Expr:
        resolved_axis = self._resolve_axis(b_axis or axis)
        coord_sys = self.topo.domains.get(resolved_axis, {}).get("coord_sys", "cartesian")
        if coord_sys == "unstructured":
            if resolved_axis in self.layout.mesh_offsets and "volumes" in self.layout.mesh_offsets[resolved_axis]:
                vol_off = self.layout.mesh_offsets[resolved_axis]["volumes"]
                return ArrayAccess("m", BinaryOp("+", Literal(vol_off), Var(int_var)))
            return Literal(1.0)

        dim_exp = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)
        vol_off = self.layout.mesh_offsets[resolved_axis]["w_V_nodes"]
        idx_expr = BinaryOp("+", Literal(vol_off + start), Var(int_var))
        m_val = ArrayAccess("m", idx_expr)

        l_phys = Var(f"L_phys_{resolved_axis}")
        scale = l_phys if dim_exp == 1.0 else FuncCall("std::pow", [l_phys, Literal(dim_exp)])
        return BinaryOp("*", m_val, scale)

    def _generate_ale_dilution(self, state_name: str, idx_mgr: IndexManager, current_axis: Optional[str]) -> List[Expr]:
        ale = []
        domain = getattr(self.state_map.get(state_name), "domain", None)
        if not domain:
            return ale

        coord_sys = getattr(domain, "coord_sys", "cartesian")
        dim_mult = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)

        for d_name, binding in self.semantic_ctx.dynamic_domains.items():
            if domain.name == d_name:
                rhs_ir = self._lower_ast_dict(binding["rhs"])
                L = self.lower_expr(rhs_ir, idx_mgr, current_axis)
                L_dot = self.lower_expr(MathDt(rhs_ir), idx_mgr, current_axis)

                y_curr = ArrayAccess("y", BinaryOp("+", Literal(self.layout.state_offsets[state_name][0]),
                                                   idx_mgr.get_flat_index(d_name)))
                div_v = BinaryOp("*", Literal(dim_mult), BinaryOp("/", L_dot, FuncCall("std::max", [Literal(1e-12), L])))
                ale.append(BinaryOp("*", UnaryMinus(y_curr), div_v))

        return ale

    def _lower_ast_dict(self, node: Any) -> MathExpr:
        """Helper to lower raw boundary AST nodes into MathExpr preserving _bc_id."""
        if not isinstance(node, dict):
            return MathScalar(float(node) if isinstance(node, (int, float)) else 0.0)

        bc_id = node.get("_bc_id")
        t = node.get("type")
        if t == "Scalar":
            return MathScalar(float(node["value"]), bc_id=bc_id)
        if t == "Parameter":
            p_name = node["name"]
            p_off = self.layout.get_param_offset(p_name) if self.layout else 0
            return MathParameter(name=p_name, offset=p_off, bc_id=bc_id)
        if t == "State":
            s_name = node["name"]
            s_obj = self.state_map.get(s_name)
            s_dom = getattr(s_obj, "domain", None)
            s_dom_name = s_dom.name if s_dom else None
            off, size = self.layout.state_offsets[s_name] if self.layout and s_name in self.layout.state_offsets else (0, 1)
            return MathState(name=s_name, domain_name=s_dom_name, offset=off, size=size, is_ydot=False, bc_id=bc_id)
        if t == "Boundary":
            child_ir = self._lower_ast_dict(node["child"])
            return MathBoundaryRef(child=child_ir, side=node["side"], domain=node.get("domain"), bc_id=bc_id)
        if t == "BinaryOp":
            return MathBinaryOp(node["op"], self._lower_ast_dict(node["left"]), self._lower_ast_dict(node["right"]), bc_id=bc_id)
        if t == "UnaryOp":
            op = node["op"]
            child = node["child"]
            if op == "coords":
                axis = self._resolve_axis(node.get("axis"))
                return MathCoords(axis=axis, bc_id=bc_id)
            if op == "grad":
                axis = self._resolve_axis(node.get("axis"))
                return MathGrad(child=self._lower_ast_dict(child), axis=axis, bc_id=bc_id)
            if op == "div":
                axis = self._resolve_axis(node.get("axis"))
                return MathDiv(child=self._lower_ast_dict(child), axis=axis, bc_id=bc_id)
            if op == "dt":
                return MathDt(child=self._lower_ast_dict(child), bc_id=bc_id)
            if op == "integral":
                return MathIntegral(child=self._lower_ast_dict(child), over_domain=node.get("over"), bc_id=bc_id)
            return MathUnaryOp(node["op"], self._lower_ast_dict(child), bc_id=bc_id)
        return MathScalar(0.0)