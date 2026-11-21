"""
Finite Volume Method (FVM) Discretizer.

Transforms continuum N-Dimensional Math IR (MathSystem) into 1D loop-level
Compute IR statements suitable for Ahead-Of-Time (AOT) C++ code generation.

Architecture Overview:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           FVMDiscretizer                                │
  │   Orchestrates equation loops, Dirichlet overrides, and diagnostics     │
  └───────┬──────────────┬──────────────┬──────────────┬──────────────┬─────┘
          │              │              │              │              │
          ▼              ▼              ▼              ▼              ▼
   ┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐
   │ MeshMetrics ││StateSampler ││  Spatial    ││  Piecewise  ││     ALE     │
   │  C-array    ││Face inter-  ││  Stencils   ││   Stitcher  ││ Kinematics  │
   │  geometry   ││polation & BC││ Grad & Div  ││Harmonic flux││Grid dilation│
   └─────────────┘└─────────────┘└─────────────┘└─────────────┘└─────────────┘
"""

from typing import Dict, Any, Optional, List, Tuple
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler._2_middle_end.semantics import SemanticContext
from ion_flux.compiler._3_backend.compute_ir import (
    Expr, Stmt, Literal, Var, ArrayAccess, BinaryOp, FuncCall, Ternary,
    UnaryMinus, VarDecl, Loop, IfElse, Assign, Comment, UnstructuredRead, Reduction
)
from ion_flux.compiler._3_backend.math_ir import (
    MathExpr, MathScalar, MathParameter, MathState, MathBinaryOp,
    MathUnaryOp, MathClamp, MathGrad, MathDiv, MathDt, MathCoords, MathIntegral,
    MathBoundaryRef, MathEquation, MathObservable, MathDirichletOverride,
    MathInitialCondition, MathNeumannBC, MathSystem, extract_math_state_names, extract_domain_name
)


# =============================================================================
# Helper Utilities & Index Linearization
# =============================================================================

def resolve_axis(topo: TopologyAnalyzer, axis_name: Optional[str]) -> Optional[str]:
    """Resolves an axis name (including composite cross-products) to its fundamental 1D base axis."""
    if not axis_name:
        return None
    axes = topo.get_axes(axis_name)
    if axes:
        return topo.get_base_axis(axes[-1])
    return topo.get_base_axis(axis_name)


def _simplify_idx_expr(expr: Expr) -> Expr:
    """Fold affine integer offsets: (x + c1) - c2 -> x + (c1 - c2)."""
    match expr:
        case BinaryOp(op="+" | "-" as op, left=l, right=r):
            sl = _simplify_idx_expr(l)
            sr = _simplify_idx_expr(r)
            if isinstance(sl, Literal) and isinstance(sr, Literal):
                try:
                    lv, rv = int(float(sl.val)), int(float(sr.val))
                    return Literal(lv + rv if op == "+" else lv - rv)
                except (TypeError, ValueError):
                    return BinaryOp(op, sl, sr)
            if isinstance(sr, Literal):
                try:
                    rv = int(float(sr.val))
                except (TypeError, ValueError):
                    return BinaryOp(op, sl, sr)
                if rv == 0:
                    return sl
                if op == "-" and rv < 0:
                    return _simplify_idx_expr(BinaryOp("+", sl, Literal(-rv)))
                if isinstance(sl, BinaryOp) and sl.op in ("+", "-") and isinstance(sl.right, Literal):
                    try:
                        c1 = int(float(sl.right.val)) * (1 if sl.op == "+" else -1)
                        c2 = rv * (1 if op == "+" else -1)
                        total = c1 + c2
                        if total == 0:
                            return sl.left
                        if total > 0:
                            return BinaryOp("+", sl.left, Literal(total))
                        return BinaryOp("-", sl.left, Literal(-total))
                    except (TypeError, ValueError):
                        pass
            return BinaryOp(op, sl, sr)
        case _:
            return expr


def _eval_idx_bounds(expr: Expr, var_bounds: Optional[Dict[str, Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """Computes conservative inclusive [min, max] integer bounds for an index expression."""
    match expr:
        case Literal(val=v):
            try:
                iv = int(float(v))
                return (iv, iv)
            except (TypeError, ValueError):
                return None
        case Var(name=n):
            return var_bounds.get(n) if var_bounds else None
        case BinaryOp(op="+", left=l, right=r):
            bl = _eval_idx_bounds(l, var_bounds)
            br = _eval_idx_bounds(r, var_bounds)
            return (bl[0] + br[0], bl[1] + br[1]) if (bl and br) else None
        case BinaryOp(op="-", left=l, right=r):
            bl = _eval_idx_bounds(l, var_bounds)
            br = _eval_idx_bounds(r, var_bounds)
            return (bl[0] - br[1], bl[1] - br[0]) if (bl and br) else None
        case BinaryOp(op="*", left=l, right=r):
            bl = _eval_idx_bounds(l, var_bounds)
            br = _eval_idx_bounds(r, var_bounds)
            if bl and br and bl[0] >= 0 and br[0] >= 0:
                return (bl[0] * br[0], bl[1] * br[1])
            return None
        case _:
            return None


def _clamp_or_direct(idx_expr: Expr, bound: int, var_bounds: Optional[Dict[str, Tuple[int, int]]]) -> Expr:
    clean = _simplify_idx_expr(idx_expr)
    b = _eval_idx_bounds(clean, var_bounds)
    if b is not None and 0 <= b[0] and b[1] < bound:
        return clean
    return FuncCall("CLAMP", [clean, Literal(bound)])


class IndexManager:
    """
    Manages active loop variable contexts and linearizes N-dimensional
    domain coordinates into flat, 1D contiguous memory strides.
    """

    def __init__(self, topo: TopologyAnalyzer):
        self.topo = topo
        self.active_indices: Dict[str, Expr] = {}
        self.var_bounds: Dict[str, Tuple[int, int]] = {}

    def register(self, axis: str, expr: Expr) -> None:
        """Binds a spatial coordinate axis to an active index expression."""
        base_axis = self.topo.get_base_axis(axis)
        self.active_indices[base_axis] = expr

    def get_local(self, axis: str) -> Expr:
        """Retrieves the active index expression for a given axis (defaults to 0)."""
        base_axis = self.topo.get_base_axis(axis)
        return self.active_indices.get(base_axis, Literal(0))

    def get_flat_index(self, domain_name: Optional[str]) -> Expr:
        """
        Computes the linearized 1D stride offset for a multidimensional domain:
          flat_idx = sum((local_idx[axis] - start_idx[axis]) * stride[axis])
        """
        if not domain_name:
            return Literal(0)

        axes = self.topo.get_axes(domain_name)
        strides = self.topo.get_strides(domain_name)

        terms: List[Expr] = []
        for axis in axes:
            base_axis = self.topo.get_base_axis(axis)
            abs_idx = self.active_indices.get(base_axis, Literal(0))

            start_idx = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)

            if start_idx == 0:
                local_idx = abs_idx
            elif isinstance(abs_idx, BinaryOp) and abs_idx.op == "+" and isinstance(abs_idx.right, Literal) and abs_idx.right.val == start_idx:
                local_idx = abs_idx.left
            elif isinstance(abs_idx, Literal):
                local_idx = Literal(int(abs_idx.val) - start_idx)
            else:
                local_idx = BinaryOp("-", abs_idx, Literal(start_idx))

            local_idx = _simplify_idx_expr(local_idx)
            bounds = _eval_idx_bounds(local_idx, self.var_bounds)
            if isinstance(local_idx, Literal):
                clamped_idx = Literal(max(0, min(int(local_idx.val), res - 1)))
            elif bounds is not None and 0 <= bounds[0] and bounds[1] < res:
                clamped_idx = local_idx
            elif isinstance(local_idx, Var) and local_idx.name == f"idx_{axis}":
                clamped_idx = local_idx
            else:
                clamped_idx = FuncCall("CLAMP", [local_idx, Literal(res)])

            stride = strides[axis]
            if stride > 1:
                terms.append(BinaryOp("*", clamped_idx, Literal(stride)))
            else:
                terms.append(clamped_idx)

        if not terms:
            return Literal(0)

        flat_expr = terms[0]
        for term in terms[1:]:
            flat_expr = BinaryOp("+", flat_expr, term)
        return _simplify_idx_expr(flat_expr)

    def clone(self) -> "IndexManager":
        """Creates an independent copy of active index bindings."""
        cloned = IndexManager(self.topo)
        cloned.active_indices = self.active_indices.copy()
        cloned.var_bounds = self.var_bounds.copy()
        return cloned


# =============================================================================
# Mesh Metrics: C-Array Geometry Pointer Math
# =============================================================================

class MeshMetrics:
    """
    Encapsulates raw C-array pointer arithmetic (`m[...]`) into physical
    finite-volume quantities: control volume sizes, face areas, and node spacings.
    """

    def __init__(self, layout: Any, topo: TopologyAnalyzer):
        self.layout = layout
        self.topo = topo

    def domain_length(self, base_axis: Optional[str]) -> Expr:
        """Returns the physical length variable for an axis (or default 1.0)."""
        return Var(f"L_phys_{base_axis}") if base_axis else Var("L_phys_default")

    def face_dx(self, base_axis: str, face_idx: Expr, resolution: int, var_bounds: Optional[Dict[str, Tuple[int, int]]] = None) -> Expr:
        """Returns the distance between cell centers across a given face."""
        off_w_dx = self.layout.mesh_offsets[base_axis]["w_dx_faces"]
        clamped_face = _clamp_or_direct(face_idx, max(resolution - 1, 1), var_bounds)
        w_dx = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_face), symbol=base_axis, field="w_dx_faces", base_offset=off_w_dx)
        l_phys = self.domain_length(base_axis)
        return FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", l_phys, w_dx)])

    def node_span_dx(self, base_axis: str, node_idx: Expr, resolution: int, var_bounds: Optional[Dict[str, Tuple[int, int]]] = None) -> Expr:
        """Returns the total distance spanning across the cell (dx_left + dx_right)."""
        off_w_dx = self.layout.mesh_offsets[base_axis]["w_dx_faces"]
        clamped_r = _clamp_or_direct(node_idx, max(resolution - 1, 1), var_bounds)
        clamped_l = _clamp_or_direct(BinaryOp("-", node_idx, Literal(1)), max(resolution - 1, 1), var_bounds)
        w_dx_r = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_r), symbol=base_axis, field="w_dx_faces", base_offset=off_w_dx)
        w_dx_l = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_l), symbol=base_axis, field="w_dx_faces", base_offset=off_w_dx)
        l_phys = self.domain_length(base_axis)
        return FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", l_phys, BinaryOp("+", w_dx_r, w_dx_l))])

    def face_area(self, base_axis: str, face_idx: Expr) -> Expr:
        """Returns the interfacial face area A_face at face_idx."""
        off_area = self.layout.mesh_offsets[base_axis]["w_A_faces"]
        return ArrayAccess("m", BinaryOp("+", Literal(off_area), _simplify_idx_expr(face_idx)), symbol=base_axis, field="w_A_faces", base_offset=off_area)

    def scaled_cell_volume(self, base_axis: str, node_idx: Expr) -> Expr:
        """Returns the control volume V_cell scaled by physical domain length."""
        off_vol = self.layout.mesh_offsets[base_axis]["w_V_nodes"]
        vol_node = ArrayAccess("m", BinaryOp("+", Literal(off_vol), node_idx), symbol=base_axis, field="w_V_nodes", base_offset=off_vol)
        l_phys = self.domain_length(base_axis)
        return FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", vol_node, l_phys)])

    def cell_coordinate(self, base_axis: str, node_idx: Expr) -> Expr:
        """Returns the continuous physical coordinate at node_idx."""
        centers_offset = self.layout.mesh_offsets[base_axis]["w_centers"]
        normalized_center = ArrayAccess("m", BinaryOp("+", Literal(centers_offset), node_idx), symbol=base_axis, field="w_centers", base_offset=centers_offset)
        bounds = self.topo.domains.get(base_axis, {}).get("bounds", (0.0, 1.0))
        l_phys = self.domain_length(base_axis)
        return BinaryOp("+", Literal(bounds[0]), BinaryOp("*", l_phys, normalized_center))


# =============================================================================
# State Sampling & Face Interpolation
# =============================================================================

class StateSampler:
    """
    Handles memory array sampling into y[] and ydot[], cell-to-face interpolation,
    and Dirichlet boundary condition replacements.
    """

    def __init__(self, layout: Any, topo: TopologyAnalyzer, dirichlet_bcs: Optional[Dict[str, Dict[str, MathExpr]]] = None):
        self.layout = layout
        self.topo = topo
        self.dirichlet_bcs = dirichlet_bcs or {}

    def sample_cell(self, node: MathState, idx_mgr: IndexManager, force_ydot: bool = False) -> Expr:
        """Directly samples a cell-centered state from y[] or ydot[]."""
        flat_idx = idx_mgr.get_flat_index(node.domain_name)
        array_name = "ydot" if (node.is_ydot or force_ydot) else "y"
        idx_expr = Literal(node.offset) if not node.domain_name else BinaryOp("+", Literal(node.offset), flat_idx)
        return ArrayAccess(array_name, idx_expr, symbol=node.name, base_offset=node.offset)

    def sample_face(
        self,
        node: MathState,
        idx_mgr: IndexManager,
        face: str,
        current_axis: str,
        lower_expr_fn: Any,
        force_ydot: bool = False
    ) -> Expr:
        """
        Interpolates cell-centered states to cell faces: (y[i] + y[i ± 1]) / 2.
        Intercepts boundary faces governed by explicit Dirichlet constraints.
        """
        base_access = self.sample_cell(node, idx_mgr, force_ydot)

        base_axis = resolve_axis(self.topo, current_axis)
        res = self.topo.domains.get(current_axis, {}).get("resolution", 1)
        start = self.topo.domains.get(current_axis, {}).get("start_idx", 0)

        idx_local = idx_mgr.get_local(base_axis)
        shifted_expr = BinaryOp("+", idx_local, Literal(1)) if face == "right" else BinaryOp("-", idx_local, Literal(1))
        idx_shifted = idx_mgr.clone()
        idx_shifted.register(base_axis, shifted_expr)

        neighbor_idx = idx_shifted.get_flat_index(node.domain_name)
        array_name = "ydot" if (node.is_ydot or force_ydot) else "y"
        n_idx_expr = Literal(node.offset) if not node.domain_name else BinaryOp("+", Literal(node.offset), neighbor_idx)
        neighbor_access = ArrayAccess(array_name, n_idx_expr, symbol=node.name, base_offset=node.offset)
        interpolated = BinaryOp("*", Literal(0.5), BinaryOp("+", base_access, neighbor_access))

        # Check for explicit Dirichlet boundary overrides applied at this face
        dirichlet_bcs = self.dirichlet_bcs.get(node.name)
        if dirichlet_bcs:
            local_idx = _simplify_idx_expr(BinaryOp("-", idx_mgr.get_local(base_axis), Literal(start)))
            if face == "left" and "left" in dirichlet_bcs:
                is_edge = BinaryOp("==", local_idx, Literal(0))
                boundary_ir = lower_expr_fn(dirichlet_bcs["left"], idx_mgr, current_axis)
                return Ternary(is_edge, boundary_ir, interpolated)

            if face == "right" and "right" in dirichlet_bcs:
                is_edge = BinaryOp("==", local_idx, Literal(res - 1))
                boundary_ir = lower_expr_fn(dirichlet_bcs["right"], idx_mgr, current_axis)
                return Ternary(is_edge, boundary_ir, interpolated)

        return interpolated


# =============================================================================
# Piecewise Interface Flux Stitching
# =============================================================================

class PiecewiseFluxStitcher:
    """
    Enforces flux conservation across piecewise regional interfaces by applying
    a continuous harmonic mean at adjacent boundary faces:
      N_harmonic = 2 * a * b / (|a| + |b|)
    """

    def __init__(self, topo: TopologyAnalyzer):
        self.topo = topo

    def harmonic_mean(self, flux_a: Expr, flux_b: Expr) -> Expr:
        """Emits a call to the stabilized harmonic mean helper for interfacial fluxes."""
        return FuncCall("ion_flux_harmonic", [flux_a, flux_b])

    def stitch_interfaces(
        self,
        right_flux: Expr,
        left_flux: Expr,
        idx_mgr: IndexManager,
        axis_name: str,
        current_eq: Optional[MathEquation],
        lower_expr_fn: Any
    ) -> Tuple[Expr, Expr]:
        """Blends neighboring regional fluxes if evaluating at an internal interface face."""
        if not (current_eq and current_eq.is_piecewise and current_eq.current_region):
            return right_flux, left_flux

        current_region = current_eq.current_region
        start_idx = current_region.start_idx
        end_idx = current_region.end_idx
        base_axis = self.topo.get_base_axis(axis_name)

        is_at_right_boundary = BinaryOp("==", idx_mgr.get_local(base_axis), Literal(end_idx - 1))
        is_at_left_boundary = BinaryOp("==", idx_mgr.get_local(base_axis), Literal(start_idx))

        for neighbor in (current_eq.regions or []):
            if neighbor.start_idx == end_idx and neighbor.div_flux:
                neighbor_flux = lower_expr_fn(neighbor.div_flux, idx_mgr, axis_name, face="right", current_eq=current_eq)
                right_flux = Ternary(is_at_right_boundary, self.harmonic_mean(right_flux, neighbor_flux), right_flux)

            if neighbor.end_idx == start_idx and neighbor.div_flux:
                neighbor_flux = lower_expr_fn(neighbor.div_flux, idx_mgr, axis_name, face="left", current_eq=current_eq)
                left_flux = Ternary(is_at_left_boundary, self.harmonic_mean(left_flux, neighbor_flux), left_flux)

        return right_flux, left_flux


# =============================================================================
# Spatial Calculus Stencils: Gradient & Divergence
# =============================================================================

class SpatialStencils:
    """
    Implements discrete finite-volume spatial operators for 1D slabs,
    cylindrical shells, spherical shells, and 3D unstructured CSR graphs.
    """

    def __init__(
        self,
        metrics: MeshMetrics,
        stitcher: PiecewiseFluxStitcher,
        topo: TopologyAnalyzer,
        layout: Any,
        neumann_bcs: Optional[Dict[str, Dict[str, MathNeumannBC]]] = None,
        discretizer: Optional[Any] = None,
    ):
        self.metrics = metrics
        self.stitcher = stitcher
        self.topo = topo
        self.layout = layout
        self.neumann_bcs = neumann_bcs or {}
        self.discretizer = discretizer

    def gradient(
        self,
        child: MathExpr,
        axis_name: Optional[str],
        idx_mgr: IndexManager,
        face: Optional[str],
        current_eq: Optional[MathEquation],
        lower_expr_fn: Any
    ) -> Expr:
        """
        Computes finite-volume spatial gradients:
          Face Gradient:   (c[i+1] - c[i]) / dx
          Center Gradient: (c[i+1] - c[i-1]) / (dx_left + dx_right)
        """
        base_axis = resolve_axis(self.topo, axis_name)
        coord_sys = self.topo.domains.get(base_axis, {}).get("coord_sys", "cartesian") if base_axis else "cartesian"

        if coord_sys == "unstructured":
            return Literal(0.0)

        res = self.topo.domains.get(base_axis, {}).get("resolution", 1) if base_axis else 1
        idx_expr = idx_mgr.get_local(base_axis) if base_axis else Literal(0)

        # Staggered Face Gradient
        if face in ("left", "right"):
            shifted_expr = BinaryOp("+", idx_expr, Literal(1)) if face == "right" else BinaryOp("-", idx_expr, Literal(1))
            idx_shifted = idx_mgr.clone()
            idx_shifted.register(base_axis, shifted_expr)

            c_shifted = lower_expr_fn(child, idx_shifted, axis_name, face=None, current_eq=current_eq)
            c_current = lower_expr_fn(child, idx_mgr, axis_name, face=None, current_eq=current_eq)

            face_idx = idx_expr if face == "right" else BinaryOp("-", idx_expr, Literal(1))
            dist_safe = self.metrics.face_dx(base_axis, face_idx, res, var_bounds=idx_mgr.var_bounds)

            if face == "right":
                return BinaryOp("/", BinaryOp("-", c_shifted, c_current), dist_safe)
            return BinaryOp("/", BinaryOp("-", c_current, c_shifted), dist_safe)

        # Centered Node Gradient
        idx_right = idx_mgr.clone()
        idx_left = idx_mgr.clone()
        idx_right.register(base_axis, BinaryOp("+", idx_expr, Literal(1)))
        idx_left.register(base_axis, BinaryOp("-", idx_expr, Literal(1)))

        val_right = lower_expr_fn(child, idx_right, axis_name, face=None, current_eq=current_eq)
        val_left = lower_expr_fn(child, idx_left, axis_name, face=None, current_eq=current_eq)
        dist_safe = self.metrics.node_span_dx(base_axis, idx_expr, res, var_bounds=idx_mgr.var_bounds)

        return BinaryOp("/", BinaryOp("-", val_right, val_left), dist_safe)

    def divergence(
        self,
        child: MathExpr,
        axis_name: Optional[str],
        idx_mgr: IndexManager,
        current_eq: Optional[MathEquation],
        lower_expr_fn: Any
    ) -> Expr:
        """
        Computes finite volume flux divergence:
          div(N) = (A_right * N_right - A_left * N_left) / V_node
        """
        base_axis = resolve_axis(self.topo, axis_name)
        coord_sys = self.topo.domains.get(base_axis, {}).get("coord_sys", "cartesian") if base_axis else "cartesian"

        if coord_sys == "unstructured":
            return self._unstructured_divergence(child, axis_name, idx_mgr, current_eq, lower_expr_fn)

        if self.discretizer is not None and self.discretizer._stmt_sink is not None:
            return self.discretizer._emit_divergence_stmts(
                child, axis_name, base_axis, idx_mgr, current_eq
            )

        right_flux = lower_expr_fn(child, idx_mgr, axis_name, face="right", current_eq=current_eq)
        left_flux = lower_expr_fn(child, idx_mgr, axis_name, face="left", current_eq=current_eq)

        right_flux, left_flux = self.stitcher.stitch_interfaces(
            right_flux, left_flux, idx_mgr, axis_name, current_eq, lower_expr_fn
        )

        idx_expr = idx_mgr.get_local(base_axis)
        area_left = self.metrics.face_area(base_axis, idx_expr)
        area_right = self.metrics.face_area(base_axis, BinaryOp("+", idx_expr, Literal(1)))
        vol_safe = self.metrics.scaled_cell_volume(base_axis, idx_expr)

        net_flux = BinaryOp("-", BinaryOp("*", area_right, right_flux), BinaryOp("*", area_left, left_flux))
        return BinaryOp("/", net_flux, vol_safe)

    def _unstructured_divergence(
        self,
        child: MathExpr,
        axis_name: str,
        idx_mgr: IndexManager,
        current_eq: Optional[MathEquation],
        lower_expr_fn: Any
    ) -> Expr:
        """Evaluates divergence on 3D unstructured CSR tetrahedral graphs with boundary flux injection."""
        offsets = self.layout.mesh_offsets[axis_name]
        rp_off = Literal(offsets["row_ptr"])
        ci_off = Literal(offsets["col_ind"])
        w_off = Literal(offsets["weights"])

        state_names = extract_math_state_names(child)
        if not state_names:
            raise ValueError(f"Could not resolve a primary State target from MathExpr: {child}")

        state_offset = Literal(self.layout.state_offsets[state_names[0]][0])
        idx_expr = idx_mgr.get_local(self.topo.get_base_axis(axis_name))

        bulk_div = UnstructuredRead(state_offset, rp_off, ci_off, w_off, idx_expr)
        multiplier = lower_expr_fn(self._strip_grad(child), idx_mgr, axis_name, current_eq=current_eq)
        result_ir = BinaryOp("*", multiplier, bulk_div)

        bc_id = getattr(child, "bc_id", None)
        if bc_id:
            bc_map = self.neumann_bcs.get(bc_id, {})
            for surface_tag in ("left", "right", "top", "bottom"):
                if surface_tag in offsets.get("surfaces", {}) and surface_tag in bc_map:
                    bc_val = lower_expr_fn(bc_map[surface_tag].expr, idx_mgr, axis_name)
                    mask = ArrayAccess("m", BinaryOp("+", Literal(offsets["surfaces"][surface_tag]), idx_expr))

                    if "volumes" in offsets:
                        vol = FuncCall(
                            "std::max",
                            [Literal("1e-30"), ArrayAccess("m", BinaryOp("+", Literal(offsets["volumes"]), idx_expr))]
                        )
                        flux_injection = BinaryOp("/", BinaryOp("*", bc_val, mask), vol)
                    else:
                        flux_injection = BinaryOp("*", bc_val, mask)

                    result_ir = BinaryOp("+", result_ir, flux_injection)

        return result_ir

    def _strip_grad(self, expr: MathExpr) -> MathExpr:
        """Strips MathGrad nodes from flux expressions for unstructured divergence assembly."""
        match expr:
            case MathGrad():
                return MathScalar(1.0)
            case MathBinaryOp(op=op, left=left, right=right):
                return MathBinaryOp(op, self._strip_grad(left), self._strip_grad(right))
            case MathUnaryOp(op=op, child=child):
                return MathUnaryOp(op, self._strip_grad(child))
            case _:
                return expr


# =============================================================================
# Grid Kinematics (ALE) & Integral Reductions
# =============================================================================

class AleKinematics:
    """
    Computes grid deformation velocities and geometric dilution terms for
    Arbitrary Lagrangian-Eulerian (ALE) moving mesh problems:
      dilution = -y * (dim_mult * L_dot / L)
    """

    def __init__(self, dynamic_domain_bindings: Dict[str, Dict[str, Any]], state_map: Dict[str, Any], layout: Any):
        self.dynamic_domain_bindings = dynamic_domain_bindings
        self.state_map = state_map
        self.layout = layout

    def generate_dilution_terms(
        self,
        state_name: str,
        idx_mgr: IndexManager,
        current_axis: Optional[str],
        lower_expr_fn: Any,
    ) -> List[Expr]:
        """Calculates ALE dilution terms if the state's domain is dynamically bound to a moving boundary."""
        dilution_terms: List[Expr] = []
        domain = getattr(self.state_map.get(state_name), "domain", None)
        if not domain:
            return dilution_terms

        coord_sys = getattr(domain, "coord_sys", "cartesian")
        dim_multiplier = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)

        for domain_name, binding in self.dynamic_domain_bindings.items():
            if domain.name == domain_name:
                rhs_math_ir = binding["rhs_expr"]
                length_expr = lower_expr_fn(rhs_math_ir, idx_mgr, current_axis)
                length_velocity_expr = lower_expr_fn(MathDt(rhs_math_ir), idx_mgr, current_axis)

                state_offset = self.layout.state_offsets[state_name][0]
                flat_idx = idx_mgr.get_flat_index(domain_name)
                y_access = ArrayAccess("y", BinaryOp("+", Literal(state_offset), flat_idx), symbol=state_name, base_offset=state_offset)

                safe_length = FuncCall("std::max", [Literal(1e-12), length_expr])
                divergence_v = BinaryOp("*", Literal(dim_multiplier), BinaryOp("/", length_velocity_expr, safe_length))

                dilution_terms.append(BinaryOp("*", UnaryMinus(y_access), divergence_v))

        return dilution_terms


class IntegralQuadrature:
    """Lowers multidimensional integral reductions into discrete control-volume summations."""

    def __init__(self, metrics: MeshMetrics, topo: TopologyAnalyzer, layout: Any, discretizer: Optional[Any] = None):
        self.metrics = metrics
        self.topo = topo
        self.layout = layout
        self.discretizer = discretizer
        self._reduction_counter = 0

    def lower_integral(
        self,
        child: MathExpr,
        over_domain: str,
        idx_mgr: IndexManager,
        lower_expr_fn: Any
    ) -> Expr:
        """Constructs an unrolled discrete summation loop accumulating over control volumes: sum(c * dV)."""
        axes = self.topo.get_axes(over_domain)
        reduction_idx_mgr = idx_mgr.clone()
        integral_id = self._reduction_counter
        self._reduction_counter += 1

        loops: List[Tuple[str, Expr]] = []
        volume_exprs: List[Expr] = []

        for axis in axes:
            base_axis = self.topo.get_base_axis(axis)
            start_idx = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            iter_var = f"i_{integral_id}_{axis}"

            loops.append((iter_var, Literal(res)))
            reduction_idx_mgr.var_bounds[iter_var] = (0, max(res - 1, 0))
            reduction_idx_mgr.register(base_axis, BinaryOp("+", Var(iter_var), Literal(start_idx)))
            volume_exprs.append(self._get_volume_weight(axis, base_axis, iter_var, start_idx))

        saved_sink = self.discretizer._stmt_sink if self.discretizer is not None else None
        if self.discretizer is not None:
            self.discretizer._stmt_sink = None
        try:
            child_ir = lower_expr_fn(child, reduction_idx_mgr, current_axis=axes[-1] if axes else None)
        finally:
            if self.discretizer is not None:
                self.discretizer._stmt_sink = saved_sink
        return Reduction(loops, child_ir, volume_exprs)

    def _get_volume_weight(self, axis: str, base_axis: str, iter_var: str, start_idx: int) -> Expr:
        resolved_axis = resolve_axis(self.topo, base_axis or axis)
        coord_sys = self.topo.domains.get(resolved_axis, {}).get("coord_sys", "cartesian")

        if coord_sys == "unstructured":
            if resolved_axis in self.layout.mesh_offsets and "volumes" in self.layout.mesh_offsets[resolved_axis]:
                vol_offset = self.layout.mesh_offsets[resolved_axis]["volumes"]
                return ArrayAccess("m", BinaryOp("+", Literal(vol_offset), Var(iter_var)), symbol=resolved_axis, field="volumes", base_offset=vol_offset)
            return Literal(1.0)

        dim_exponent = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)
        vol_offset = self.layout.mesh_offsets[resolved_axis]["w_V_nodes"]
        idx_sum = Var(iter_var) if start_idx == 0 else BinaryOp("+", Literal(start_idx), Var(iter_var))
        volume_weight = ArrayAccess("m", BinaryOp("+", Literal(vol_offset), idx_sum), symbol=resolved_axis, field="w_V_nodes", base_offset=vol_offset)

        l_phys = self.metrics.domain_length(resolved_axis)
        scale_factor = l_phys if dim_exponent == 1.0 else FuncCall("std::pow", [l_phys, Literal(dim_exponent)])
        return BinaryOp("*", volume_weight, scale_factor)


# =============================================================================
# Primary FVM Discretizer Coordinator
# =============================================================================

class FVMDiscretizer:
    """
    Central compiler pass coordinating the discretization of Math IR into Compute IR.
    Delegates spatial stencils, state sampling, flux stitching, and ALE kinematics
    to specialized sub-components.
    """

    _BINARY_OPERATORS = {
        "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "std::pow",
        "gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="
    }

    _UNARY_OPERATORS = {
        "abs": "std::abs", "exp": "std::exp", "log": "std::log",
        "sin": "std::sin", "cos": "std::cos", "sqrt": "std::sqrt"
    }

    def __init__(
        self,
        layout: Any,
        topo: TopologyAnalyzer,
        semantic_ctx: Optional[SemanticContext] = None,
        state_map: Optional[Dict[str, Any]] = None,
        target: str = "cpu",
        math_sys: Optional[MathSystem] = None
    ):
        self.layout = layout
        self.topo = topo
        self.semantic_ctx = semantic_ctx
        self.state_map = state_map or {}
        self.target = target
        self.audit_compiler = False
        self._clamp_watchpoints: Dict[tuple, Dict[str, Any]] = {}
        self._cse_counter = 0
        self._stmt_sink: Optional[List[Stmt]] = None

        self.neumann_bcs = math_sys.neumann_bcs if math_sys else {}
        self.dirichlet_bcs = math_sys.dirichlet_bcs if math_sys else {}
        self.dynamic_domain_bindings = math_sys.dynamic_domain_bindings if math_sys else {}

        # Initialize modular subsystems
        self.metrics = MeshMetrics(layout, topo)
        self.sampler = StateSampler(layout, topo, self.dirichlet_bcs)
        self.stitcher = PiecewiseFluxStitcher(topo)
        self.stencils = SpatialStencils(self.metrics, self.stitcher, topo, layout, self.neumann_bcs, discretizer=self)
        self.ale = AleKinematics(self.dynamic_domain_bindings, self.state_map, layout)
        self.quadrature = IntegralQuadrature(self.metrics, topo, layout, discretizer=self)

    def _bind_math_system(self, math_sys: MathSystem) -> None:
        self.neumann_bcs = math_sys.neumann_bcs
        self.dirichlet_bcs = math_sys.dirichlet_bcs
        self.dynamic_domain_bindings = math_sys.dynamic_domain_bindings
        self.sampler.dirichlet_bcs = self.dirichlet_bcs
        self.stencils.neumann_bcs = self.neumann_bcs
        self.ale.dynamic_domain_bindings = self.dynamic_domain_bindings

    # -------------------------------------------------------------------------
    # Top-Level Entry Points
    # -------------------------------------------------------------------------

    def discretize_system(self, math_sys: MathSystem, audit_compiler: bool = False) -> Tuple[List[Stmt], List[Stmt], List[Stmt]]:
        """
        Translates a complete MathSystem into C++ Compute IR statement blocks.

        Returns:
            Tuple of (l_phys_stmts, equation_stmts, observable_stmts).
        """
        self.audit_compiler = audit_compiler
        self._clamp_watchpoints = {}
        self._cse_counter = 0
        self._bind_math_system(math_sys)
        l_phys_stmts = self._generate_physical_length_constants(math_sys)

        eq_stmts: List[Stmt] = []
        for eq in math_sys.equations:
            eq_stmts.extend(self.discretize_equation(eq))

        for override in math_sys.dirichlet_overrides:
            eq_stmts.extend(self.discretize_dirichlet_override(override))

        eq_stmts = self._deduplicate_top_level_reductions(eq_stmts)

        obs_stmts: List[Stmt] = []
        for obs in math_sys.observables:
            obs_stmts.extend(self.discretize_observable(obs))

        obs_stmts = self._deduplicate_top_level_reductions(obs_stmts)

        return l_phys_stmts, eq_stmts, obs_stmts

    def discretize_audit_observables(self) -> Tuple[List[Stmt], List[Dict[str, Any]]]:
        """Emits scalar lower/upper clamp saturation fraction observables into audit_obs[]."""
        if not self.audit_compiler or not self._clamp_watchpoints:
            return [], []

        stmts: List[Stmt] = []
        meta: List[Dict[str, Any]] = []
        for k, (_, wp) in enumerate(self._clamp_watchpoints.items()):
            target_domain = wp["target_domain"]
            bounds_override = wp["bounds_override"]
            axes = self.topo.get_axes(target_domain)
            base_axis = resolve_axis(self.topo, axes[-1]) if axes else None
            idx_mgr = self._build_index_manager(axes, bounds_override)

            raw_ir = self.lower_expr(wp["child"], idx_mgr, current_axis=base_axis)
            lo_ir = self.lower_expr(wp["lower"], idx_mgr, current_axis=base_axis)
            hi_ir = self.lower_expr(wp["upper"], idx_mgr, current_axis=base_axis)

            raw_var = f"_cl_raw_{k}"
            lo_var = f"_cl_lo_{k}"
            hi_var = f"_cl_hi_{k}"
            cnt_lo = f"_cl_hits_lo_{k}"
            cnt_hi = f"_cl_hits_hi_{k}"

            total_nodes = 1
            for ax in axes:
                b_ax = self.topo.get_base_axis(ax)
                res = self.topo.domains.get(ax, {}).get("resolution", 1)
                _, loop_res = bounds_override.get(b_ax, (0, res))
                total_nodes *= max(loop_res, 1)

            stmts.append(Comment(f"Clamp Watchpoint [{k}]: {wp['label']} ({wp['state_name']})", is_section=True))
            stmts.append(VarDecl("int", cnt_lo, Literal(0)))
            stmts.append(VarDecl("int", cnt_hi, Literal(0)))

            body: List[Stmt] = [
                VarDecl("const double", raw_var, raw_ir),
                VarDecl("const double", lo_var, lo_ir),
                VarDecl("const double", hi_var, hi_ir),
                Assign(Var(cnt_lo), BinaryOp("+", Var(cnt_lo), Ternary(BinaryOp("<", Var(raw_var), Var(lo_var)), Literal(1), Literal(0)))),
                Assign(Var(cnt_hi), BinaryOp("+", Var(cnt_hi), Ternary(BinaryOp(">", Var(raw_var), Var(hi_var)), Literal(1), Literal(0)))),
            ]
            stmts.extend(self._wrap_in_loop_nest(body, axes, bounds_override))
            stmts.append(Assign(ArrayAccess("audit_obs", Literal(2 * k)), BinaryOp("/", Var(f"(double){cnt_lo}"), Literal(float(total_nodes)))))
            stmts.append(Assign(ArrayAccess("audit_obs", Literal(2 * k + 1)), BinaryOp("/", Var(f"(double){cnt_hi}"), Literal(float(total_nodes)))))

            meta.append({
                "index": k,
                "lower_slot": 2 * k,
                "upper_slot": 2 * k + 1,
                "label": wp["label"],
                "state": wp["state_name"],
                "domain": wp["region_name"] or target_domain or "0D",
                "total_nodes": total_nodes,
            })

        return stmts, meta

    def discretize_initial_conditions(self, math_sys: MathSystem) -> Tuple[List[Stmt], List[Stmt]]:
        """Lowers initial conditions into 1D Compute IR assignments to y0[]."""
        self._bind_math_system(math_sys)
        l_phys_ic_stmts = self._generate_physical_length_constants(math_sys, include_dynamic=False)
        ic_stmts: List[Stmt] = []
        for ic in math_sys.initial_conditions:
            ic_stmts.extend(self._discretize_single_ic(ic))
        return l_phys_ic_stmts, ic_stmts

    def _discretize_single_ic(self, ic: MathInitialCondition) -> List[Stmt]:
        axes = self.topo.get_axes(ic.target_domain)
        base_axis = resolve_axis(self.topo, axes[-1]) if axes else None
        idx_mgr = self._build_index_manager(axes, {})

        offset = self.layout.state_offsets[ic.state_name][0]
        flat_idx = idx_mgr.get_flat_index(ic.target_domain)
        idx_expr = Literal(offset) if not ic.target_domain else BinaryOp("+", Literal(offset), flat_idx)
        y0_access = ArrayAccess("y0", idx_expr, symbol=ic.state_name, base_offset=offset)
        rhs_expr = self.lower_expr(ic.expr, idx_mgr, current_axis=base_axis)

        body = self._hoist_cse_and_reductions(Assign(y0_access, rhs_expr))
        return [Comment(f"IC: {ic.state_name}")] + self._wrap_in_loop_nest(body, axes, {})

    def discretize_equation(self, eq: MathEquation) -> List[Stmt]:
        """Lowers an implicit MathEquation into a nested coordinate loop assigning to res[]."""
        axes = self.topo.get_axes(eq.target_domain)
        bounds_override = eq.bounds_override or {}
        base_axis = resolve_axis(self.topo, axes[-1]) if axes else None

        idx_mgr = self._build_index_manager(axes, bounds_override)

        offset = self.layout.state_offsets[eq.state_name][0]
        flat_idx = idx_mgr.get_flat_index(eq.target_domain)
        idx_expr = Literal(offset) if not eq.target_domain else BinaryOp("+", Literal(offset), flat_idx)
        res_access = ArrayAccess("res", idx_expr, symbol=eq.state_name, base_offset=offset)

        self._active_eq_ctx = {
            "state_name": eq.state_name,
            "target_domain": eq.target_domain,
            "bounds_override": bounds_override,
            "region_name": eq.current_region.domain_name if (eq.is_piecewise and eq.current_region) else None,
        }

        sink: List[Stmt] = []
        self._stmt_sink = sink
        try:
            lhs_expr = self.lower_expr(eq.lhs, idx_mgr, current_axis=base_axis, current_eq=eq)
            rhs_expr = self.lower_expr(eq.rhs, idx_mgr, current_axis=base_axis, current_eq=eq)

            # Inject ALE grid velocity dilution terms if domain boundary is dynamic
            for ale_term in self.ale.generate_dilution_terms(
                eq.state_name, idx_mgr, base_axis, self.lower_expr
            ):
                rhs_expr = BinaryOp("+", rhs_expr, ale_term)
        finally:
            self._stmt_sink = None
            self._active_eq_ctx = None

        # Implicit residual convention: res = lhs - rhs (drop redundant - 0.0)
        if isinstance(rhs_expr, Literal) and float(rhs_expr.val) == 0.0:
            res_expr = lhs_expr
        else:
            res_expr = BinaryOp("-", lhs_expr, rhs_expr)
        assignment = Assign(res_access, res_expr)
        body = sink + self._hoist_cse_and_reductions(assignment)
        reg_tag = f", region={eq.current_region.domain_name}" if (eq.is_piecewise and eq.current_region) else ""
        header = Comment(f"Eq: {eq.state_name} (domain={eq.target_domain or '0D'}{reg_tag})", is_section=True)
        return [header] + self._wrap_in_loop_nest(body, axes, bounds_override)

    def discretize_dirichlet_override(self, override: MathDirichletOverride) -> List[Stmt]:
        """Directly enforces Dirichlet state constraints: res[boundary_idx] = y - y_boundary."""
        state_obj = self.state_map.get(override.state_name)
        domain = getattr(state_obj, "domain", None)
        target_domain_name = domain.name if domain else None
        last_axis = self.topo.get_axes(target_domain_name)[-1] if target_domain_name else None
        base_axis = resolve_axis(self.topo, last_axis) if last_axis else None

        if last_axis:
            res = self.topo.domains.get(last_axis, {}).get("resolution", 1)
            start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
        else:
            res = 1
            start = 0

        target_idx = start if override.side == "left" else start + res - 1

        idx_mgr = IndexManager(self.topo)
        if base_axis:
            idx_mgr.register(base_axis, Literal(target_idx))

        offset = self.layout.state_offsets[override.state_name][0]
        flat_idx = idx_mgr.get_flat_index(target_domain_name)

        idx_expr = Literal(offset) if not target_domain_name else BinaryOp("+", Literal(offset), flat_idx)
        res_access = ArrayAccess("res", idx_expr, symbol=override.state_name, base_offset=offset)
        y_access = ArrayAccess("y", idx_expr, symbol=override.state_name, base_offset=offset)
        rhs_expr = self.lower_expr(override.value_expr, idx_mgr, current_axis=base_axis)

        return [
            Comment(f"Dirichlet BC: {override.state_name}.{override.side}"),
            Assign(res_access, BinaryOp("-", y_access, rhs_expr)),
        ]

    def discretize_observable(self, obs: MathObservable) -> List[Stmt]:
        """Lowers an observable diagnostic metric into an assignment to obs[]."""
        axes = self.topo.get_axes(obs.target_domain)
        bounds_override = obs.bounds_override or {}
        base_axis = resolve_axis(self.topo, axes[-1]) if axes else None

        idx_mgr = self._build_index_manager(axes, bounds_override)

        offset = self.layout.obs_offsets[obs.name][0]
        flat_idx = idx_mgr.get_flat_index(obs.target_domain)
        idx_expr = Literal(offset) if not obs.target_domain else BinaryOp("+", Literal(offset), flat_idx)
        obs_access = ArrayAccess("obs", idx_expr, symbol=obs.name, base_offset=offset)

        rhs_expr = self.lower_expr(obs.expr, idx_mgr, current_axis=base_axis)
        assignment = Assign(obs_access, rhs_expr)
        body = self._hoist_cse_and_reductions(assignment)

        return [Comment(f"Obs: {obs.name}", is_section=True)] + self._wrap_in_loop_nest(body, axes, bounds_override)

    # -------------------------------------------------------------------------
    # Expression Translation Pipeline
    # -------------------------------------------------------------------------

    def lower_expr(
        self,
        node: MathExpr,
        idx_mgr: IndexManager,
        current_axis: Optional[str] = None,
        face: Optional[str] = None,
        current_eq: Optional[MathEquation] = None
    ) -> Expr:
        """Translates a typed MathExpr into Compute IR, intercepting Neumann boundary tags."""
        if face and getattr(node, "bc_id", None):
            bc_info = self.neumann_bcs.get(node.bc_id, {}).get(face)
            if bc_info is not None:
                axis = resolve_axis(self.topo, bc_info.domain) or current_axis
                res = self.topo.domains.get(axis, {}).get("resolution", 1)
                start = self.topo.domains.get(axis, {}).get("start_idx", 0)
                base_axis = self.topo.get_base_axis(axis)

                edge_idx = start if face == "left" else start + res - 1
                cond = self._simplify_eq_cond(idx_mgr.get_local(base_axis), edge_idx, idx_mgr.var_bounds)
                if cond is False:
                    return self._dispatch(node, idx_mgr, current_axis, face, current_eq)
                bc_ir = self.lower_expr(bc_info.expr, idx_mgr, current_axis, face=None, current_eq=current_eq)
                if cond is True:
                    return bc_ir
                return Ternary(cond, bc_ir, self._dispatch(node, idx_mgr, current_axis, face, current_eq))

        return self._dispatch(node, idx_mgr, current_axis, face, current_eq)

    def _simplify_eq_cond(
        self,
        idx_expr: Expr,
        target_val: int,
        var_bounds: Optional[Dict[str, Tuple[int, int]]],
    ) -> Any:
        idx_s = _simplify_idx_expr(idx_expr)
        b = _eval_idx_bounds(idx_s, var_bounds)
        if b is not None:
            if target_val < b[0] or target_val > b[1]:
                return False
            if b[0] == b[1] == target_val:
                return True
        return BinaryOp("==", idx_s, Literal(target_val))

    def _emit_face_flux(
        self,
        child: MathExpr,
        axis_name: str,
        idx_mgr: IndexManager,
        face: str,
        current_eq: Optional[MathEquation],
        flux_var: str,
    ) -> List[Stmt]:
        saved_sink = self._stmt_sink
        self._stmt_sink = None
        try:
            bc_info = (
                self.neumann_bcs.get(child.bc_id, {}).get(face)
                if getattr(child, "bc_id", None) else None
            )

            neighbor = None
            if current_eq and current_eq.is_piecewise and current_eq.current_region:
                cr = current_eq.current_region
                for nb in (current_eq.regions or []):
                    if face == "right" and nb.start_idx == cr.end_idx and nb.div_flux:
                        neighbor = nb
                        break
                    if face == "left" and nb.end_idx == cr.start_idx and nb.div_flux:
                        neighbor = nb
                        break

            def cse_assign_stmts(target_var: str, expr: Expr) -> List[Stmt]:
                return self._hoist_cse_and_reductions(Assign(Var(target_var), expr))

            def cse_var_stmts(target_var: str, expr: Expr) -> List[Stmt]:
                stmts = self._hoist_cse_and_reductions(Assign(Var(target_var), expr))
                last = stmts[-1]
                if isinstance(last, Assign):
                    stmts[-1] = VarDecl("const double", target_var, last.rhs)
                return stmts

            if bc_info is not None:
                bc_axis = resolve_axis(self.topo, bc_info.domain) or axis_name
                res = self.topo.domains.get(bc_axis, {}).get("resolution", 1)
                start = self.topo.domains.get(bc_axis, {}).get("start_idx", 0)
                base_ax = self.topo.get_base_axis(bc_axis)
                edge_idx = start if face == "left" else start + res - 1
                cond = self._simplify_eq_cond(idx_mgr.get_local(base_ax), edge_idx, idx_mgr.var_bounds)
                if cond is True:
                    bc_ir = self.lower_expr(bc_info.expr, idx_mgr, axis_name, face=None, current_eq=current_eq)
                    return cse_var_stmts(flux_var, bc_ir)
                if cond is not False:
                    bc_ir = self.lower_expr(bc_info.expr, idx_mgr, axis_name, face=None, current_eq=current_eq)
                    bulk_ir = self._dispatch(child, idx_mgr, axis_name, face, current_eq)
                    return [
                        VarDecl("double", flux_var, Literal(0.0)),
                        IfElse(cond, cse_assign_stmts(flux_var, bc_ir), cse_assign_stmts(flux_var, bulk_ir)),
                    ]

            if neighbor is not None:
                cr = current_eq.current_region
                base_ax = self.topo.get_base_axis(axis_name)
                bnd_idx = (cr.end_idx - 1) if face == "right" else cr.start_idx
                bnd_cond = self._simplify_eq_cond(idx_mgr.get_local(base_ax), bnd_idx, idx_mgr.var_bounds)
                this_flux = self._dispatch(child, idx_mgr, axis_name, face, current_eq)
                if bnd_cond is False:
                    return cse_var_stmts(flux_var, this_flux)

                nb_flux = self.lower_expr(neighbor.div_flux, idx_mgr, axis_name, face=face, current_eq=current_eq)
                fa_var = f"_fa_{self._cse_counter}"
                fb_var = f"_fb_{self._cse_counter}"
                self._cse_counter += 1
                then_stmts = (
                    cse_var_stmts(fa_var, this_flux)
                    + cse_var_stmts(fb_var, nb_flux)
                    + [Assign(Var(flux_var), self.stitcher.harmonic_mean(Var(fa_var), Var(fb_var)))]
                )
                if bnd_cond is True:
                    return [VarDecl("double", flux_var, Literal(0.0))] + then_stmts
                return [
                    VarDecl("double", flux_var, Literal(0.0)),
                    IfElse(bnd_cond, then_stmts, cse_assign_stmts(flux_var, this_flux)),
                ]

            bulk_ir = self._dispatch(child, idx_mgr, axis_name, face, current_eq)
            return cse_var_stmts(flux_var, bulk_ir)
        finally:
            self._stmt_sink = saved_sink

    def _emit_divergence_stmts(
        self,
        child: MathExpr,
        axis_name: Optional[str],
        base_axis: Optional[str],
        idx_mgr: IndexManager,
        current_eq: Optional[MathEquation],
    ) -> Expr:
        div_id = self._cse_counter
        self._cse_counter += 1
        r_var = f"_flux_r_{div_id}"
        l_var = f"_flux_l_{div_id}"
        d_var = f"_div_{div_id}"

        stmts_r = self._emit_face_flux(child, axis_name or "", idx_mgr, "right", current_eq, r_var)
        stmts_l = self._emit_face_flux(child, axis_name or "", idx_mgr, "left", current_eq, l_var)
        self._stmt_sink.extend(stmts_r)
        self._stmt_sink.extend(stmts_l)

        idx_expr = idx_mgr.get_local(base_axis)
        area_left = self.metrics.face_area(base_axis, idx_expr)
        area_right = self.metrics.face_area(base_axis, BinaryOp("+", idx_expr, Literal(1)))
        vol_safe = self.metrics.scaled_cell_volume(base_axis, idx_expr)
        net_flux = BinaryOp("-", BinaryOp("*", area_right, Var(r_var)), BinaryOp("*", area_left, Var(l_var)))
        self._stmt_sink.append(VarDecl("const double", d_var, BinaryOp("/", net_flux, vol_safe)))
        return Var(d_var)

    def _dispatch(
        self,
        node: MathExpr,
        idx_mgr: IndexManager,
        current_axis: Optional[str],
        face: Optional[str],
        current_eq: Optional[MathEquation]
    ) -> Expr:
        """Pattern matches typed Math IR expressions to their discrete Compute IR nodes."""
        match node:
            case MathScalar(value=value):
                return Literal(value)

            case MathParameter(name=p_name, offset=offset):
                return ArrayAccess("p", Literal(offset), symbol=p_name, base_offset=offset)

            case MathState():
                if face and current_axis:
                    return self.sampler.sample_face(
                        node, idx_mgr, face, current_axis, self.lower_expr
                    )
                return self.sampler.sample_cell(node, idx_mgr)

            case MathBoundaryRef():
                return self._lower_boundary_ref(node, idx_mgr, current_axis)

            case MathBinaryOp(op=op, left=left, right=right):
                lhs_ir = self.lower_expr(left, idx_mgr, current_axis, face, current_eq)
                rhs_ir = self.lower_expr(right, idx_mgr, current_axis, face, current_eq)

                if op in ("max", "min"):
                    return FuncCall(f"std::{op}", [lhs_ir, rhs_ir])

                bin_op = (
                    BinaryOp(self._BINARY_OPERATORS[op], lhs_ir, rhs_ir)
                    if op != "pow" else
                    FuncCall("std::pow", [lhs_ir, rhs_ir])
                )

                if op in ("gt", "lt", "ge", "le", "eq", "ne"):
                    return Ternary(bin_op, Literal(1.0), Literal(0.0))
                return bin_op

            case MathUnaryOp(op="neg", child=child):
                child_ir = self.lower_expr(child, idx_mgr, current_axis, face, current_eq)
                return UnaryMinus(child_ir)

            case MathClamp(child=child, lower=lower, upper=upper, label=lbl):
                if self.audit_compiler and getattr(self, "_active_eq_ctx", None):
                    ctx = self._active_eq_ctx
                    reg_key = ctx["region_name"] or ctx["target_domain"] or "0D"
                    wp_key = (lbl or f"clamp_{len(self._clamp_watchpoints)}", reg_key, id(child))
                    if wp_key not in self._clamp_watchpoints:
                        self._clamp_watchpoints[wp_key] = {
                            "label": lbl or f"clamp_{len(self._clamp_watchpoints)}",
                            "state_name": ctx["state_name"],
                            "target_domain": ctx["target_domain"],
                            "bounds_override": ctx["bounds_override"],
                            "region_name": ctx["region_name"],
                            "child": child,
                            "lower": lower,
                            "upper": upper,
                        }
                c_ir = self.lower_expr(child, idx_mgr, current_axis, face, current_eq)
                lo_ir = self.lower_expr(lower, idx_mgr, current_axis, face, current_eq)
                hi_ir = self.lower_expr(upper, idx_mgr, current_axis, face, current_eq)
                return FuncCall("std::min", [FuncCall("std::max", [c_ir, lo_ir]), hi_ir])

            case MathUnaryOp(op=op, child=child):
                child_ir = self.lower_expr(child, idx_mgr, current_axis, face, current_eq)
                func_name = self._UNARY_OPERATORS.get(op, op)
                return FuncCall(func_name, [child_ir])

            case MathDt(child=MathState() as state):
                return self.sampler.sample_cell(state, idx_mgr, force_ydot=True)

            case MathDt(child=child):
                return self.lower_expr(child, idx_mgr, current_axis, face, current_eq)

            case MathCoords(axis=axis):
                base_axis = resolve_axis(self.topo, axis or current_axis)
                if not base_axis or self.topo.domains.get(base_axis, {}).get("coord_sys") == "unstructured":
                    return Literal(0.0)
                return self.metrics.cell_coordinate(base_axis, idx_mgr.get_local(base_axis))

            case MathGrad(child=child, axis=axis):
                return self.stencils.gradient(
                    child, axis or current_axis, idx_mgr, face, current_eq, self.lower_expr
                )

            case MathDiv(child=child, axis=axis):
                return self.stencils.divergence(
                    child, axis or current_axis, idx_mgr, current_eq, self.lower_expr
                )

            case MathIntegral(child=child, over_domain=over_domain):
                return self.quadrature.lower_integral(
                    child, over_domain, idx_mgr, self.lower_expr
                )

            case _:
                return Literal(0.0)

    # -------------------------------------------------------------------------
    # Private Helpers & Legacy AST Dict Fallback
    # -------------------------------------------------------------------------

    def _lower_boundary_ref(
        self,
        node: MathBoundaryRef,
        idx_mgr: IndexManager,
        current_axis: Optional[str]
    ) -> Expr:
        """Evaluates an expression pinned to a domain boundary face."""
        cloned_idx_mgr = idx_mgr.clone()
        domain_name = node.domain or extract_domain_name(node.child)

        if domain_name:
            base_axis = resolve_axis(self.topo, domain_name)
            start = self.topo.domains.get(base_axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(base_axis, {}).get("resolution", 1)
            pinned_idx = start if node.side == "left" else start + res - 1
            cloned_idx_mgr.register(base_axis, Literal(pinned_idx))

        return self.lower_expr(node.child, cloned_idx_mgr, current_axis=domain_name or current_axis)

    def _generate_physical_length_constants(self, math_sys: MathSystem, include_dynamic: bool = True) -> List[Stmt]:
        """Declares local C++ physical domain lengths (L_phys_*), handling dynamic ALE moving boundaries."""
        l_phys_stmts: List[Stmt] = [VarDecl("double", "L_phys_default", Literal(1.0))]

        for domain_name, domain_info in self.topo.domains.items():
            if domain_info.get("type") == "composite":
                continue

            if include_dynamic and domain_name in math_sys.dynamic_domain_bindings:
                binding = math_sys.dynamic_domain_bindings[domain_name]
                idx_mgr = IndexManager(self.topo)
                idx_mgr.register(self.topo.get_base_axis(domain_name), Literal(0))
                rhs_ir = self.lower_expr(binding["rhs_expr"], idx_mgr, current_axis=domain_name)
                l_phys_stmts.append(
                    VarDecl("double", f"L_phys_{domain_name}", FuncCall("std::max", [Literal("1e-12"), rhs_ir]))
                )
            else:
                bounds = domain_info.get("bounds", (0.0, 1.0))
                l_phys_stmts.append(
                    VarDecl("double", f"L_phys_{domain_name}", Literal(float(bounds[1] - bounds[0])))
                )

        return l_phys_stmts

    def _build_index_manager(self, axes: List[str], bounds_override: Dict[str, tuple]) -> IndexManager:
        """Instantiates an IndexManager pre-registered with active coordinate loop variables."""
        idx_mgr = IndexManager(self.topo)
        for axis in axes:
            base_axis = self.topo.get_base_axis(axis)
            start_idx = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            loop_start, loop_res = bounds_override.get(base_axis, (start_idx, res))
            loop_var = f"idx_{axis}"
            idx_mgr.var_bounds[loop_var] = (0, max(loop_res - 1, 0))
            idx_expr = Var(loop_var) if loop_start == 0 else BinaryOp("+", Var(loop_var), Literal(loop_start))
            idx_mgr.register(base_axis, idx_expr)
        return idx_mgr

    def _wrap_in_loop_nest(
        self,
        body: List[Stmt],
        axes: List[str],
        bounds_override: Dict[str, tuple]
    ) -> List[Stmt]:
        """Wraps Compute IR statements in nested coordinate loops, injecting OpenMP pragmas if targeted."""
        current_body = body
        for i, axis in reversed(list(enumerate(axes))):
            base_axis = self.topo.get_base_axis(axis)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            _, loop_res = bounds_override.get(base_axis, (0, res))
            loop_var = f"idx_{axis}"
            pragma = "#pragma omp parallel for" if i == 0 and loop_res > 50 and "omp" in self.target else ""
            current_body = [Loop(loop_var, Literal(0), Literal(loop_res), current_body, pragma)]
        return current_body

    def _hoist_cse_and_reductions(self, assign: Assign) -> List[Stmt]:
        """
        Hoists Reduction/UnstructuredRead loops and repeated unconditional Compute IR
        subexpressions within a loop body into sequential VarDecl temporaries.
        Stops at Ternary branches to preserve conditional short-circuiting.
        """
        hoisted: List[Stmt] = []

        def hoist_loops(node: Expr) -> Expr:
            match node:
                case Reduction(loops=loops, child_expr=child_expr, vol_exprs=vol_exprs):
                    pre_stmts, clean_child = self._cse_expr(child_expr)
                    red = Reduction(loops, clean_child, vol_exprs, pre_stmts=pre_stmts)
                    name = f"_red_{self._cse_counter}"
                    self._cse_counter += 1
                    hoisted.append(VarDecl("double", name, red))
                    return Var(name)
                case UnstructuredRead():
                    name = f"_red_{self._cse_counter}"
                    self._cse_counter += 1
                    hoisted.append(VarDecl("const double", name, node))
                    return Var(name)
                case Ternary(cond=c, true_val=tv, false_val=fv):
                    return Ternary(hoist_loops(c), tv, fv)
                case BinaryOp(op=op, left=l, right=r):
                    return BinaryOp(op, hoist_loops(l), hoist_loops(r))
                case UnaryMinus(expr=e):
                    return UnaryMinus(hoist_loops(e))
                case FuncCall(func=f, args=args):
                    return FuncCall(f, [hoist_loops(a) for a in args])
                case _:
                    return node

        rhs_no_lambdas = hoist_loops(assign.rhs)
        cse_stmts, final_rhs = self._cse_expr(rhs_no_lambdas)
        return hoisted + cse_stmts + [Assign(assign.lhs, final_rhs)]

    def _cse_expr(self, root: Expr) -> Tuple[List[Stmt], Expr]:
        """Deduplicates repeated unconditional subexpressions within an expression."""
        def expr_key(node: Expr) -> tuple:
            match node:
                case Literal(val=v):
                    return ("L", v)
                case Var(name=n):
                    return ("V", n)
                case ArrayAccess(array_name=arr, index=idx):
                    return ("A", arr, expr_key(idx))
                case BinaryOp(op=op, left=l, right=r):
                    return ("B", op, expr_key(l), expr_key(r))
                case UnaryMinus(expr=e):
                    return ("N", expr_key(e))
                case FuncCall(func=f, args=args):
                    return ("F", f, tuple(expr_key(a) for a in args))
                case Ternary(cond=c, true_val=tv, false_val=fv):
                    return ("T", expr_key(c), expr_key(tv), expr_key(fv))
                case _:
                    return ("O", id(node))

        def node_size(node: Expr) -> int:
            match node:
                case Literal() | Var():
                    return 1
                case ArrayAccess(index=idx):
                    return 1 + node_size(idx)
                case UnaryMinus(expr=e):
                    return 1 + node_size(e)
                case BinaryOp(left=l, right=r):
                    return 1 + node_size(l) + node_size(r)
                case FuncCall(args=args):
                    return 1 + sum(node_size(a) for a in args)
                case _:
                    return 1

        counts: Dict[tuple, int] = {}
        samples: Dict[tuple, Expr] = {}

        def count_unconditional(node: Expr) -> None:
            if isinstance(node, (BinaryOp, FuncCall)) and node_size(node) >= 5:
                k = expr_key(node)
                counts[k] = counts.get(k, 0) + 1
                if k not in samples:
                    samples[k] = node
            match node:
                case BinaryOp(left=l, right=r):
                    count_unconditional(l)
                    count_unconditional(r)
                case UnaryMinus(expr=e):
                    count_unconditional(e)
                case FuncCall(args=args):
                    for a in args:
                        count_unconditional(a)
                case Ternary(cond=c):
                    count_unconditional(c)

        count_unconditional(root)

        ancestor_max: Dict[tuple, int] = {}

        def propagate_dominance(node: Expr, max_hoisted_above: int) -> None:
            k = expr_key(node) if isinstance(node, (BinaryOp, FuncCall)) and node_size(node) >= 5 else None
            if k is not None:
                ancestor_max[k] = max(ancestor_max.get(k, 0), max_hoisted_above)
                cnt = counts.get(k, 0)
                if cnt >= 2 and cnt > max_hoisted_above:
                    max_hoisted_above = cnt
            match node:
                case BinaryOp(left=l, right=r):
                    propagate_dominance(l, max_hoisted_above)
                    propagate_dominance(r, max_hoisted_above)
                case UnaryMinus(expr=e):
                    propagate_dominance(e, max_hoisted_above)
                case FuncCall(args=args):
                    for a in args:
                        propagate_dominance(a, max_hoisted_above)
                case Ternary(cond=c):
                    propagate_dominance(c, max_hoisted_above)

        propagate_dominance(root, 0)

        cse_candidates = {k for k, c in counts.items() if c >= 2 and c > ancestor_max.get(k, 0)}
        if not cse_candidates:
            return [], root

        hoisted: List[Stmt] = []
        cse_aliases: Dict[tuple, Var] = {}

        def rewrite(node: Expr, in_ternary_branch: bool = False) -> Expr:
            if not in_ternary_branch:
                k = expr_key(node)
                if k in cse_aliases:
                    return cse_aliases[k]

            new_node = node
            match node:
                case BinaryOp(op=op, left=l, right=r):
                    new_node = BinaryOp(op, rewrite(l, in_ternary_branch), rewrite(r, in_ternary_branch))
                case UnaryMinus(expr=e):
                    new_node = UnaryMinus(rewrite(e, in_ternary_branch))
                case FuncCall(func=f, args=args):
                    new_node = FuncCall(f, [rewrite(a, in_ternary_branch) for a in args])
                case Ternary(cond=c, true_val=tv, false_val=fv):
                    new_node = Ternary(
                        rewrite(c, in_ternary_branch),
                        rewrite(tv, True),
                        rewrite(fv, True),
                    )

            if not in_ternary_branch:
                k = expr_key(node)
                if k in cse_candidates and k not in cse_aliases:
                    vname = f"_cse_{self._cse_counter}"
                    self._cse_counter += 1
                    hoisted.append(VarDecl("const double", vname, new_node))
                    vref = Var(vname)
                    cse_aliases[k] = vref
                    return vref
            return new_node

        final_rhs = rewrite(root, False)
        return hoisted, final_rhs

    def _deduplicate_top_level_reductions(self, stmts: List[Stmt]) -> List[Stmt]:
        """Reuses identical top-level (0D) Reduction loops across equations or observables."""
        def norm_key(node: Expr, loop_map: Dict[str, str]) -> tuple:
            match node:
                case Literal(val=v):
                    return ("L", v)
                case Var(name=n):
                    return ("V", loop_map.get(n, n))
                case ArrayAccess(array_name=arr, index=idx):
                    return ("A", arr, norm_key(idx, loop_map))
                case BinaryOp(op=op, left=l, right=r):
                    return ("B", op, norm_key(l, loop_map), norm_key(r, loop_map))
                case UnaryMinus(expr=e):
                    return ("N", norm_key(e, loop_map))
                case FuncCall(func=f, args=args):
                    return ("F", f, tuple(norm_key(a, loop_map) for a in args))
                case Ternary(cond=c, true_val=tv, false_val=fv):
                    return ("T", norm_key(c, loop_map), norm_key(tv, loop_map), norm_key(fv, loop_map))
                case _:
                    return ("O", id(node))

        def reduction_key(red: Reduction) -> tuple:
            lmap = {v: f"__lv_{i}" for i, (v, _) in enumerate(red.loops)}
            sub_map = dict(lmap)
            for ps in red.pre_stmts:
                if isinstance(ps, VarDecl):
                    sub_map[ps.name] = f"__ps_{norm_key(ps.init, sub_map)}"
            loops_k = tuple((norm_key(end, lmap)) for _, end in red.loops)
            vols_k = tuple(norm_key(v, lmap) for v in red.vol_exprs)
            return (loops_k, vols_k, norm_key(red.child_expr, sub_map))

        def subst_vars(expr: Expr, aliases: Dict[str, str]) -> Expr:
            if not aliases:
                return expr
            match expr:
                case Var(name=n) if n in aliases:
                    return Var(aliases[n])
                case ArrayAccess(array_name=a, index=idx, symbol=s, field=f, base_offset=bo):
                    return ArrayAccess(a, subst_vars(idx, aliases), symbol=s, field=f, base_offset=bo)
                case BinaryOp(op=op, left=l, right=r):
                    return BinaryOp(op, subst_vars(l, aliases), subst_vars(r, aliases))
                case UnaryMinus(expr=e):
                    return UnaryMinus(subst_vars(e, aliases))
                case FuncCall(func=f, args=args):
                    return FuncCall(f, [subst_vars(a, aliases) for a in args])
                case Ternary(cond=c, true_val=tv, false_val=fv):
                    return Ternary(subst_vars(c, aliases), subst_vars(tv, aliases), subst_vars(fv, aliases))
                case _:
                    return expr

        seen_reds: Dict[tuple, str] = {}
        var_aliases: Dict[str, str] = {}
        out: List[Stmt] = []
        for stmt in stmts:
            match stmt:
                case VarDecl(var_type=vt, name=name, init=Reduction() as red):
                    rk = reduction_key(red)
                    if rk in seen_reds:
                        var_aliases[name] = seen_reds[rk]
                    else:
                        seen_reds[rk] = name
                        out.append(stmt)
                case VarDecl(var_type=vt, name=name, init=init):
                    out.append(VarDecl(vt, name, subst_vars(init, var_aliases)))
                case Assign(lhs=lhs, rhs=rhs):
                    out.append(Assign(lhs, subst_vars(rhs, var_aliases)))
                case _:
                    out.append(stmt)
        return out