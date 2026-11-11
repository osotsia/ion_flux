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
from ion_flux.compiler._4_codegen.cpp_emitter import CppEmitter
from ion_flux.compiler._4_codegen.compute_ir import (
    Expr, Stmt, Literal, Var, ArrayAccess, BinaryOp, FuncCall, Ternary,
    UnaryMinus, Loop, Assign, RawCpp, UnstructuredRead, Reduction
)
from ion_flux.compiler._3_backend.math_ir import (
    MathExpr, MathScalar, MathParameter, MathState, MathBinaryOp,
    MathUnaryOp, MathGrad, MathDiv, MathDt, MathCoords, MathIntegral,
    MathBoundaryRef, MathEquation, MathObservable, MathDirichletOverride,
    MathSystem, extract_math_state_names
)


# =============================================================================
# Helper Utilities & Index Linearization
# =============================================================================

def extract_domain_name(expr: MathExpr) -> Optional[str]:
    """Recursively resolves the governing spatial domain identifier of an expression."""
    match expr:
        case MathState(domain_name=domain_name):
            return domain_name
        case MathBoundaryRef(domain=domain, child=child):
            return domain or extract_domain_name(child)
        case MathGrad(axis=axis, child=child) | MathDiv(axis=axis, child=child):
            return axis or extract_domain_name(child)
        case MathCoords(axis=axis):
            return axis
        case MathUnaryOp(child=child) | MathDt(child=child):
            return extract_domain_name(child)
        case MathBinaryOp(left=left, right=right):
            return extract_domain_name(left) or extract_domain_name(right)
        case _:
            return None


class IndexManager:
    """
    Manages active loop variable contexts and linearizes N-dimensional
    domain coordinates into flat, 1D contiguous memory strides.
    """

    def __init__(self, topo: TopologyAnalyzer):
        self.topo = topo
        self.active_indices: Dict[str, Expr] = {}

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

            local_idx = BinaryOp("-", abs_idx, Literal(start_idx))
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
        return flat_expr

    def clone(self) -> "IndexManager":
        """Creates an independent copy of active index bindings."""
        cloned = IndexManager(self.topo)
        cloned.active_indices = self.active_indices.copy()
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

    def face_dx(self, base_axis: str, face_idx: Expr, resolution: int) -> Expr:
        """Returns the distance between cell centers across a given face."""
        off_w_dx = self.layout.mesh_offsets[base_axis]["w_dx_faces"]
        clamped_face = FuncCall("CLAMP", [face_idx, Literal(max(resolution - 1, 1))])
        w_dx = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_face))
        l_phys = self.domain_length(base_axis)
        return FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", l_phys, w_dx)])

    def node_span_dx(self, base_axis: str, node_idx: Expr, resolution: int) -> Expr:
        """Returns the total distance spanning across the cell (dx_left + dx_right)."""
        off_w_dx = self.layout.mesh_offsets[base_axis]["w_dx_faces"]
        clamped_r = FuncCall("CLAMP", [node_idx, Literal(max(resolution - 1, 1))])
        clamped_l = FuncCall("CLAMP", [BinaryOp("-", node_idx, Literal(1)), Literal(max(resolution - 1, 1))])
        w_dx_r = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_r))
        w_dx_l = ArrayAccess("m", BinaryOp("+", Literal(off_w_dx), clamped_l))
        l_phys = self.domain_length(base_axis)
        return FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", l_phys, BinaryOp("+", w_dx_r, w_dx_l))])

    def face_area(self, base_axis: str, face_idx: Expr) -> Expr:
        """Returns the interfacial face area A_face at face_idx."""
        off_area = self.layout.mesh_offsets[base_axis]["w_A_faces"]
        return ArrayAccess("m", BinaryOp("+", Literal(off_area), face_idx))

    def scaled_cell_volume(self, base_axis: str, node_idx: Expr) -> Expr:
        """Returns the control volume V_cell scaled by physical domain length."""
        off_vol = self.layout.mesh_offsets[base_axis]["w_V_nodes"]
        vol_node = ArrayAccess("m", BinaryOp("+", Literal(off_vol), node_idx))
        l_phys = self.domain_length(base_axis)
        return FuncCall("std::max", [Literal("1e-30"), BinaryOp("*", vol_node, l_phys)])

    def cell_coordinate(self, base_axis: str, node_idx: Expr) -> Expr:
        """Returns the continuous physical coordinate at node_idx."""
        centers_offset = self.layout.mesh_offsets[base_axis]["w_centers"]
        normalized_center = ArrayAccess("m", BinaryOp("+", Literal(centers_offset), node_idx))
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

    def __init__(self, layout: Any, topo: TopologyAnalyzer, semantic_ctx: SemanticContext):
        self.layout = layout
        self.topo = topo
        self.semantic_ctx = semantic_ctx

    def sample_cell(self, node: MathState, idx_mgr: IndexManager, force_ydot: bool = False) -> Expr:
        """Directly samples a cell-centered state from y[] or ydot[]."""
        flat_idx = idx_mgr.get_flat_index(node.domain_name)
        array_name = "ydot" if (node.is_ydot or force_ydot) else "y"
        return ArrayAccess(array_name, BinaryOp("+", Literal(node.offset), flat_idx))

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

        base_axis = self._resolve_axis(current_axis)
        res = self.topo.domains.get(current_axis, {}).get("resolution", 1)
        start = self.topo.domains.get(current_axis, {}).get("start_idx", 0)

        shift = 1 if face == "right" else -1
        idx_shifted = idx_mgr.clone()
        idx_shifted.register(base_axis, BinaryOp("+", idx_mgr.get_local(base_axis), Literal(shift)))

        neighbor_idx = idx_shifted.get_flat_index(node.domain_name)
        array_name = "ydot" if (node.is_ydot or force_ydot) else "y"
        neighbor_access = ArrayAccess(array_name, BinaryOp("+", Literal(node.offset), neighbor_idx))
        interpolated = BinaryOp("*", Literal(0.5), BinaryOp("+", base_access, neighbor_access))

        # Check for explicit Dirichlet boundary overrides applied at this face
        dirichlet_bcs = self.semantic_ctx.get_dirichlet_bc(node.name)
        if dirichlet_bcs:
            local_idx = BinaryOp("-", idx_mgr.get_local(base_axis), Literal(start))
            if face == "left" and "left" in dirichlet_bcs:
                is_edge = BinaryOp("==", local_idx, Literal(0))
                boundary_ir = lower_expr_fn(dirichlet_bcs["left"], idx_mgr, current_axis)
                return Ternary(is_edge, boundary_ir, interpolated)

            if face == "right" and "right" in dirichlet_bcs:
                is_edge = BinaryOp("==", local_idx, Literal(res - 1))
                boundary_ir = lower_expr_fn(dirichlet_bcs["right"], idx_mgr, current_axis)
                return Ternary(is_edge, boundary_ir, interpolated)

        return interpolated

    def _resolve_axis(self, axis_name: Optional[str]) -> Optional[str]:
        if not axis_name:
            return None
        axes = self.topo.get_axes(axis_name)
        if axes:
            return self.topo.get_base_axis(axes[-1])
        return self.topo.get_base_axis(axis_name)


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
        """Computes a smooth, stabilized harmonic mean of two interfacial fluxes."""
        abs_a = FuncCall("std::abs", [flux_a])
        abs_b = FuncCall("std::abs", [flux_b])
        base_num = BinaryOp("+", BinaryOp("*", flux_a, abs_b), BinaryOp("*", flux_b, abs_a))
        num = BinaryOp("+", base_num, BinaryOp("*", Literal("5e-31"), BinaryOp("+", flux_a, flux_b)))
        den = BinaryOp("+", BinaryOp("+", abs_a, abs_b), Literal("1e-30"))
        return BinaryOp("/", num, den)

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
        semantic_ctx: SemanticContext
    ):
        self.metrics = metrics
        self.stitcher = stitcher
        self.topo = topo
        self.layout = layout
        self.semantic_ctx = semantic_ctx

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
        base_axis = self._resolve_axis(axis_name)
        coord_sys = self.topo.domains.get(base_axis, {}).get("coord_sys", "cartesian") if base_axis else "cartesian"

        if coord_sys == "unstructured":
            return Literal(0.0)

        res = self.topo.domains.get(base_axis, {}).get("resolution", 1) if base_axis else 1
        idx_expr = idx_mgr.get_local(base_axis) if base_axis else Literal(0)

        # Staggered Face Gradient
        if face in ("left", "right"):
            shift = 1 if face == "right" else -1
            idx_shifted = idx_mgr.clone()
            idx_shifted.register(base_axis, BinaryOp("+", idx_expr, Literal(shift)))

            c_shifted = lower_expr_fn(child, idx_shifted, axis_name, face=None, current_eq=current_eq)
            c_current = lower_expr_fn(child, idx_mgr, axis_name, face=None, current_eq=current_eq)

            face_idx = idx_expr if face == "right" else BinaryOp("-", idx_expr, Literal(1))
            dist_safe = self.metrics.face_dx(base_axis, face_idx, res)

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
        dist_safe = self.metrics.node_span_dx(base_axis, idx_expr, res)

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
        base_axis = self._resolve_axis(axis_name)
        coord_sys = self.topo.domains.get(base_axis, {}).get("coord_sys", "cartesian") if base_axis else "cartesian"

        if coord_sys == "unstructured":
            return self._unstructured_divergence(child, axis_name, idx_mgr, current_eq, lower_expr_fn)

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
            for surface_tag in ("left", "right", "top", "bottom"):
                if surface_tag in offsets.get("surfaces", {}) and self.semantic_ctx.get_neumann_bc(bc_id, surface_tag):
                    bc_ast = self.semantic_ctx.get_neumann_bc(bc_id, surface_tag)["ast"]
                    bc_val = lower_expr_fn(bc_ast, idx_mgr, axis_name)
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

    def _resolve_axis(self, axis_name: Optional[str]) -> Optional[str]:
        if not axis_name:
            return None
        axes = self.topo.get_axes(axis_name)
        if axes:
            return self.topo.get_base_axis(axes[-1])
        return self.topo.get_base_axis(axis_name)


# =============================================================================
# Grid Kinematics (ALE) & Integral Reductions
# =============================================================================

class AleKinematics:
    """
    Computes grid deformation velocities and geometric dilution terms for
    Arbitrary Lagrangian-Eulerian (ALE) moving mesh problems:
      dilution = -y * (dim_mult * L_dot / L)
    """

    def __init__(self, semantic_ctx: SemanticContext, state_map: Dict[str, Any], layout: Any):
        self.semantic_ctx = semantic_ctx
        self.state_map = state_map
        self.layout = layout

    def generate_dilution_terms(
        self,
        state_name: str,
        idx_mgr: IndexManager,
        current_axis: Optional[str],
        lower_expr_fn: Any,
        lower_ast_dict_fn: Any
    ) -> List[Expr]:
        """Calculates ALE dilution terms if the state's domain is dynamically bound to a moving boundary."""
        dilution_terms: List[Expr] = []
        domain = getattr(self.state_map.get(state_name), "domain", None)
        if not domain:
            return dilution_terms

        coord_sys = getattr(domain, "coord_sys", "cartesian")
        dim_multiplier = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)

        for domain_name, binding in self.semantic_ctx.dynamic_domains.items():
            if domain.name == domain_name:
                rhs_math_ir = lower_ast_dict_fn(binding["rhs"])
                length_expr = lower_expr_fn(rhs_math_ir, idx_mgr, current_axis)
                length_velocity_expr = lower_expr_fn(MathDt(rhs_math_ir), idx_mgr, current_axis)

                state_offset = self.layout.state_offsets[state_name][0]
                flat_idx = idx_mgr.get_flat_index(domain_name)
                y_access = ArrayAccess("y", BinaryOp("+", Literal(state_offset), flat_idx))

                safe_length = FuncCall("std::max", [Literal(1e-12), length_expr])
                divergence_v = BinaryOp("*", Literal(dim_multiplier), BinaryOp("/", length_velocity_expr, safe_length))

                dilution_terms.append(BinaryOp("*", UnaryMinus(y_access), divergence_v))

        return dilution_terms


class IntegralQuadrature:
    """Lowers multidimensional integral reductions into discrete control-volume summations."""

    def __init__(self, metrics: MeshMetrics, topo: TopologyAnalyzer, layout: Any):
        self.metrics = metrics
        self.topo = topo
        self.layout = layout

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
        integral_id = id(child)

        loops: List[Tuple[str, Expr]] = []
        volume_exprs: List[Expr] = []

        for axis in axes:
            base_axis = self.topo.get_base_axis(axis)
            start_idx = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            iter_var = f"i_{integral_id}_{axis}"

            loops.append((iter_var, Literal(res)))
            reduction_idx_mgr.register(base_axis, BinaryOp("+", Var(iter_var), Literal(start_idx)))
            volume_exprs.append(self._get_volume_weight(axis, base_axis, iter_var, start_idx))

        child_ir = lower_expr_fn(child, reduction_idx_mgr, current_axis=axes[-1] if axes else None)
        return Reduction(loops, child_ir, volume_exprs)

    def _get_volume_weight(self, axis: str, base_axis: str, iter_var: str, start_idx: int) -> Expr:
        resolved_axis = self._resolve_axis(base_axis or axis)
        coord_sys = self.topo.domains.get(resolved_axis, {}).get("coord_sys", "cartesian")

        if coord_sys == "unstructured":
            if resolved_axis in self.layout.mesh_offsets and "volumes" in self.layout.mesh_offsets[resolved_axis]:
                vol_offset = self.layout.mesh_offsets[resolved_axis]["volumes"]
                return ArrayAccess("m", BinaryOp("+", Literal(vol_offset), Var(iter_var)))
            return Literal(1.0)

        dim_exponent = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)
        vol_offset = self.layout.mesh_offsets[resolved_axis]["w_V_nodes"]
        volume_weight = ArrayAccess("m", BinaryOp("+", Literal(vol_offset + start_idx), Var(iter_var)))

        l_phys = self.metrics.domain_length(resolved_axis)
        scale_factor = l_phys if dim_exponent == 1.0 else FuncCall("std::pow", [l_phys, Literal(dim_exponent)])
        return BinaryOp("*", volume_weight, scale_factor)

    def _resolve_axis(self, axis_name: Optional[str]) -> Optional[str]:
        if not axis_name:
            return None
        axes = self.topo.get_axes(axis_name)
        if axes:
            return self.topo.get_base_axis(axes[-1])
        return self.topo.get_base_axis(axis_name)


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
        semantic_ctx: SemanticContext,
        state_map: Dict[str, Any],
        target: str = "cpu"
    ):
        self.layout = layout
        self.topo = topo
        self.semantic_ctx = semantic_ctx
        self.state_map = state_map
        self.target = target

        # Initialize modular subsystems
        self.metrics = MeshMetrics(layout, topo)
        self.sampler = StateSampler(layout, topo, semantic_ctx)
        self.stitcher = PiecewiseFluxStitcher(topo)
        self.stencils = SpatialStencils(self.metrics, self.stitcher, topo, layout, semantic_ctx)
        self.ale = AleKinematics(semantic_ctx, state_map, layout)
        self.quadrature = IntegralQuadrature(self.metrics, topo, layout)
        self.emitter = CppEmitter()

    # -------------------------------------------------------------------------
    # Top-Level Entry Points
    # -------------------------------------------------------------------------

    def discretize_system(self, math_sys: MathSystem) -> Tuple[List[Stmt], List[Stmt], List[Stmt]]:
        """
        Translates a complete MathSystem into C++ Compute IR statement blocks.

        Returns:
            Tuple of (l_phys_stmts, equation_stmts, observable_stmts).
        """
        l_phys_stmts = self._generate_physical_length_constants(math_sys)

        eq_stmts: List[Stmt] = []
        for eq in math_sys.equations:
            eq_stmts.extend(self.discretize_equation(eq))

        for override in math_sys.dirichlet_overrides:
            eq_stmts.extend(self.discretize_dirichlet_override(override))

        obs_stmts: List[Stmt] = []
        for obs in math_sys.observables:
            obs_stmts.extend(self.discretize_observable(obs))

        return l_phys_stmts, eq_stmts, obs_stmts

    def discretize_equation(self, eq: MathEquation) -> List[Stmt]:
        """Lowers an implicit MathEquation into a nested coordinate loop assigning to res[]."""
        axes = self.topo.get_axes(eq.target_domain)
        bounds_override = eq.bounds_override or {}
        base_axis = self._resolve_axis(axes[-1]) if axes else None

        idx_mgr = self._build_index_manager(axes, bounds_override)

        offset = self.layout.state_offsets[eq.state_name][0]
        flat_idx = idx_mgr.get_flat_index(eq.target_domain)
        res_access = ArrayAccess("res", BinaryOp("+", Literal(offset), flat_idx))

        lhs_expr = self.lower_expr(eq.lhs, idx_mgr, current_axis=base_axis, current_eq=eq)
        rhs_expr = self.lower_expr(eq.rhs, idx_mgr, current_axis=base_axis, current_eq=eq)

        # Inject ALE grid velocity dilution terms if domain boundary is dynamic
        for ale_term in self.ale.generate_dilution_terms(
            eq.state_name, idx_mgr, base_axis, self.lower_expr, self._lower_ast_dict
        ):
            rhs_expr = BinaryOp("+", rhs_expr, ale_term)

        # Implicit residual convention: res = lhs - rhs
        assignment = Assign(res_access, BinaryOp("-", lhs_expr, rhs_expr))
        return self._wrap_in_loop_nest([assignment], axes, bounds_override)

    def discretize_dirichlet_override(self, override: MathDirichletOverride) -> List[Stmt]:
        """Directly enforces Dirichlet state constraints: res[boundary_idx] = y - y_boundary."""
        state_obj = self.state_map.get(override.state_name)
        domain = getattr(state_obj, "domain", None)
        target_domain_name = domain.name if domain else None
        last_axis = self.topo.get_axes(target_domain_name)[-1] if target_domain_name else None
        base_axis = self._resolve_axis(last_axis) if last_axis else None

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

        res_access = ArrayAccess("res", BinaryOp("+", Literal(offset), flat_idx))
        y_access = ArrayAccess("y", BinaryOp("+", Literal(offset), flat_idx))
        rhs_expr = self.lower_expr(override.value_expr, idx_mgr, current_axis=base_axis)

        return [Assign(res_access, BinaryOp("-", y_access, rhs_expr))]

    def discretize_observable(self, obs: MathObservable) -> List[Stmt]:
        """Lowers an observable diagnostic metric into an assignment to obs[]."""
        axes = self.topo.get_axes(obs.target_domain)
        bounds_override = obs.bounds_override or {}
        base_axis = self._resolve_axis(axes[-1]) if axes else None

        idx_mgr = self._build_index_manager(axes, bounds_override)

        offset = self.layout.obs_offsets[obs.name][0]
        flat_idx = idx_mgr.get_flat_index(obs.target_domain)
        obs_access = ArrayAccess("obs", BinaryOp("+", Literal(offset), flat_idx))

        rhs_expr = self.lower_expr(obs.expr, idx_mgr, current_axis=base_axis)
        assignment = Assign(obs_access, rhs_expr)

        return self._wrap_in_loop_nest([assignment], axes, bounds_override)

    # -------------------------------------------------------------------------
    # Expression Translation Pipeline
    # -------------------------------------------------------------------------

    def lower_expr(
        self,
        node: Any,
        idx_mgr: IndexManager,
        current_axis: Optional[str] = None,
        face: Optional[str] = None,
        current_eq: Optional[MathEquation] = None
    ) -> Expr:
        """Translates a MathExpr or raw AST dict into Compute IR, intercepting Neumann boundary tags."""
        # Unpack raw dictionary payloads if passing legacy AST fragments
        if isinstance(node, dict):
            node = self._lower_ast_dict(node)
        elif isinstance(node, MathDt) and isinstance(node.child, dict):
            node = MathDt(self._lower_ast_dict(node.child), bc_id=node.bc_id)

        # Intercept Neumann boundary faces via ternary operators: (is_edge ? bc_value : bulk_value)
        if face and getattr(node, "bc_id", None):
            bc_info = self.semantic_ctx.get_neumann_bc(node.bc_id, face)
            if bc_info:
                bc_expr = self._lower_ast_dict(bc_info["ast"])
                bc_ir = self.lower_expr(bc_expr, idx_mgr, current_axis, face=None, current_eq=current_eq)

                axis = self._resolve_axis(bc_info.get("domain")) or current_axis
                res = self.topo.domains.get(axis, {}).get("resolution", 1)
                start = self.topo.domains.get(axis, {}).get("start_idx", 0)
                base_axis = self.topo.get_base_axis(axis)

                edge_idx = start if face == "left" else start + res - 1
                is_edge = BinaryOp("==", idx_mgr.get_local(base_axis), Literal(edge_idx))

                return Ternary(is_edge, bc_ir, self._dispatch(node, idx_mgr, current_axis, face, current_eq))

        return self._dispatch(node, idx_mgr, current_axis, face, current_eq)

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

            case MathParameter(offset=offset):
                return ArrayAccess("p", Literal(offset))

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

            case MathUnaryOp(op="coords"):
                base_axis = self._resolve_axis(current_axis)
                if not base_axis or self.topo.domains.get(base_axis, {}).get("coord_sys") == "unstructured":
                    return Literal(0.0)
                return self.metrics.cell_coordinate(base_axis, idx_mgr.get_local(base_axis))

            case MathUnaryOp(op=op, child=child):
                child_ir = self.lower_expr(child, idx_mgr, current_axis, face, current_eq)
                func_name = self._UNARY_OPERATORS.get(op, op)
                return FuncCall(func_name, [child_ir])

            case MathDt(child=MathState() as state):
                return self.sampler.sample_cell(state, idx_mgr, force_ydot=True)

            case MathDt(child=child):
                return self.lower_expr(child, idx_mgr, current_axis, face, current_eq)

            case MathCoords(axis=axis):
                base_axis = self._resolve_axis(axis or current_axis)
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
            base_axis = self._resolve_axis(domain_name)
            start = self.topo.domains.get(base_axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(base_axis, {}).get("resolution", 1)
            pinned_idx = start if node.side == "left" else start + res - 1
            cloned_idx_mgr.register(base_axis, Literal(pinned_idx))

        return self.lower_expr(node.child, cloned_idx_mgr, current_axis=domain_name or current_axis)

    def _resolve_axis(self, axis_name: Optional[str]) -> Optional[str]:
        """Resolves an axis name (including composite cross-products) to its fundamental 1D base axis."""
        if not axis_name:
            return None
        axes = self.topo.get_axes(axis_name)
        if axes:
            return self.topo.get_base_axis(axes[-1])
        return self.topo.get_base_axis(axis_name)

    def _generate_physical_length_constants(self, math_sys: MathSystem) -> List[Stmt]:
        """Declares local C++ physical domain lengths (L_phys_*), handling dynamic ALE moving boundaries."""
        l_phys_stmts: List[Stmt] = [RawCpp("double L_phys_default = 1.0;")]

        for domain_name, domain_info in self.topo.domains.items():
            if domain_info.get("type") == "composite":
                continue

            if domain_name in math_sys.dynamic_domain_bindings:
                binding = math_sys.dynamic_domain_bindings[domain_name]
                idx_mgr = IndexManager(self.topo)
                idx_mgr.register(self.topo.get_base_axis(domain_name), Literal(0))
                rhs_ir = self.lower_expr(binding["rhs_expr"], idx_mgr, current_axis=domain_name)
                rhs_str = self.emitter.emit(rhs_ir)
                l_phys_stmts.append(
                    RawCpp(f"double L_phys_{domain_name} = std::max(1e-12, (double)({rhs_str}));")
                )
            else:
                bounds = domain_info.get("bounds", (0.0, 1.0))
                l_phys_stmts.append(
                    RawCpp(f"double L_phys_{domain_name} = {float(bounds[1] - bounds[0])};")
                )

        return l_phys_stmts

    def _build_index_manager(self, axes: List[str], bounds_override: Dict[str, tuple]) -> IndexManager:
        """Instantiates an IndexManager pre-registered with active coordinate loop variables."""
        idx_mgr = IndexManager(self.topo)
        for axis in axes:
            base_axis = self.topo.get_base_axis(axis)
            start_idx = self.topo.domains.get(axis, {}).get("start_idx", 0)
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            loop_start, _ = bounds_override.get(base_axis, (start_idx, res))
            loop_var = f"idx_{axis}"
            idx_mgr.register(base_axis, BinaryOp("+", Var(loop_var), Literal(loop_start)))
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

    def _lower_ast_dict(self, node: Any) -> MathExpr:
        """Fallback lowering to translate raw embedded frontend AST dictionaries into typed MathExpr nodes."""
        bc_id = node.get("_bc_id") if isinstance(node, dict) else None

        match node:
            case int() | float():
                return MathScalar(float(node))

            case {"type": "Scalar", "value": val}:
                return MathScalar(float(val), bc_id=bc_id)

            case {"type": "Parameter", "name": param_name}:
                offset = self.layout.get_param_offset(param_name) if self.layout else 0
                return MathParameter(name=param_name, offset=offset, bc_id=bc_id)

            case {"type": "State", "name": state_name}:
                state_obj = self.state_map.get(state_name)
                domain = getattr(state_obj, "domain", None)
                domain_name = domain.name if domain else None
                offset, size = (
                    self.layout.state_offsets[state_name]
                    if self.layout and state_name in self.layout.state_offsets
                    else (0, 1)
                )
                return MathState(
                    name=state_name, domain_name=domain_name,
                    offset=offset, size=size, is_ydot=False, bc_id=bc_id
                )

            case {"type": "Boundary", "child": child, "side": side}:
                return MathBoundaryRef(
                    child=self._lower_ast_dict(child),
                    side=side,
                    domain=node.get("domain"),
                    bc_id=bc_id
                )

            case {"type": "BinaryOp", "op": op, "left": left, "right": right}:
                return MathBinaryOp(
                    op=op,
                    left=self._lower_ast_dict(left),
                    right=self._lower_ast_dict(right),
                    bc_id=bc_id
                )

            case {"type": "UnaryOp", "op": "coords"}:
                axis = self._resolve_axis(node.get("axis"))
                return MathCoords(axis=axis, bc_id=bc_id)

            case {"type": "UnaryOp", "op": "grad", "child": child}:
                axis = self._resolve_axis(node.get("axis"))
                return MathGrad(child=self._lower_ast_dict(child), axis=axis, bc_id=bc_id)

            case {"type": "UnaryOp", "op": "div", "child": child}:
                axis = self._resolve_axis(node.get("axis"))
                return MathDiv(child=self._lower_ast_dict(child), axis=axis, bc_id=bc_id)

            case {"type": "UnaryOp", "op": "dt", "child": child}:
                return MathDt(child=self._lower_ast_dict(child), bc_id=bc_id)

            case {"type": "UnaryOp", "op": "integral", "child": child}:
                return MathIntegral(child=self._lower_ast_dict(child), over_domain=node.get("over"), bc_id=bc_id)

            case {"type": "UnaryOp", "op": op, "child": child}:
                return MathUnaryOp(op=op, child=self._lower_ast_dict(child), bc_id=bc_id)

            case _:
                return MathScalar(0.0)