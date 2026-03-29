from typing import Dict, Any, Optional
from .ast_analysis import extract_state_name
from .topology import get_stride, get_local_index, get_coord_sys, get_resolution
from .ale import get_ale_terms

class CppTranslator:
    """Recursively translates Python AST nodes into native C++ mathematical strings."""
    
    _SIMPLE_MATH_OPS = {"neg": "-", "abs": "std::abs", "exp": "std::exp", "log": "std::log", "sin": "std::sin", "cos": "std::cos"}
    _BINARY_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/", "gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="}

    def __init__(self, layout, state_map, dynamic_domains, dx_symbol, neumann_bcs):
        self.layout = layout
        self.state_map = state_map
        self.dynamic_domains = dynamic_domains
        self.dx_symbol = dx_symbol
        self.neumann_bcs = neumann_bcs
        self.use_ydot = False
        self.current_domain = None

    def translate(self, node: Dict[str, Any], idx: str, face: Optional[str] = None) -> str:
        """Dynamic dispatch to the appropriate node translation method with Finite Volume face context."""
        node_type = node.get("type")
        handler = getattr(self, f"_translate_{node_type}", None)
        if not handler:
            raise ValueError(f"Unknown AST node type: {node_type}")
        return handler(node, idx, face)

    def _translate_Scalar(self, node: Dict[str, Any], idx: str, face: Optional[str] = None) -> str:
        return str(node["value"])

    def _translate_Parameter(self, node: Dict[str, Any], idx: str, face: Optional[str] = None) -> str:
        return f"p[{self.layout.get_param_offset(node['name'])}]"

    def _translate_State(self, node: Dict[str, Any], idx: str, face: Optional[str] = None) -> str:
        offset, size = self.layout.state_offsets[node["name"]]
        arr = "ydot" if self.use_ydot else "y"
        
        state_obj = self.state_map.get(node["name"])
        target_domain = getattr(state_obj, "domain", None)
        
        eval_idx = idx
        
        # Map indices cleanly across hierarchical macro-micro domain boundaries
        if self.current_domain and target_domain and self.current_domain != target_domain:
            if type(self.current_domain).__name__ == "CompositeDomain" and len(self.current_domain.domains) == 2:
                d_mac, d_mic = self.current_domain.domains
                if target_domain.name == d_mac.name:
                    eval_idx = f"(({idx}) / {d_mic.resolution})"
                elif target_domain.name == d_mic.name:
                    eval_idx = f"(({idx}) % {d_mic.resolution})"
                    
        # Apply Finite Volume face-staggering interpolation
        if face == "right":
            idx_curr = f"{arr}[{offset} + CLAMP({eval_idx}, {size})]"
            idx_next = f"{arr}[{offset} + CLAMP(({eval_idx}) + 1, {size})]"
            return f"(0.5 * ({idx_curr} + {idx_next}))"
        elif face == "left":
            idx_curr = f"{arr}[{offset} + CLAMP({eval_idx}, {size})]"
            idx_prev = f"{arr}[{offset} + CLAMP(({eval_idx}) - 1, {size})]"
            return f"(0.5 * ({idx_curr} + {idx_prev}))"
        else:
            return f"{arr}[{offset} + CLAMP({eval_idx}, {size})]"

    def _translate_Boundary(self, node: Dict[str, Any], idx: str, face: Optional[str] = None) -> str:
        try:
            state_name = extract_state_name(node, self.layout)
            _, size = self.layout.state_offsets[state_name]
            side = node["side"]
            state_obj = self.state_map.get(state_name)
            
            # Map index specifically for macro-micro nested dimensions
            if state_obj and hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
                d_mac, d_mic = state_obj.domain.domains
                if node.get("domain") == d_mic.name:
                    if self.current_domain and self.current_domain.name == d_mac.name:
                        base = f"(({idx}) * {d_mic.resolution})"
                    else:
                        base = f"((({idx}) / {d_mic.resolution}) * {d_mic.resolution})"
                        
                    b_idx = base if side == "left" else f"({base} + {d_mic.resolution - 1})"
                    return self.translate(node["child"], b_idx, face=None)

            b_idx = "0" if side == "left" else f"{size - 1}"
            # Boundaries target absolute nodes, ignoring internal finite volume face offsets
            return self.translate(node["child"], b_idx, face=None)
        except ValueError:
            return self.translate(node["child"], idx, face=None)

    def _translate_BinaryOp(self, node: Dict[str, Any], idx: str, face: Optional[str] = None) -> str:
        l = self.translate(node["left"], idx, face)
        r = self.translate(node["right"], idx, face)
        op = node["op"]
        
        if op in self._BINARY_SYMBOLS:
            sym = self._BINARY_SYMBOLS[op]
            return f"(({l}) {sym} ({r}) ? 1.0 : 0.0)" if op in ("gt", "lt", "ge", "le", "eq", "ne") else f"({l} {sym} {r})"
        if op == "pow": return f"std::pow({l}, {r})"
        if op == "max": return f"std::max((double)({l}), (double)({r}))"
        if op == "min": return f"std::min((double)({l}), (double)({r}))"
        raise ValueError(f"Unknown BinaryOp: {op}")

    def _translate_UnaryOp(self, node: Dict[str, Any], idx: str, face: Optional[str] = None) -> str:
        op = node["op"]
        if op == "coords": 
            shift = ""
            if face == "right": shift = f" + 0.5 * {self.dx_symbol}"
            elif face == "left": shift = f" - 0.5 * {self.dx_symbol}"
            return f"(({idx} * {self.dx_symbol}){shift})"
        
        if op == "dt":
            child = node["child"]
            state_name = extract_state_name(child, self.layout)
            offset, size = self.layout.state_offsets[state_name]
            base_dt = f"ydot[{offset} + CLAMP({idx}, {size})]"
            
            ale_terms = get_ale_terms(state_name, offset, size, self.state_map, self.dynamic_domains, self, idx)
            return f"({base_dt} - {' - '.join(ale_terms)})" if ale_terms else base_dt

        if op in ("grad", "div"):
            return self._build_spatial_operator(node, idx, is_div=(op == "div"), face=face)

        if op == "integral":
            return self._build_integral(node, idx, face)

        # Handle simple math (sin, cos, exp, neg) seamlessly interpolating at faces
        child_expr = self.translate(node["child"], idx, face)
        if op in self._SIMPLE_MATH_OPS:
            func = self._SIMPLE_MATH_OPS[op]
            return f"(-{child_expr})" if op == "neg" else f"{func}({child_expr})"
            
        raise ValueError(f"Unknown UnaryOp: {op}")

    def _build_spatial_operator(self, node: Dict[str, Any], idx: str, is_div: bool, face: Optional[str] = None) -> str:
        axis_name, domain = self._resolve_axis_and_domain(node)
        state_name = extract_state_name(node["child"], self.layout)
        
        target_dx = f"dx_{axis_name}" if axis_name else self.dx_symbol
        stride = get_stride(domain, axis_name) if domain else "1"
        local_idx = get_local_index(idx, domain, axis_name) if domain else idx
        res_val = get_resolution(domain, axis_name)
        coord_sys = get_coord_sys(domain, axis_name)

        if coord_sys == "unstructured":
            return self._build_unstructured_operator(
                node, idx, is_div, domain, axis_name, state_name, local_idx, face
            )

        if is_div:
            # Shift translation context to evaluate strictly on cell boundaries
            right_face = self.translate(node["child"], idx, face="right")
            left_face = self.translate(node["child"], idx, face="left")
            center = self.translate(node["child"], idx, face=None)

            # Neumann conditions implicitly override the discrete face fluxes
            left_face, center, right_face = self._apply_neumann_bcs(
                state_name, idx, local_idx, res_val, left_face, center, right_face
            )

            fd_stencil = f"(({right_face}) - ({left_face})) / {target_dx}"

            if coord_sys == "spherical":
                return self._build_spherical_divergence(
                    domain, axis_name, local_idx, target_dx, fd_stencil, center, right_face
                )
            return fd_stencil
        else:
            # Gradients consume the face context to generate perfectly compact 2-point stencils
            if face == "right":
                right = self.translate(node["child"], f"({idx}) + {stride}", face=None)
                curr = self.translate(node["child"], idx, face=None)
                return f"(({right}) - ({curr})) / {target_dx}"
            elif face == "left":
                curr = self.translate(node["child"], idx, face=None)
                left = self.translate(node["child"], f"({idx}) - {stride}", face=None)
                return f"(({curr}) - ({left})) / {target_dx}"
            else:
                curr = self.translate(node["child"], idx, face=None)
                right = self.translate(node["child"], f"({idx}) + {stride}", face=None)
                left = self.translate(node["child"], f"({idx}) - {stride}", face=None)
                denominator = f"(2.0 * {target_dx})"
                # Fixed to prevent topological bleeding across nested grids.
                # Uses mathematically exact asymmetric backward/forward differences at boundaries.
                return (
                    f"(({local_idx}) == 0 ? (({right}) - ({curr})) / {target_dx} : "
                    f"(({local_idx}) == {res_val} - 1 ? (({curr}) - ({left})) / {target_dx} : "
                    f"(({right}) - ({left})) / {denominator}))"
                )

    def _build_spherical_divergence(
        self, domain, axis_name, local_idx, target_dx, fd_stencil, center, right_face
    ) -> str:
        lower_bound = 0.0
        if domain:
            domains = getattr(domain, "domains", [domain])
            for d in domains:
                if d.name == axis_name:
                    lower_bound = float(d.bounds[0])
                    break

        if lower_bound == 0.0:
            r_coord = f"(double)({local_idx}) * {target_dx}"
        else:
            r_coord = f"{lower_bound} + (double)({local_idx}) * {target_dx}"
            
        r_coord_safe = f"(std::max(1e-12, {r_coord}))"
        std_div = f"({fd_stencil}) + (2.0 / {r_coord_safe}) * ({center})"
        
        # L'Hopital's Limit at r=0. Distance from center to right face is dx/2.
        # Approximating dF/dr as (F_{1/2} - 0) / (0.5 * dx). Resulting limit simplifies to 6 * F_{1/2} / dx
        if lower_bound == 0.0:
            return f"(({local_idx}) == 0 ? (6.0 * ({right_face}) / {target_dx}) : ({std_div}))"
            
        return std_div

    def _apply_neumann_bcs(self, state_name, idx, local_idx, res_val, left, center, right):
        bcs = self.neumann_bcs.get(state_name, {})
        
        if "right" in bcs:
            bc_expr = self.translate(bcs["right"]["rhs"], idx, face=None)
            right = f"(({local_idx}) == {res_val} - 1 ? ({bc_expr}) : ({right}))"
            center = f"(({local_idx}) == {res_val} - 1 ? ({bc_expr}) : ({center}))"
        else:
            # Default to no-flux to hermetically seal uncoupled macro-micro boundaries
            right = f"(({local_idx}) == {res_val} - 1 ? (0.0) : ({right}))"
            center = f"(({local_idx}) == {res_val} - 1 ? (0.0) : ({center}))"
            
        if "left" in bcs:
            bc_expr = self.translate(bcs["left"]["rhs"], idx, face=None)
            left = f"(({local_idx}) == 0 ? ({bc_expr}) : ({left}))"
            center = f"(({local_idx}) == 0 ? ({bc_expr}) : ({center}))"
        else:
            left = f"(({local_idx}) == 0 ? (0.0) : ({left}))"
            center = f"(({local_idx}) == 0 ? (0.0) : ({center}))"
            
        return left, center, right

    def _resolve_axis_and_domain(self, node: Dict[str, Any]):
        axis_name = node.get("axis")
        state_name = extract_state_name(node["child"], self.layout)
        domain = self.state_map.get(state_name).domain if state_name in self.state_map else None
        
        if not axis_name and domain:
            axis_name = domain.domains[0].name if hasattr(domain, "domains") else domain.name
            
        return axis_name, domain
    
    def _build_unstructured_operator(
        self, node: Dict[str, Any], idx: str, is_div: bool, 
        domain: Any, axis_name: str, state_name: str, local_idx: str, face: Optional[str] = None
    ) -> str:
        domain_name = axis_name if axis_name else (domain.name if domain else "unstructured_mesh")
        offsets = self.layout.mesh_offsets[domain_name]
        
        off_w = offsets["weights"]
        off_rp = offsets["row_ptr"]
        off_ci = offsets["col_ind"]
        
        if not is_div:
            child_j = self.translate(node["child"], "j", face=None)
            child_i = self.translate(node["child"], idx, face=None)
            return f"(w * ({child_j} - ({child_i})))"

        j_global_expr = "j_local"
        
        if domain and hasattr(domain, "domains") and len(domain.domains) == 2:
            d_mac, d_mic = domain.domains[0], domain.domains[1]
            if axis_name == d_mac.name:
                j_global_expr = f"(j_local * {d_mic.resolution} + ({idx} % {d_mic.resolution}))"
            elif axis_name == d_mic.name:
                j_global_expr = f"((({idx}) / {d_mic.resolution}) * {d_mic.resolution} + j_local)"

        sum_expr = self.translate(node["child"], idx, face=None)
        
        std_div = (
            f"[&]() {{\n"
            f"    double s = 0.0;\n"
            f"    int start = (int)p[{off_rp} + {local_idx}];\n"
            f"    int end = (int)p[{off_rp} + {local_idx} + 1];\n"
            f"    for(int k = start; k < end; ++k) {{\n"
            f"        int j_local = (int)p[{off_ci} + k];\n"
            f"        int j = {j_global_expr};\n"
            f"        double w = p[{off_w} + k];\n"
            f"        s += {sum_expr};\n"
            f"    }}\n"
            f"    return s;\n"
            f"}}()"
        )
        
        if state_name in self.neumann_bcs:
            for tag, bc_data in self.neumann_bcs[state_name].items():
                if tag in offsets.get("surfaces", {}):
                    off_surf = offsets["surfaces"][tag]
                    bc_expr = self.translate(bc_data["rhs"], idx, face=None)
                    std_div = f"(p[{off_surf} + {local_idx}] > 0.5 ? ({std_div} + ({bc_expr})) : ({std_div}))"
                    
        return std_div

    def _build_integral(self, node: Dict[str, Any], idx: str, face: Optional[str] = None) -> str:
        child = node["child"]
        state_name = extract_state_name(child, self.layout)
        domain = self.state_map.get(state_name).domain if state_name in self.state_map else None
        target_dx = f"dx_{node.get('over')}" if node.get('over') else self.dx_symbol
        
        loop_size, eval_idx = "1", "j"
        if domain and hasattr(domain, "domains") and len(domain.domains) == 2:
            d_mac, d_mic = domain.domains[0], domain.domains[1]
            
            target_domain = getattr(self, "current_domain", None)
            
            if node.get("over") == d_mic.name:
                loop_size = str(d_mic.resolution)
                if target_domain and target_domain.name == d_mac.name:
                    eval_idx = f"(({idx}) * {d_mic.resolution} + j)"
                else:
                    eval_idx = f"((({idx}) / {d_mic.resolution}) * {d_mic.resolution} + j)"
            elif node.get("over") == d_mac.name:
                loop_size = str(d_mac.resolution)
                if target_domain and target_domain.name == d_mic.name:
                    eval_idx = f"(j * {d_mic.resolution} + ({idx}))"
                else:
                    eval_idx = f"(j * {d_mic.resolution} + (({idx}) % {d_mic.resolution}))"
        else:
            loop_size = str(self.layout.state_offsets[state_name][1])
            eval_idx = "j"
            
        sum_expr = self.translate(child, eval_idx, face=None)
        return f"[&]() {{ double s = 0.0;\nfor(int j=0; j<{loop_size}; ++j) s += {sum_expr}; return s * {target_dx};\n}}()"