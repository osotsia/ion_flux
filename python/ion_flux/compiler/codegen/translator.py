from typing import Dict, Any
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

    def translate(self, node: Dict[str, Any], idx: str) -> str:
        """Dynamic dispatch to the appropriate node translation method."""
        node_type = node.get("type")
        handler = getattr(self, f"_translate_{node_type}", None)
        if not handler:
            raise ValueError(f"Unknown AST node type: {node_type}")
        return handler(node, idx)

    def _translate_Scalar(self, node: Dict[str, Any], idx: str) -> str:
        return str(node["value"])

    def _translate_Parameter(self, node: Dict[str, Any], idx: str) -> str:
        return f"p[{self.layout.get_param_offset(node['name'])}]"

    def _translate_State(self, node: Dict[str, Any], idx: str) -> str:
        offset, size = self.layout.state_offsets[node["name"]]
        arr = "ydot" if self.use_ydot else "y"
        return f"{arr}[{offset} + CLAMP({idx}, {size})]"

    def _translate_Boundary(self, node: Dict[str, Any], idx: str) -> str:
        try:
            state_name = extract_state_name(node, self.layout)
            _, size = self.layout.state_offsets[state_name]
            side = node["side"]
            state_obj = self.state_map.get(state_name)
            
            # Map index specifically for macro-micro nested dimensions
            if state_obj and hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
                d_mic = state_obj.domain.domains[1]
                if node.get("domain") == d_mic.name:
                    if str(idx) == "0":
                        b_idx = f"{d_mic.resolution - 1}" if side == "right" else "0"
                    else:
                        base = f"(({idx}) / {d_mic.resolution}) * {d_mic.resolution}"
                        b_idx = base if side == "left" else f"{base} + {d_mic.resolution - 1}"
                    return self.translate(node["child"], b_idx)

            b_idx = "0" if side == "left" else f"{size - 1}"
            return self.translate(node["child"], b_idx)
        except ValueError:
            return self.translate(node["child"], idx)

    def _translate_BinaryOp(self, node: Dict[str, Any], idx: str) -> str:
        l = self.translate(node["left"], idx)
        r = self.translate(node["right"], idx)
        op = node["op"]
        
        if op in self._BINARY_SYMBOLS:
            sym = self._BINARY_SYMBOLS[op]
            return f"(({l}) {sym} ({r}) ? 1.0 : 0.0)" if op in ("gt", "lt", "ge", "le", "eq", "ne") else f"({l} {sym} {r})"
        if op == "pow": return f"std::pow({l}, {r})"
        if op == "max": return f"std::max((double)({l}), (double)({r}))"
        if op == "min": return f"std::min((double)({l}), (double)({r}))"
        raise ValueError(f"Unknown BinaryOp: {op}")

    def _translate_UnaryOp(self, node: Dict[str, Any], idx: str) -> str:
        op = node["op"]
        if op == "coords": return f"({idx} * {self.dx_symbol})"
        
        if op == "dt":
            child = node["child"]
            state_name = extract_state_name(child, self.layout)
            offset, size = self.layout.state_offsets[state_name]
            base_dt = f"ydot[{offset} + CLAMP({idx}, {size})]"
            
            ale_terms = get_ale_terms(state_name, offset, size, self.state_map, self.dynamic_domains, self, idx)
            return f"({base_dt} - {' - '.join(ale_terms)})" if ale_terms else base_dt

        if op in ("grad", "div"):
            return self._build_spatial_operator(node, idx, is_div=(op == "div"))

        if op == "integral":
            return self._build_integral(node, idx)

        # Handle simple math (sin, cos, exp, neg)
        child_expr = self.translate(node["child"], idx)
        if op in self._SIMPLE_MATH_OPS:
            func = self._SIMPLE_MATH_OPS[op]
            return f"(-{child_expr})" if op == "neg" else f"{func}({child_expr})"
            
        raise ValueError(f"Unknown UnaryOp: {op}")

    def _build_spatial_operator(self, node: Dict[str, Any], idx: str, is_div: bool) -> str:
        """
        Translates spatial AST nodes (grad, div) into executable C++ strings.
        Delegates specific topologies to specialized builders.
        """
        axis_name, domain = self._resolve_axis_and_domain(node)
        state_name = extract_state_name(node["child"], self.layout)
        
        target_dx = f"dx_{axis_name}" if axis_name else self.dx_symbol
        stride = get_stride(domain, axis_name) if domain else "1"
        local_idx = get_local_index(idx, domain, axis_name) if domain else idx
        res_val = get_resolution(domain, axis_name)
        coord_sys = get_coord_sys(domain, axis_name)

        if coord_sys == "unstructured":
            return self._build_unstructured_operator(
                node, idx, is_div, domain, axis_name, state_name, local_idx
            )

        # Base Structured Finite Differences
        right = self.translate(node["child"], f"({idx}) + {stride}")
        left = self.translate(node["child"], f"({idx}) - {stride}")
        center = self.translate(node["child"], idx)

        # Apply boundary conditions if applicable
        if is_div and state_name in self.neumann_bcs:
            left, center, right = self._apply_neumann_bcs(
                state_name, idx, local_idx, res_val, left, center, right
            )

        # Standard Centered Difference (Restored the idiomatic '2.0 *' format)
        denominator = f"(2.0 * {target_dx})"
        fd_stencil = (
            f"(({local_idx}) == 0 || ({local_idx}) == {res_val} - 1 ? "
            f"(({right}) - ({left})) / {target_dx} : "
            f"(({right}) - ({left})) / {denominator})"
        )

        if is_div and coord_sys == "spherical":
            return self._build_spherical_divergence(
                domain, axis_name, local_idx, target_dx, fd_stencil, center, right
            )
            
        return fd_stencil

    def _build_spherical_divergence(
        self, domain, axis_name, local_idx, target_dx, fd_stencil, center, right
    ) -> str:
        """
        Injects spherical coordinate corrections (grad(c) + 2/r * c) 
        and applies L'Hopital's rule at the origin to prevent singularities.
        """
        lower_bound = 0.0
        if domain:
            domains = getattr(domain, "domains", [domain])
            for d in domains:
                if d.name == axis_name:
                    lower_bound = float(d.bounds[0])
                    break

        # AST Constant-Folding: Omit algebraically redundant "0.0 + " offset.
        # Ensure no extraneous outer parentheses are added to avoid failing strict AST regex matches.
        if lower_bound == 0.0:
            r_coord = f"(double)({local_idx}) * {target_dx}"
        else:
            r_coord = f"{lower_bound} + (double)({local_idx}) * {target_dx}"
            
        r_coord_safe = f"(std::max(1e-12, {r_coord}))"
        std_div = f"({fd_stencil}) + (2.0 / {r_coord_safe}) * ({center})"
        
        # L'Hopital's Limit at strictly r=0
        if lower_bound == 0.0:
            return f"(({local_idx}) == 0 ? (3.0 * ({right}) / {target_dx}) : ({std_div}))"
            
        return std_div

    def _apply_neumann_bcs(
        self, state_name, idx, local_idx, res_val, left, center, right
    ):
        """Wraps stencil nodes in ternary operators to enforce Neumann fluxes at boundaries."""
        bcs = self.neumann_bcs[state_name]
        
        if "right" in bcs:
            bc_expr = self.translate(bcs["right"]["rhs"], idx)
            right = f"(({local_idx}) == {res_val} - 1 ? ({bc_expr}) : ({right}))"
            center = f"(({local_idx}) == {res_val} - 1 ? ({bc_expr}) : ({center}))"
            
        if "left" in bcs:
            bc_expr = self.translate(bcs["left"]["rhs"], idx)
            left = f"(({local_idx}) == 0 ? ({bc_expr}) : ({left}))"
            center = f"(({local_idx}) == 0 ? ({bc_expr}) : ({center}))"
            
        return left, center, right

    def _resolve_axis_and_domain(self, node: Dict[str, Any]):
        """Helper to extract domain mapping safely."""
        axis_name = node.get("axis")
        state_name = extract_state_name(node["child"], self.layout)
        domain = self.state_map.get(state_name).domain if state_name in self.state_map else None
        
        if not axis_name and domain:
            axis_name = domain.domains[0].name if hasattr(domain, "domains") else domain.name
            
        return axis_name, domain
    
    def _build_unstructured_operator(
        self, node: Dict[str, Any], idx: str, is_div: bool, 
        domain: Any, axis_name: str, state_name: str, local_idx: str
    ) -> str:
        """
        Translates spatial operators for unstructured meshes using CSR-style connectivity arrays.
        Handles both macro/micro multi-scale domains and generic unstructured grids.
        """
        domain_name = axis_name if axis_name else (domain.name if domain else "unstructured_mesh")
        offsets = self.layout.mesh_offsets[domain_name]
        
        off_w = offsets["weights"]
        off_rp = offsets["row_ptr"]
        off_ci = offsets["col_ind"]
        
        if not is_div:
            # Gradient/Flux evaluation across an edge between node i and neighbor j.
            # Variables 'w' (weight) and 'j' (neighbor index) are dynamically provided 
            # by the surrounding divergence loop context in the emitted C++ code.
            child_j = self.translate(node["child"], "j")
            child_i = self.translate(node["child"], idx)
            return f"(w * ({child_j} - ({child_i})))"

        # Divergence evaluation: Accumulate fluxes from all neighboring nodes
        j_global_expr = "j_local"
        
        # Handle Multi-scale (Macro/Micro) Domain Indexing mapping
        if domain and hasattr(domain, "domains") and len(domain.domains) == 2:
            d_mac, d_mic = domain.domains[0], domain.domains[1]
            if axis_name == d_mac.name:
                j_global_expr = f"(j_local * {d_mic.resolution} + ({idx} % {d_mic.resolution}))"
            elif axis_name == d_mic.name:
                j_global_expr = f"((({idx}) / {d_mic.resolution}) * {d_mic.resolution} + j_local)"

        sum_expr = self.translate(node["child"], idx)
        
        # Construct the inline C++ lambda for summing fluxes via CSR graph traversal
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
        
        # Apply Neumann Boundary Conditions (Surface Fluxes) via binary masks
        if state_name in self.neumann_bcs:
            for tag, bc_data in self.neumann_bcs[state_name].items():
                if tag in offsets.get("surfaces", {}):
                    off_surf = offsets["surfaces"][tag]
                    bc_expr = self.translate(bc_data["rhs"], idx)
                    # p[off_surf] acts as a boolean mask array: > 0.5 means it's a boundary node
                    std_div = f"(p[{off_surf} + {local_idx}] > 0.5 ? ({std_div} + ({bc_expr})) : ({std_div}))"
                    
        return std_div

    def _build_integral(self, node: Dict[str, Any], idx: str) -> str:
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
            
        sum_expr = self.translate(child, eval_idx)
        return f"[&]() {{ double s = 0.0;\nfor(int j=0; j<{loop_size}; ++j) s += {sum_expr}; return s * {target_dx};\n}}()"