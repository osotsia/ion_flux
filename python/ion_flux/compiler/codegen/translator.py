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
        axis_name = node.get("axis")
        state_name = extract_state_name(node["child"], self.layout)
        domain = self.state_map.get(state_name).domain if state_name in self.state_map else None
        
        if not axis_name and domain:
            axis_name = domain.domains[0].name if hasattr(domain, "domains") else domain.name
            
        target_dx = f"dx_{axis_name}" if axis_name else self.dx_symbol
        stride = get_stride(domain, axis_name) if domain else "1"
        local_idx = get_local_index(idx, domain, axis_name) if domain else idx
        res_val = get_resolution(domain, axis_name)
        coord_sys = get_coord_sys(domain, axis_name)

        if coord_sys == "unstructured":
            domain_name = axis_name if axis_name else (domain.name if domain else "unstructured_mesh")
            offsets = self.layout.mesh_offsets[domain_name]
            off_w, off_rp, off_ci = offsets["weights"], offsets["row_ptr"], offsets["col_ind"]
            
            if is_div:
                j_global_expr = "j_local"
                if domain and hasattr(domain, "domains") and len(domain.domains) == 2:
                    d_mac, d_mic = domain.domains[0], domain.domains[1]
                    if axis_name == d_mac.name:
                        j_global_expr = f"(j_local * {d_mic.resolution} + ({idx} % {d_mic.resolution}))"
                    elif axis_name == d_mic.name:
                        j_global_expr = f"((({idx}) / {d_mic.resolution}) * {d_mic.resolution} + j_local)"

                sum_expr = self.translate(node["child"], idx)
                std_div = f"[&]() {{ double s = 0.0;\nint start = (int)p[{off_rp} + {local_idx}]; int end = (int)p[{off_rp} + {local_idx} + 1];\nfor(int k=start; k<end; ++k) {{ int j_local = (int)p[{off_ci} + k]; int j = {j_global_expr};\ndouble w = p[{off_w} + k]; s += {sum_expr}; }} return s;\n}}()"
                
                # Dynamic Neumann boundary injections via surface 3D masks
                if state_name in self.neumann_bcs:
                    for tag, bc_data in self.neumann_bcs[state_name].items():
                        if tag in offsets["surfaces"]:
                            off_surf = offsets["surfaces"][tag]
                            bc_expr = self.translate(bc_data["rhs"], idx)
                            std_div = f"(p[{off_surf} + {local_idx}] > 0.5 ? ({std_div} + {bc_expr}) : ({std_div}))"
                return std_div
            else:
                child_j = self.translate(node["child"], "j")
                child_i = self.translate(node["child"], idx)
                return f"(w * ({child_j} - {child_i}))"

        right = self.translate(node["child"], f"({idx}) + {stride}")
        left = self.translate(node["child"], f"({idx}) - {stride}")
        center = self.translate(node["child"], idx)

        if is_div and state_name in self.neumann_bcs:
            if "right" in self.neumann_bcs[state_name]:
                bc_expr = self.translate(self.neumann_bcs[state_name]["right"]["rhs"], idx)
                right = f"(({local_idx}) == {res_val} - 1 ? ({bc_expr}) : ({right}))"
                center = f"(({local_idx}) == {res_val} - 1 ? ({bc_expr}) : ({center}))"
            if "left" in self.neumann_bcs[state_name]:
                bc_expr = self.translate(self.neumann_bcs[state_name]["left"]["rhs"], idx)
                left = f"(({local_idx}) == 0 ? ({bc_expr}) : ({left}))"
                center = f"(({local_idx}) == 0 ? ({bc_expr}) : ({center}))"

        if is_div and coord_sys == "spherical":
            grad_n = f"(({right}) - ({left})) / (2.0 * {target_dx})"
            r_coord = f"(std::max(1e-12, (double)({local_idx}) * {target_dx}))"
            std_div = f"({grad_n}) + (2.0 / {r_coord}) * ({center})"
            return f"(({local_idx}) == 0 ? (3.0 * ({right}) / {target_dx}) : ({std_div}))"
            
        return f"(({right}) - ({left})) / (2.0 * {target_dx})"

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