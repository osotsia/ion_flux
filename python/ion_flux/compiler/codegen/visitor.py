from typing import Dict, Any
from .utils import extract_state_name
from .topology import get_stride, get_local_index, get_coord_sys
from .ale import get_ale_terms

class CPPEmitter:
    def __init__(self, layout, state_map, dynamic_domains, dx_symbol, neumann_bcs):
        self.layout = layout
        self.state_map = state_map
        self.dynamic_domains = dynamic_domains
        self.dx_symbol = dx_symbol
        self.neumann_bcs = neumann_bcs
        self.use_ydot = False

    def visit(self, node: Dict[str, Any], idx: str) -> str:
        t = node.get("type")
        if t == "Scalar": return str(node["value"])
        if t == "Parameter": return f"p[{self.layout.get_param_offset(node['name'])}]"
        
        if t == "State":
            offset, size = self.layout.state_offsets[node["name"]]
            arr = "ydot" if self.use_ydot else "y"
            return f"{arr}[{offset} + CLAMP({idx}, {size})]"

        # NEW: Handle reading Boundary values inside RHS equations
        if t == "Boundary":
            try:
                state_name = extract_state_name(node, self.layout)
                offset, size = self.layout.state_offsets[state_name]
                side = node["side"]
                state_obj = self.state_map.get(state_name)
                
                # Macro-Micro boundary localized extraction
                if state_obj and hasattr(state_obj.domain, "domains") and len(state_obj.domain.domains) == 2:
                    d_mic = state_obj.domain.domains[1]
                    if node.get("domain") == d_mic.name:
                        if str(idx) == "0":
                            b_idx = f"{d_mic.resolution - 1}" if side == "right" else "0"
                        else:
                            b_idx = f"(({idx}) / {d_mic.resolution}) * {d_mic.resolution}" if side == "left" else f"(({idx}) / {d_mic.resolution}) * {d_mic.resolution} + {d_mic.resolution - 1}"
                        return self.visit(node["child"], b_idx)

                b_idx = "0" if side == "left" else f"{size - 1}"
                return self.visit(node["child"], b_idx)
            except ValueError:
                return self.visit(node["child"], idx)
            
        if t == "BinaryOp":
            l = self.visit(node["left"], idx)
            r = self.visit(node["right"], idx)
            op = node["op"]
            if op == "add": return f"({l} + {r})"
            if op == "sub": return f"({l} - {r})"
            if op == "mul": return f"({l} * {r})"
            if op == "div": return f"({l} / {r})"
            if op == "pow": return f"std::pow({l}, {r})"
            if op == "max": return f"std::max((double)({l}), (double)({r}))"
            if op == "min": return f"std::min((double)({l}), (double)({r}))"
            if op in ("gt", "lt", "ge", "le", "eq", "ne"):
                sym = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="}[op]
                return f"(({l}) {sym} ({r}) ? 1.0 : 0.0)"
                
        if t == "UnaryOp":
            op = node["op"]
            if op == "coords": return f"({idx} * {self.dx_symbol})"
            
            if op == "dt":
                child = node["child"]
                state_name = extract_state_name(child, self.layout)
                offset, size = self.layout.state_offsets[state_name]
                base_dt = f"ydot[{offset} + CLAMP({idx}, {size})]"
                ale_terms = get_ale_terms(state_name, offset, size, self.state_map, self.dynamic_domains, self, idx)
                return f"({base_dt} - {' - '.join(ale_terms)})" if ale_terms else base_dt

            if op == "grad": return self._visit_spatial_op(node, idx, is_div=False)
            if op == "div": return self._visit_spatial_op(node, idx, is_div=True)
            if op == "integral": return self._visit_integral(node, idx)

            child = self.visit(node["child"], idx)
            if op == "neg": return f"(-{child})"
            if op in ("abs", "exp", "log", "sin", "cos"): return f"std::{op}({child})"

        raise ValueError(f"Unknown AST node type: {t}")

    def _visit_spatial_op(self, node: Dict[str, Any], idx: str, is_div: bool) -> str:
        axis_name = node.get("axis")
        state_name = extract_state_name(node["child"], self.layout)
        domain = self.state_map.get(state_name).domain if state_name in self.state_map else None
        
        # FIXED: Infer axis_name directly from the child's spatial domain if not specified.
        if not axis_name and domain:
            axis_name = domain.domains[0].name if hasattr(domain, "domains") else domain.name
            
        target_dx = f"dx_{axis_name}" if axis_name else self.dx_symbol
        stride = get_stride(domain, axis_name) if domain else "1"
        local_idx = get_local_index(idx, domain, axis_name) if domain else idx
        coord_sys = get_coord_sys(domain, axis_name)
        
        res_val = "1"
        if domain:
            ds = domain.domains if hasattr(domain, "domains") else [domain]
            for d in ds:
                if d.name == axis_name: res_val = str(d.resolution)

        right = self.visit(node["child"], f"({idx}) + {stride}")
        left = self.visit(node["child"], f"({idx}) - {stride}")
        center = self.visit(node["child"], idx)

        # Natively inject Neumann flux conditions into the operator
        if is_div and state_name in self.neumann_bcs:
            if "right" in self.neumann_bcs[state_name]:
                bc_expr = self.visit(self.neumann_bcs[state_name]["right"]["rhs"], idx)
                right = f"(({local_idx}) == {res_val} - 1 ? ({bc_expr}) : ({right}))"
                center = f"(({local_idx}) == {res_val} - 1 ? ({bc_expr}) : ({center}))"
            if "left" in self.neumann_bcs[state_name]:
                bc_expr = self.visit(self.neumann_bcs[state_name]["left"]["rhs"], idx)
                left = f"(({local_idx}) == 0 ? ({bc_expr}) : ({left}))"
                center = f"(({local_idx}) == 0 ? ({bc_expr}) : ({center}))"

        if is_div and coord_sys == "spherical":
            grad_N = f"(({right}) - ({left})) / (2.0 * {target_dx})"
            r_coord = f"(std::max(1e-12, (double)({local_idx}) * {target_dx}))"
            std_div = f"({grad_N}) + (2.0 / {r_coord}) * ({center})"
            return f"(({local_idx}) == 0 ? (3.0 * ({right}) / {target_dx}) : ({std_div}))"
        else:
            return f"(({right}) - ({left})) / (2.0 * {target_dx})"

    def _visit_integral(self, node: Dict[str, Any], idx: str) -> str:
        child = node["child"]
        state_name = extract_state_name(child, self.layout)
        domain = self.state_map.get(state_name).domain if state_name in self.state_map else None
        target_dx = f"dx_{node.get('over')}" if node.get('over') else self.dx_symbol
        
        loop_size, eval_idx = "1", "j"
        if domain and hasattr(domain, "domains") and len(domain.domains) == 2:
            d_mac, d_mic = domain.domains[0], domain.domains[1]
            if node.get("over") == d_mic.name:
                loop_size = str(d_mic.resolution)
                eval_idx = f"((({idx}) / {d_mic.resolution}) * {d_mic.resolution} + j)"
            elif node.get("over") == d_mac.name:
                loop_size = str(d_mac.resolution)
                eval_idx = f"(j * {d_mic.resolution} + (({idx}) % {d_mic.resolution}))"
        else:
            loop_size = str(self.layout.state_offsets[state_name][1])
            
        sum_expr = self.visit(child, eval_idx)
        return f"[&]() {{ double s = 0.0; for(int j=0; j<{loop_size}; ++j) s += {sum_expr}; return s * {target_dx}; }}()"