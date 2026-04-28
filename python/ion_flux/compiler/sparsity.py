import itertools
from typing import Dict, Any, List, Set, Tuple
from .codegen.topology import TopologyAnalyzer

class Dependency:
    """Represents a topological relationship between a target row and a source state."""
    __slots__ = ["state", "rel", "axis", "side", "domain"]
    def __init__(self, state: str, rel: str, axis: str = None, side: str = None, domain: str = None):
        self.state = state
        self.rel = rel # Types: 'local', 'stencil', 'integral', 'boundary', 'global'
        self.axis = axis
        self.side = side
        self.domain = domain

    def __repr__(self):
        return f"Dependency(state={self.state}, rel={self.rel})"

class SparsityAnalyzer:
    """
    Statically traces the Abstract Syntax Tree to calculate the structural non-zero 
    pattern of the Jacobian matrix. Guarantees a safe superset for CPR Graph Coloring.
    """
    def __init__(self, ast_payload: Dict[str, Any], layout: Any, states: List[Any]):
        self.ast_payload = ast_payload
        self.layout = layout
        self.topo = TopologyAnalyzer(ast_payload.get("domains", {}))
        self.state_domains = {s.name: getattr(s.domain, "name", None) if s.domain else None for s in states}
        
        self.sparse_triplets: Set[Tuple[int, int]] = set()
        self.bc_map: Dict[str, Dict[str, Any]] = {}
        self.obs_map: Dict[str, Any] = {}
        
        self._build_bc_map()
        self._build_obs_map()
        self._analyze()

    def _build_bc_map(self):
        """Map Neumann boundary AST blocks to their corresponding operator tags."""
        for bc in self.ast_payload.get("boundaries", []):
            if bc["type"] == "neumann":
                self.bc_map[bc["node_id"]] = bc["bcs"]

    def _build_obs_map(self):
        """Map Observables so they can be recursively traversed when encountered."""
        for obs in self.ast_payload.get("observables", []):
            self.obs_map[obs["state"]] = obs

    def flat_to_coords(self, k: int, d_name: str) -> Dict[str, int]:
        """Projects a flat C-array index back into a multi-dimensional topological coordinate."""
        if not d_name: return {}
        axes = self.topo.get_axes(d_name)
        strides = self.topo.get_strides(d_name)
        coords = {}
        rem = k
        for axis in axes:
            stride = strides[axis]
            res = self.topo.domains.get(axis, {}).get("resolution", 1)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            base_axis = self.topo.get_base_axis(axis)
            coords[base_axis] = start + ((rem // stride) % res)
            rem = rem % stride
        return coords

    def coords_to_flat(self, coords: Dict[str, int], d_name: str) -> int:
        """Flattens a multi-dimensional coordinate safely back into a localized C-array index."""
        if not d_name: return 0
        axes = self.topo.get_axes(d_name)
        strides = self.topo.get_strides(d_name)
        flat = 0
        for axis in axes:
            base_axis = self.topo.get_base_axis(axis)
            start = self.topo.domains.get(axis, {}).get("start_idx", 0)
            local_idx = coords.get(base_axis, start) - start
            flat += local_idx * strides[axis]
        return flat

    def _extract_deps(self, node: Any) -> List[Dependency]:
        """Recursively parses AST expressions to find structural interactions."""
        if not isinstance(node, dict): return []
        t = node.get("type")
        res = []
        
        # Inject Neumann boundaries directly into the dependency chain
        if "_bc_id" in node and node["_bc_id"] in self.bc_map:
            for side, bc_ast in self.bc_map[node["_bc_id"]].items():
                res.extend(self._extract_deps(bc_ast))
                
        if t == "State":
            res.append(Dependency(node["name"], "local"))
        elif t == "Observable":
            obs_name = node["name"]
            if obs_name in self.obs_map:
                obs_data = self.obs_map[obs_name]
                if obs_data["type"] == "piecewise":
                    for reg in obs_data["regions"]:
                        res.extend(self._extract_deps(reg["eq"]))
                else:
                    res.extend(self._extract_deps(obs_data["eq"]))
        elif t == "Boundary":
            child_deps = self._extract_deps(node["child"])
            for d in child_deps:
                res.append(Dependency(d.state, "boundary", side=node["side"], domain=node.get("domain")))
        elif t == "UnaryOp":
            op = node["op"]
            child_deps = self._extract_deps(node["child"])
            if op in ("grad", "div"):
                for d in child_deps:
                    if d.rel == "local":
                        res.append(Dependency(d.state, "stencil", axis=node.get("axis")))
                    else:
                        res.append(d)
            elif op == "integral":
                for d in child_deps:
                    if d.rel == "local":
                        res.append(Dependency(d.state, "integral", domain=node.get("over")))
                    else:
                        res.append(d)
            else:
                res.extend(child_deps)
        elif t == "BinaryOp":
            res.extend(self._extract_deps(node["left"]))
            res.extend(self._extract_deps(node["right"]))
        elif t == "Piecewise":
            for reg in node["regions"]:
                res.extend(self._extract_deps(reg["eq"]))
        elif t == "dirichlet_bnd":
            res.extend(self._extract_deps(node["node"]))
        else:
            for v in node.values():
                if isinstance(v, dict): res.extend(self._extract_deps(v))
                elif isinstance(v, list): 
                    for item in v: res.extend(self._extract_deps(item))
        return res

    def _apply_deps(self, target_state: str, deps: List[Dependency], restrict_axis: str = None, restrict_range: Tuple[int, int] = None):
        """Translates abstract AST dependencies into physical Jacobian (Row, Col) triplets."""
        if target_state not in self.layout.state_offsets: return
        off_T, size_T = self.layout.state_offsets[target_state]
        domain_T = self.state_domains.get(target_state)
        
        for k in range(size_T):
            row = off_T + k
            coords_T = self.flat_to_coords(k, domain_T)
            
            # Mask constraints to piecewise regional bounds
            if restrict_axis and restrict_range:
                val = coords_T.get(restrict_axis, -1)
                if val < restrict_range[0] or val >= restrict_range[1]:
                    continue
                    
            for d in deps:
                off_S, size_S = self.layout.state_offsets.get(d.state, (None, None))
                if off_S is None: continue # Safe pass-through for uncoupled variables/observables
                
                domain_S = self.state_domains.get(d.state)
                
                if size_S == 1:
                    self.sparse_triplets.add((row, off_S))
                    continue
                    
                if d.rel == "integral":
                    axes_S = self.topo.get_axes(domain_S)
                    int_axes = self.topo.get_axes(d.domain) if d.domain else axes_S
                    
                    ranges = []
                    for ax in axes_S:
                        b_ax = self.topo.get_base_axis(ax)
                        if ax in int_axes or b_ax in [self.topo.get_base_axis(ia) for ia in int_axes]:
                            res = self.topo.domains.get(ax, {}).get("resolution", 1)
                            start = self.topo.domains.get(ax, {}).get("start_idx", 0)
                            ranges.append(list(range(start, start + res)))
                        else:
                            ranges.append([coords_T.get(b_ax, self.topo.domains.get(ax, {}).get("start_idx", 0))])
                            
                    for indices in itertools.product(*ranges):
                        c_S = {self.topo.get_base_axis(ax): idx for ax, idx in zip(axes_S, indices)}
                        col = off_S + self.coords_to_flat(c_S, domain_S)
                        self.sparse_triplets.add((row, col))
                    continue

                # Contextual anchoring for localized states
                base_c_S = {}
                for ax in self.topo.get_axes(domain_S):
                    b_ax = self.topo.get_base_axis(ax)
                    start = self.topo.domains.get(ax, {}).get("start_idx", 0)
                    base_c_S[b_ax] = coords_T.get(b_ax, start)
                
                if d.rel == "boundary":
                    target_domain_name = d.domain if d.domain else self.topo.get_axes(domain_S)[-1]
                    res = self.topo.domains.get(target_domain_name, {}).get("resolution", 1)
                    start = self.topo.domains.get(target_domain_name, {}).get("start_idx", 0)
                    b_axis_target = self.topo.get_base_axis(target_domain_name)
                    base_c_S[b_axis_target] = start if d.side == "left" else start + res - 1
                        
                col = off_S + self.coords_to_flat(base_c_S, domain_S)
                self.sparse_triplets.add((row, col))
                
                if d.rel == "stencil":
                    target_domain_name = d.axis if d.axis else (self.topo.get_axes(domain_S)[-1] if domain_S else None)
                    if target_domain_name:
                        s_axis = self.topo.get_base_axis(target_domain_name)
                        coord_sys = self.topo.domains.get(target_domain_name, {}).get("coord_sys", "cartesian")
                        
                        if coord_sys == "unstructured":
                            moff = self.layout.mesh_offsets.get(target_domain_name, {})
                            if "row_ptr" in moff:
                                rp_off = moff["row_ptr"]
                                ci_off = moff["col_ind"]
                                flat_S = self.coords_to_flat(base_c_S, domain_S)
                                start_ptr = int(self.layout.mesh_cache[rp_off + flat_S])
                                end_ptr = int(self.layout.mesh_cache[rp_off + flat_S + 1])
                                for ptr in range(start_ptr, end_ptr):
                                    neighbor_local = int(self.layout.mesh_cache[ci_off + ptr])
                                    n_c_S = base_c_S.copy()
                                    start = self.topo.domains.get(target_domain_name, {}).get("start_idx", 0)
                                    n_c_S[s_axis] = start + neighbor_local
                                    self.sparse_triplets.add((row, off_S + self.coords_to_flat(n_c_S, domain_S)))
                        else:
                            res = self.topo.domains.get(target_domain_name, {}).get("resolution", 1)
                            start = self.topo.domains.get(target_domain_name, {}).get("start_idx", 0)
                            curr = base_c_S.get(s_axis, start)
                            if curr > start:
                                n_c_S = base_c_S.copy()
                                n_c_S[s_axis] = curr - 1
                                self.sparse_triplets.add((row, off_S + self.coords_to_flat(n_c_S, domain_S)))
                            if curr < start + res - 1:
                                n_c_S = base_c_S.copy()
                                n_c_S[s_axis] = curr + 1
                                self.sparse_triplets.add((row, off_S + self.coords_to_flat(n_c_S, domain_S)))

    def _analyze(self):
        for eq in self.ast_payload.get("equations", []):
            state_name = eq["state"]
            if eq["type"] == "piecewise":
                for reg in eq["regions"]:
                    b_axis = self.topo.get_base_axis(reg["domain"])
                    start = reg["start_idx"]
                    end = reg["end_idx"]
                    deps = self._extract_deps(reg["eq"])
                    deps.append(Dependency(state_name, "local")) # Diagonal guarantee
                    self._apply_deps(state_name, deps, restrict_axis=b_axis, restrict_range=(start, end))
            else:
                deps = self._extract_deps(eq["eq"])
                deps.append(Dependency(state_name, "local"))
                self._apply_deps(state_name, deps)
                
        for bc in self.ast_payload.get("boundaries", []):
            if bc["type"] == "dirichlet":
                state_name = bc["state"]
                d_name = self.state_domains.get(state_name)
                last_axis = self.topo.get_axes(d_name)[-1] if d_name else None
                b_axis = self.topo.get_base_axis(last_axis) if last_axis else None
                
                if b_axis:
                    res = self.topo.domains.get(last_axis, {}).get("resolution", 1)
                    start = self.topo.domains.get(last_axis, {}).get("start_idx", 0)
                    for side, val_dict in bc["bcs"].items():
                        idx = start if side == "left" else start + res - 1
                        deps = self._extract_deps(val_dict)
                        deps.append(Dependency(state_name, "local"))
                        self._apply_deps(state_name, deps, restrict_axis=b_axis, restrict_range=(idx, idx+1))
                        
            elif bc["type"] == "moving_domain":
                d_name = bc["domain"]
                for side, rhs_ast in bc["bcs"].items():
                    ale_deps = self._extract_deps(rhs_ast)
                    for s_name, s_dom in self.state_domains.items():
                        if s_dom == d_name:
                            self._apply_deps(s_name, ale_deps)