import os
import tempfile
import itertools
import logging
from typing import Dict, Any, List, Optional, Tuple

from ion_flux.dsl.core import PDE, State, Parameter, Observable
from ion_flux.dsl.spatial import Domain, CompositeDomain
from ion_flux.compiler._1_analysis.memory_layout import MemoryLayout
from ion_flux.compiler._1_analysis.topology import TopologyAnalyzer
from ion_flux.compiler._1_analysis.semantics import SemanticContext
from ion_flux.compiler._2_lowering.normalization import NormalizationPass
from ion_flux.compiler._1_analysis.verification import verify_manifold
from ion_flux.compiler._4_codegen.builder import generate_cpp
from ion_flux.compiler._5_toolchain.clang_invoker import NativeCompiler
from ion_flux.compiler._3_optimization.sparsity_tracer import SparsityAnalyzer
from ion_flux.compiler._3_optimization.cpr_coloring import HybridGraphColorer
from ion_flux.runtime.manifest import ExecutableManifest

def build_manifest(model: PDE, target: str = "cpu:serial", cache: bool = True, jacobian_bandwidth: Optional[int] = None, mock_execution: bool = False) -> ExecutableManifest:
    """Orchestrates the Compiler pipeline to emit a frozen execution target."""
    states = model.components(State) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, State)]
    params = model.components(Parameter) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, Parameter)]
    observables = model.components(Observable) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, Observable)]
    domains = model.components(Domain) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, Domain)]
    comp_domains = model.components(CompositeDomain) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, CompositeDomain)]
    all_domains = domains + comp_domains

    layout = MemoryLayout(states, params, observables, all_domains)
    default_parameters = {p.name: p.default for p in params}
    ast_payload = model.ast() if hasattr(model, "ast") else {}
    
    state_domain_map = {s.name: getattr(s.domain, "name", "") for s in states}
    state_max_steps = {s.name: s.max_newton_step for s in states}
    
    topo = None
    if ast_payload:
        topo = TopologyAnalyzer(ast_payload.get("domains", {}))
        semantic_ctx = SemanticContext(ast_payload)
        state_map = {s.name: s for s in states + observables}
        
        ast_payload = NormalizationPass(ast_payload, topo, semantic_ctx, state_map).run()
        verify_manifold(ast_payload)
        
        targeted_states = {eq["state"] for eq in ast_payload.get("equations", [])}
        for state_name in layout.state_offsets.keys():
            if state_name not in targeted_states:
                raise ValueError(f"Unconstrained state detected: '{state_name}'. Rank deficiency in system.")

    if jacobian_bandwidth is None:
        jacobian_bandwidth = _compute_symbolic_bandwidth(layout, states, ast_payload)

    lib_path = ""
    cpp_source = ""
    cpr_cache = ([], [], [], [], [])
    
    if ast_payload and not mock_execution:
        cpp_source, eq_stmts = generate_cpp(ast_payload, layout, states, observables, target=target)
        cpr_cache = _compute_cpr(eq_stmts, layout, jacobian_bandwidth)
        
        compiler = NativeCompiler() if cache else NativeCompiler(cache_dir=os.path.join(tempfile.gettempdir(), "nocache"))
        runtime = compiler.compile(cpp_source, layout.n_states)
        lib_path = runtime.lib_path

    id_arr, spatial_diag, max_steps = _compute_static_metadata(layout, ast_payload, topo, state_domain_map, state_max_steps)

    return ExecutableManifest(
        lib_path=lib_path,
        layout=layout,
        default_parameters=default_parameters,
        ast_payload=ast_payload,
        jacobian_bandwidth=jacobian_bandwidth,
        cpr_cache=cpr_cache,
        id_arr=id_arr,
        spatial_diag=spatial_diag,
        max_steps=max_steps,
        state_domain_map=state_domain_map,
        cpp_source=cpp_source
    )

def _compute_symbolic_bandwidth(layout: MemoryLayout, states: List[State], ast_payload: Dict[str, Any]) -> int:
    from ion_flux.compiler._1_analysis.ast_utils import extract_state_names
    if any(getattr(s.domain, "coord_sys", "") == "unstructured" for s in states): return -1
    
    max_bw = 0
    def check_dependencies(target_state: str, node: Dict[str, Any]) -> int:
        nonlocal max_bw
        if target_state not in layout.state_offsets: return max_bw
        off_t, size_t = layout.state_offsets[target_state]
        if size_t > 1: max_bw = max(max_bw, 2)
        
        deps = extract_state_names(node)
        for d in deps:
            if d not in layout.state_offsets: continue
            off_d, _ = layout.state_offsets[d]
            if abs(off_t - off_d) > 0: return 0 
        return max_bw

    for bc_data in ast_payload.get("boundaries", []):
        if bc_data.get("type") == "moving_domain": return 0

    for eq_data in ast_payload.get("equations", []):
        target_state = eq_data["state"]
        if eq_data["type"] == "piecewise":
            for reg in eq_data["regions"]:
                if check_dependencies(target_state, reg["eq"]) == 0: return 0
        else:
            if check_dependencies(target_state, eq_data["eq"]) == 0: return 0
            
    return max_bw if max_bw > 0 else 0

def _compute_cpr(eq_stmts: List[Any], layout: MemoryLayout, jacobian_bandwidth: int):
    c_seeds, c_ptrs, c_rows, c_cols, c_dense = [], [], [], [], []
    if jacobian_bandwidth != -1:
        try:
            analyzer = SparsityAnalyzer(eq_stmts, layout)
            colorer = HybridGraphColorer(layout.n_states, analyzer.sparse_triplets, dense_threshold=20)
            
            c_seeds = colorer.color_seeds
            c_ptrs = [0]
            for c_idx in range(colorer.n_colors):
                count = 0
                for r, c in colorer.sparse_triplets:
                    if colorer.color_map[c] == c_idx:
                        c_rows.append(r)
                        c_cols.append(c)
                        count += 1
                c_ptrs.append(c_ptrs[-1] + count)
            c_dense = colorer.dense_rows
        except Exception as e:
            logging.warning(f"CPR Graph Coloring failed: {e}. Falling back to Dense Forward-Mode AD sweeps.")
            N = layout.n_states
            c_seeds = [[0.0] * N for _ in range(N)]
            for i in range(N): c_seeds[i][i] = 1.0
            c_ptrs = list(range(0, N * N + 1, N))
            
            c_rows, c_cols = [], []
            for c in range(N):
                c_rows.extend(range(N))
                c_cols.extend([c] * N)
                
            c_dense = []
    return (c_seeds, c_ptrs, c_rows, c_cols, c_dense)

def _compute_static_metadata(layout: MemoryLayout, ast_payload: Dict[str, Any], topo: TopologyAnalyzer, state_domain_map: Dict[str, str], state_max_steps: Dict[str, Any]):
    """Extracts immutable topological masks (e.g. IDA differential array, Spatial Root clamping)."""
    id_arr = [0.0] * layout.n_states
    spatial_diag = [0.0] * layout.n_states
    max_steps = [0.0] * layout.n_states

    if not ast_payload or not topo:
        return id_arr, spatial_diag, max_steps

    def _mark_differentials(node: Dict[str, Any], start: int, end: int) -> None:
        if isinstance(node, dict):
            if node.get("type") == "UnaryOp" and node.get("op") == "dt":
                for i in range(start, end): id_arr[i] = 1.0
            for v in node.values(): _mark_differentials(v, start, end)
        elif isinstance(node, list):
            for item in node: _mark_differentials(item, start, end)

    def _check_dt(node: Dict[str, Any]) -> bool:
        if isinstance(node, dict):
            if node.get("type") == "UnaryOp" and node.get("op") == "dt": return True
            for v in node.values():
                if _check_dt(v): return True
        elif isinstance(node, list):
            for item in node:
                if _check_dt(item): return True
        return False

    # 1. Map Time Derivatives (Differential variables == 1.0)
    for eq_data in ast_payload.get("equations", []):
        state_name = eq_data["state"]
        offset, size = layout.state_offsets[state_name]
        
        if eq_data["type"] == "piecewise":
            d_name = state_domain_map.get(state_name, "")
            for reg in eq_data["regions"]:
                if not _check_dt(reg["eq"]): continue
                if not d_name:
                    id_arr[offset] = 1.0
                    continue
                    
                axes = topo.get_axes(d_name)
                strides = topo.get_strides(d_name)
                b_axis = topo.get_base_axis(reg["domain"])
                
                ranges = []
                for axis in axes:
                    base = topo.get_base_axis(axis)
                    if base == b_axis: ranges.append(range(reg["start_idx"], reg["end_idx"]))
                    else: ranges.append(range(topo.domains.get(axis, {}).get("resolution", 1)))
                
                for indices in itertools.product(*ranges):
                    flat_idx = 0
                    for axis, idx in zip(axes, indices): flat_idx += idx * strides[axis]
                    id_arr[offset + flat_idx] = 1.0
        else:
            _mark_differentials(eq_data["eq"], offset, offset + size)
            
    # 2. Mask Dirichlet Boundary Algebraic Constraints (0.0)
    for bc_data in ast_payload.get("boundaries", []):
        if bc_data["type"] == "dirichlet":
            state_name = bc_data["state"]
            offset, size = layout.state_offsets[state_name]
            d_name = state_domain_map.get(state_name, "")
            
            if not d_name:
                if "left" in bc_data["bcs"]: id_arr[offset] = 0.0
                if "right" in bc_data["bcs"]: id_arr[offset + size - 1] = 0.0
                continue
                
            axes = topo.get_axes(d_name)
            strides = topo.get_strides(d_name)
            b_axis = axes[-1]
            coord_sys = topo.domains.get(b_axis, {}).get("coord_sys", "cartesian")
            
            if coord_sys == "unstructured":
                surfaces = layout.mesh_offsets.get(b_axis, {}).get("surfaces", {})
                for side in bc_data["bcs"]:
                    if side in surfaces:
                        mask_off = surfaces[side]
                        for i in range(size):
                            if layout.mesh_cache.get(mask_off + i, 0.0) > 0.5:
                                id_arr[offset + i] = 0.0
                continue
            
            b_res = topo.domains.get(b_axis, {}).get("resolution", 1)
            
            ranges = []
            for axis in axes:
                if axis == b_axis: ranges.append([0]) 
                else: ranges.append(range(topo.domains.get(axis, {}).get("resolution", 1)))
                    
            for indices in itertools.product(*ranges):
                base_flat = 0
                for axis, idx in zip(axes, indices): base_flat += idx * strides[axis]
                if "left" in bc_data["bcs"]: id_arr[offset + base_flat] = 0.0
                if "right" in bc_data["bcs"]: id_arr[offset + base_flat + (b_res - 1) * strides[b_axis]] = 0.0

    # 3. Apply Max Newton Steps Limiters
    for state_name, (offset, size) in layout.state_offsets.items():
        if state_max_steps.get(state_name) is not None:
            val = float(state_max_steps[state_name])
            for i in range(size): 
                max_steps[offset + i] = val

    return id_arr, spatial_diag, max_steps