import os
import tempfile
import itertools
from typing import Dict, Any, List, Optional

from ion_flux.stage1_dsl.core import PDE, State, Parameter, Observable
from ion_flux.stage1_dsl.spatial import Domain, CompositeDomain
from ion_flux.stage2_compiler._1_analysis.memory_layout import MemoryLayout
from ion_flux.stage2_compiler.pipeline import Compiler
from ion_flux.stage3_backend.clang_invoker import NativeCompiler
from ion_flux.runtime.manifest import ExecutableManifest
from ion_flux.stage2_compiler._1_analysis.topology import TopologyAnalyzer

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
    lib_path = ""
    cpp_source = ""
    cpr_cache = ([], [], [], [], [])
    
    if ast_payload:
        compiler_result = Compiler.compile(ast_payload, layout, states, observables, target, jacobian_bandwidth)
        
        cpp_source = compiler_result["cpp_source"]
        cpr_cache = compiler_result["cpr_cache"]
        ast_payload = compiler_result["ast_payload"]
        topo = compiler_result["topo"]
        jacobian_bandwidth = compiler_result["jacobian_bandwidth"]
        
        if not mock_execution:
            compiler = NativeCompiler() if cache else NativeCompiler(cache_dir=os.path.join(tempfile.gettempdir(), "nocache"))
            runtime = compiler.compile(cpp_source, layout.n_states)
            lib_path = runtime.lib_path

    id_arr, spatial_diag, max_steps = _compute_static_metadata(layout, ast_payload, topo, state_domain_map, state_max_steps)

    return ExecutableManifest(
        lib_path=lib_path,
        layout=layout,
        default_parameters=default_parameters,
        ast_payload=ast_payload,
        jacobian_bandwidth=jacobian_bandwidth or 0,
        cpr_cache=cpr_cache,
        id_arr=id_arr,
        spatial_diag=spatial_diag,
        max_steps=max_steps,
        state_domain_map=state_domain_map,
        cpp_source=cpp_source
    )

def _compute_static_metadata(layout: MemoryLayout, ast_payload: Dict[str, Any], topo: Optional[TopologyAnalyzer], state_domain_map: Dict[str, str], state_max_steps: Dict[str, Any]):
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