import math
from typing import List, Dict, Tuple, Any, Set
from ion_flux.dsl.core import State, Parameter, Observable

class MemoryLayout:
    """
    Translates hierarchical Python AST topologies into strictly contiguous 1D C-Arrays.

    Why: Modern systems-level compilers (LLVM, Rust FFI) require flat memory buffers to 
    achieve vectorization (SIMD) and bypass the Python GIL. This class pre-calculates all 
    memory strides, finite-volume geometries, and unstructured graph pointers Ahead-of-Time.
    """
    def __init__(self, states: List[State], parameters: List[Parameter], 
                 observables: List[Observable] = None, all_domains: List[Any] = None):
        
        # 1. Map Variables to Memory Offsets
        offsets_data = self._compute_variable_offsets(states, parameters, observables)
        self.state_offsets, self.n_states = offsets_data["states"], offsets_data["n_states"]
        self.param_offsets, self.n_params = offsets_data["params"], offsets_data["n_params"]
        self.obs_offsets, self.n_obs = offsets_data["obs"], offsets_data["n_obs"]
        self.p_length = self.n_params

        # 2. Extract Topologies
        structured_roots, unstructured_domains = self._extract_unique_domains(states, observables, all_domains)

        # 3. Build Mesh Geometry Cache
        self.mesh_offsets, self.mesh_cache, self.m_length = self._build_mesh_data(structured_roots, unstructured_domains)


    # =========================================================================
    # Stateless Builders
    # =========================================================================
    
    @staticmethod
    def _compute_variable_offsets(states: List[State], parameters: List[Parameter], observables: List[Observable] = None) -> Dict[str, Any]:
        """Assigns strictly deterministic memory blocks to variables by sorting alphabetically."""
        state_offs, n_states = {}, 0
        for s in sorted(states, key=lambda x: x.name):
            size = s.domain.resolution if s.domain else 1
            state_offs[s.name] = (n_states, size)
            n_states += size

        param_offs, n_params = {}, 0
        for p in sorted(parameters, key=lambda x: x.name):
            param_offs[p.name] = (n_params, 1)
            n_params += 1
            
        obs_offs, n_obs = {}, 0
        if observables:
            for o in sorted(observables, key=lambda x: x.name):
                size = o.domain.resolution if o.domain else 1
                obs_offs[o.name] = (n_obs, size)
                n_obs += size

        return {
            "states": state_offs, "n_states": n_states,
            "params": param_offs, "n_params": n_params,
            "obs": obs_offs, "n_obs": n_obs
        }

    @staticmethod
    def _extract_unique_domains(states: List[State], observables: List[Observable] = None, all_domains: List[Any] = None) -> Tuple[Set[Any], List[Any]]:
        """Walks the AST components to uncover all required spatial grids."""
        structured_roots = set()
        unstructured_domains = []

        def _get_sub_domains(d: Any) -> List[Any]:
            if d is None: return []
            if getattr(d, "type", None) == "composite" or type(d).__name__ == "CompositeDomain":
                res = []
                for sub in getattr(d, "domains", []):
                    res.extend(_get_sub_domains(sub))
                return res
            return [d]

        def _get_root_domain(d: Any) -> Any:
            while getattr(d, "parent", None) is not None:
                d = d.parent
            return d

        # Sweep explicitly bound states
        variables = states + (observables or [])
        for var in variables:
            for d in _get_sub_domains(getattr(var, "domain", None)):
                if getattr(d, "csr_data", None):
                    unstructured_domains.append(d)
                else:
                    structured_roots.add(_get_root_domain(d))
                    
        # Sweep unbound domains (e.g., used exclusively in standalone fx.integral operations)
        for d in (all_domains or []):
            for sub_d in _get_sub_domains(d):
                if getattr(sub_d, "csr_data", None):
                    unstructured_domains.append(sub_d)
                else:
                    structured_roots.add(_get_root_domain(sub_d))

        return structured_roots, unstructured_domains

    @staticmethod
    def _build_mesh_data(structured_roots: Set[Any], unstructured_domains: List[Any]) -> Tuple[Dict[str, Dict[str, int]], Dict[int, float], int]:
        """Packs CSR matrices and computes normalized FVM geometries into a flat C-ABI cache."""
        mesh_offsets = {}
        mesh_cache = {}
        m_length = 0

        # 1. Unstructured Meshes (Graph Data)
        for d in unstructured_domains:
            if d.name in mesh_offsets: continue
            
            csr = d.csr_data
            offsets = {}
            
            for key in ["weights", "row_ptr", "col_ind", "volumes"]:
                if key in csr:
                    offsets[key] = m_length
                    for v in csr[key]:
                        mesh_cache[m_length] = float(v)
                        m_length += 1
            
            offsets["surfaces"] = {}
            for tag, mask in csr.get("surface_masks", {}).items():
                offsets["surfaces"][tag] = m_length
                for v in mask:
                    mesh_cache[m_length] = float(v)
                    m_length += 1
                    
            mesh_offsets[d.name] = offsets

        # 2. Structured Meshes (FVM Geometries)
        for root_d in sorted(list(structured_roots), key=lambda d: d.name):
            if root_d.name in mesh_offsets: continue
                
            coord_sys = root_d.coord_sys
            L_phys = root_d.bounds[1] - root_d.bounds[0]
            
            # Reconstruct piecewise topological regions in strictly increasing order
            regions = root_d._sub_regions if root_d._sub_regions else [root_d]
            regions = sorted(regions, key=lambda r: r.bounds[0])
            
            faces = [0.0]
            centers = []
            
            # Node-Centered Geometry Rule: 
            # Boundary nodes control "Half Volumes" to align perfectly with Dirichlet boundaries.
            # Internal bulk nodes control "Full Volumes".
            num_regions = len(regions)
            
            for i, reg in enumerate(regions):
                L_region_norm = (reg.bounds[1] - reg.bounds[0]) / L_phys if L_phys > 0 else 0.0
                N_nodes = reg.resolution
                
                if num_regions == 1:
                    effective_cells = max(N_nodes - 1.0, 1.0)
                else:
                    if i == 0 or i == num_regions - 1:
                        effective_cells = N_nodes - 0.5
                    else:
                        effective_cells = N_nodes
                        
                du_width = L_region_norm / effective_cells if effective_cells > 0 else 0.0
                
                for j in range(N_nodes):
                    is_first = (i == 0 and j == 0)
                    is_last = (i == num_regions - 1 and j == N_nodes - 1)
                    
                    if is_first and is_last:    # 1-Node Edge Case
                        center = 0.5 * L_region_norm
                        face = L_region_norm
                    elif is_first:              # Left Boundary (Half-Volume)
                        center = 0.0
                        face = 0.5 * du_width
                    elif is_last:               # Right Boundary (Half-Volume)
                        center = 1.0
                        face = 1.0
                    else:                       # Bulk Node (Full-Volume)
                        center = faces[-1] + 0.5 * du_width
                        face = faces[-1] + du_width
                        
                    centers.append(center)
                    faces.append(face)
                    
            # 1. Distances between nodes (dx)
            w_dx_faces = [centers[i+1] - centers[i] for i in range(len(centers) - 1)]
            
            # 2. Cell Volumes
            w_V_nodes = []
            for i in range(len(centers)):
                u_L, u_R = faces[i], faces[i+1]
                if coord_sys == "spherical": vol = (4.0/3.0) * math.pi * (u_R**3 - u_L**3)
                elif coord_sys == "cylindrical": vol = 0.5 * (u_R**2 - u_L**2)
                else: vol = u_R - u_L
                w_V_nodes.append(vol)
                
            # 3. Interfacial Areas
            w_A_faces = []
            for u_f in faces:
                if coord_sys == "spherical": area = 4.0 * math.pi * (u_f**2)
                elif coord_sys == "cylindrical": area = u_f
                else: area = 1.0
                w_A_faces.append(area)
                
            MemoryLayout._verify_fvm_volume(root_d.name, coord_sys, sum(w_V_nodes))
                    
            mesh_offsets[root_d.name] = {}
            for arr_name, arr_data in [("w_dx_faces", w_dx_faces), ("w_V_nodes", w_V_nodes), ("w_A_faces", w_A_faces), ("w_centers", centers)]:
                mesh_offsets[root_d.name][arr_name] = m_length
                for val in arr_data:
                    mesh_cache[m_length] = float(val)
                    m_length += 1

        return mesh_offsets, mesh_cache, m_length

    @staticmethod
    def _verify_fvm_volume(domain_name: str, coord_sys: str, total_volume: float) -> None:
        """Safety check to ensure finite-volume discretization holds exact mass conservation."""
        expected_vol = 1.0
        if coord_sys == "spherical": expected_vol = (4.0/3.0) * math.pi
        elif coord_sys == "cylindrical": expected_vol = 0.5
        
        if abs(total_volume - expected_vol) > 1e-10:
            raise RuntimeError(
                f"Topological Discretization Failure in domain '{domain_name}'. "
                f"Normalized mesh volume integration returned {total_volume}, but expected {expected_vol}."
            )

    # =========================================================================
    # Runtime Retrieval API
    # =========================================================================

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryLayout":
        """Deserialization constructor for 0ms Serverless Cold-Starts."""
        obj = cls.__new__(cls)
        obj.state_offsets = data["state_offsets"]
        obj.param_offsets = data["param_offsets"]
        obj.obs_offsets = data.get("obs_offsets", {})
        
        obj.n_states = data["n_states"]
        obj.n_params = data["n_params"]
        obj.n_obs = data.get("n_obs", 0)
        
        obj.p_length = data.get("p_length", obj.n_params)
        obj.m_length = data.get("m_length", 0)
        
        obj.mesh_offsets = data.get("mesh_offsets", {})
        obj.mesh_cache = {int(k): float(v) for k, v in data.get("mesh_cache", {}).items()}
        return obj

    def get_state_offset(self, name: str) -> int:
        return self.state_offsets[name][0]

    def get_param_offset(self, name: str) -> int:
        return self.param_offsets[name][0]

    def get_mesh_data(self) -> List[float]:
        """Exposes the contiguous flat-array block to pass across the Rust C-ABI pointer."""
        m_list = [0.0] * self.m_length
        for k, v in self.mesh_cache.items():
            m_list[k] = v
        return m_list