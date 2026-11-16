import json
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np


def assemble_mesh_csr(
    mesh_data: Union[str, dict],
    surfaces: Optional[Dict[str, List[int]]] = None
) -> Tuple[int, Dict[str, Any]]:
    """
    Assembles node volumes, surface masks, and FEM stiffness CSR weights for tetrahedral meshes.
    Separates numerical matrix assembly from declarative domain definitions.
    """
    if isinstance(mesh_data, str):
        with open(mesh_data, "r") as f:
            mesh_data = json.load(f)

    nodes = np.array(mesh_data["nodes"], dtype=float)
    elements = np.array(mesh_data["elements"], dtype=int)
    resolution = len(nodes)

    k_global = defaultdict(float)
    v_nodes = np.zeros(resolution, dtype=float)

    for el in elements:
        if len(el) != 4:
            continue
        p = nodes[el]
        j_mat = np.array([p[1] - p[0], p[2] - p[0], p[3] - p[0]]).T
        det_j = np.linalg.det(j_mat)
        vol = abs(det_j) / 6.0
        if vol < 1e-30:
            continue

        inv_j = np.linalg.inv(j_mat)
        grad_n = np.zeros((4, 3))
        grad_n[1:4, :] = inv_j.T
        grad_n[0, :] = -np.sum(inv_j.T, axis=0)

        k_local = vol * (grad_n @ grad_n.T)

        for i in range(4):
            v_nodes[el[i]] += vol / 4.0
            for j in range(4):
                if i != j:
                    k_global[(el[i], el[j])] -= k_local[i, j]

    row_ptr = [0] * (resolution + 1)
    col_ind, weights = [], []

    for i in range(resolution):
        row_ptr[i] = len(col_ind)
        vol_i = max(v_nodes[i], 1e-30)
        for j in range(resolution):
            if (i, j) in k_global:
                col_ind.append(j)
                weights.append(k_global[(i, j)] / vol_i)
    row_ptr[resolution] = len(col_ind)

    surface_masks = {}
    if surfaces:
        for tag, indices in surfaces.items():
            mask = [0.0] * resolution
            for idx in indices:
                mask[idx] = 1.0
            surface_masks[tag] = mask

    csr_data = {
        "row_ptr": [float(x) for x in row_ptr],
        "col_ind": [float(x) for x in col_ind],
        "weights": weights,
        "volumes": [float(x) for x in v_nodes],
        "surface_masks": surface_masks,
    }
    return resolution, csr_data
