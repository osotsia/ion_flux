import pytest
import numpy as np
import shutil
import platform
import os
import sys
import ion_flux as fx

# Ensure models directory is in path for E2E tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)

import pytest
import asyncio
import os
import shutil
import platform
import numpy as np
import ion_flux as fx
from ion_flux.runtime.scheduler import MultiTenantScheduler
from ion_flux.protocols import Sequence, CC, CV, Rest
from ion_flux._core import solve_ida_native



tetrahedron_mesh = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "elements": [[0, 1, 2, 3]]
}



class UnstructuredGMRESModel(fx.PDE):
    """Validates 3D Matrix-Free CSR Graph Traversals (bandwidth = -1)."""
    mesh = fx.Domain.from_mesh(tetrahedron_mesh, name="mesh", surfaces={"top": [2, 3]})
    c = fx.State(domain=mesh, name="c")
    D = fx.Parameter(default=2.0, name="D")

    def math(self):
        flux = -self.D * fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux)
            },
            "boundaries": {
                flux: {"top": 100.0}
            },
            "initial_conditions": {
                self.c: 10.0
            }
        }




# ==============================================================================
# Concept 3: Advanced Architectures (GMRES & Unstructured Meshes)
# ==============================================================================

@REQUIRES_RUNTIME
def test_3d_unstructured_matrix_free_gmres():
    """
    Validates that unstructured meshes automatically trigger Matrix-Free GMRES 
    (bandwidth=-1), correctly traverse CSR geometries, and support Adjoint passes without OOM.
    """
    # Cache=False forces the Engine to re-emit the JVP C++ payload natively
    engine = fx.Engine(model=UnstructuredGMRESModel(), target="cpu", mock_execution=False, cache=False)
    
    assert engine.jacobian_bandwidth == -1, "Engine failed to assign GMRES to unstructured CSR graph."
    
    res = engine.solve(t_span=(0, 1.0), requires_grad=["D"])
    
    assert res.status == "completed"
    assert res["c"].data.shape[1] == 4 # Validates dynamic unrolling to the exact 4-node tetrahedron mesh
    
    # Backward pass over GMRES trajectory
    loss = fx.metrics.rmse(res["c"], np.zeros_like(res["c"].data), engine=engine, state_name="c")
    grads = loss.backward()
    
    assert isinstance(grads["D"], float)
    assert not np.isnan(grads["D"])
