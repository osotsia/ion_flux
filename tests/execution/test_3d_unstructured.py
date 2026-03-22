import pytest
import numpy as np
import ion_flux as fx
from ion_flux.compiler.memory import MemoryLayout
from ion_flux.compiler.codegen import generate_cpp
from ion_flux import Engine

@pytest.fixture
def mesh_3d_tetrahedron():
    return {
        "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "elements": [[0, 1, 2, 3]]
    }

def test_unstructured_3d_fem_mesh_codegen(mesh_3d_tetrahedron):
    """
    Validates C++ CodeGen translates abstract divergence operators into discrete
    CSR matrix graph traversals decoupled from compile-time sizing.
    """
    class UnstructuredModel(fx.PDE):
        mesh = fx.Domain.from_mesh(mesh_3d_tetrahedron, name="mesh3d", surfaces={"top": [2, 3]})
        c = fx.State(domain=mesh)
        D = fx.Parameter(default=2.0)
        
        def math(self):
            flux = -self.D * fx.grad(self.c)
            return { 
                fx.dt(self.c): -fx.div(flux),
                flux.boundary("top"): 0.5
            }

    model = UnstructuredModel()
    states = [model.c]
    layout = MemoryLayout(states, [model.D])
    
    cpp = generate_cpp(model.ast(), layout, states)
    
    # 1. Verify the static memory block packing is completely removed
    assert "static const int row_ptr_mesh3d" not in cpp
    
    # 2. Verify contextual dynamic index fetching mapped via p parameters
    assert "int start = (int)p[" in cpp
    assert "double w = p[" in cpp
    
    # 3. Verify unstructured surface mask boundary injection via > 0.5 
    assert " > 0.5 ?" in cpp


def test_unstructured_execution_matrix_free_jfnk(mesh_3d_tetrahedron):
    """
    Tests full native execution utilizing the newly implemented Rust Matrix-Free GMRES 
    Krylov subspace solver to evaluate J*v products directly.
    """
    class UnstructuredModel(fx.PDE):
        mesh = fx.Domain.from_mesh(mesh_3d_tetrahedron, name="mesh3d", surfaces={"top": [2, 3]})
        c = fx.State(domain=mesh)
        D = fx.Parameter(default=2.0)

        def math(self):
            return {
                fx.dt(self.c): fx.div(self.D * fx.grad(self.c)),
                self.c.boundary("top"): 100.0,
                self.c.t0: 100.0
            }

    # Set cache=False to force LLVM to re-emit the analytical evaluate_jvp function newly added to templates.
    engine = Engine(model=UnstructuredModel(), target="cpu", mock_execution=False, cache=False)
    
    if getattr(engine, "mock_execution", False):
        pytest.skip("Native compiler toolchain not present on this host.")

    # Validation that the orchestrator automatically mapped the model correctly to GMRES
    assert engine.jacobian_bandwidth == -1

    res = engine.solve(t_span=(0, 1.0))

    assert res.status == "completed"
    assert "c" in res._data
    # Confirms the 4-node tetrahedron mesh mapped dynamically back out correctly to Numpy
    assert res["c"].data.shape[1] == 4