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


def test_unstructured_execution_mock(mesh_3d_tetrahedron):
    """
    Simulates a full initialization of an execution block utilizing unstructured grid layouts.
    Bypasses true execution assuming compiler dependencies might be stripped on runner.
    """
    class UnstructuredModel(fx.PDE):
        mesh = fx.Domain.from_mesh(mesh_3d_tetrahedron, name="mesh3d", surfaces={"top": [2, 3]})
        c = fx.State(domain=mesh)
        D = fx.Parameter(default=2.0)

        def math(self):
            return {
                fx.dt(self.c): fx.div(self.D * fx.grad(self.c)),
                self.c.boundary("top"): 100.0, # Target specific surfaces
                self.c.t0: 100.0
            }

    engine = Engine(model=UnstructuredModel(), target="cpu", mock_execution=True)
    res = engine.solve()

    assert res.status == "completed"
    
    # Ensure Memory Layout and mock routing handled multi-dimensional structures successfully 
    # instead of strictly flat variables, preventing the KeyError
    assert "c" in res._data
    assert res["c"].data.shape[1] == 4