import pytest
import ion_flux as fx
from ion_flux.compiler.memory import MemoryLayout
from ion_flux.compiler.passes.verification import verify_manifold, TopologicalError

def test_top_down_grid_assembly():
    cell = fx.Domain(bounds=(0.0, 10.0), name="cell")
    reg_A = cell.region(bounds=(0.0, 4.0), resolution=4, name="reg_A")
    reg_B = cell.region(bounds=(4.0, 10.0), resolution=3, name="reg_B")
    
    assert cell.resolution == 7
    assert reg_B.start_idx == 4
    
    c = fx.State(domain=cell, name="c")
    layout = MemoryLayout(states=[c], parameters=[])
    
    m_list = layout.get_mesh_data()
    offsets = layout.mesh_offsets["cell"]
    
    # 7 nodes -> 6 dx_faces, 7 V_nodes, 8 A_faces
    assert "w_dx_faces" in offsets
    assert "w_V_nodes" in offsets
    assert "w_A_faces" in offsets
    
    # Total normalized volumes must sum exactly to 1.0 for cartesian mappings
    vol_off = offsets["w_V_nodes"]
    vols = m_list[vol_off : vol_off + 7]
    assert sum(vols) == pytest.approx(1.0)
    
    # Reg_A physically takes 40% of physical volume (bounds 0 to 4 out of 10)
    assert sum(vols[0:4]) == pytest.approx(0.4)
    # Reg_B physically takes 60% of physical volume
    assert sum(vols[4:7]) == pytest.approx(0.6)

def test_manifold_verification_catches_physical_gaps():
    cell = fx.Domain(bounds=(0.0, 10.0), name="cell")
    
    # Creating a physical gap from 4.0 to 5.0
    reg_A = cell.region(bounds=(0.0, 4.0), resolution=4, name="reg_A")
    reg_B = cell.region(bounds=(5.0, 10.0), resolution=3, name="reg_B")
    
    class MockModel(fx.PDE):
        c = fx.State(domain=cell, name="c")
        def math(self):
            return {"equations": {self.c: fx.dt(self.c) == 0.0}, "boundaries": {}, "initial_conditions": {}}
            
    model = MockModel()
    with pytest.raises(TopologicalError, match="Topological Gap/Overlap Detected"):
        verify_manifold(model.ast())

def test_domain_addition_deprecation():
    d1 = fx.Domain(bounds=(0, 1), resolution=10)
    d2 = fx.Domain(bounds=(1, 2), resolution=10)
    with pytest.raises(TypeError, match="deprecated"):
        d3 = d1 + d2