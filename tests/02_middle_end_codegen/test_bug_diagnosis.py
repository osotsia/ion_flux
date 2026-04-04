import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine

# ==============================================================================
# Model 1: Piecewise Equation (Works correctly)
# ==============================================================================
class PiecewiseBoundaryModel(fx.PDE):
    cell = fx.Domain(bounds=(0, 10.0), resolution=11)
    reg_A = cell.region(bounds=(0, 5.0), resolution=6, name="reg_A")
    reg_B = cell.region(bounds=(5.0, 10.0), resolution=6, name="reg_B")
    
    c = fx.State(domain=cell, name="c") # Bound to PARENT
    
    def math(self):
        flux = -fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_A: fx.dt(self.c) == -fx.div(flux),
                    self.reg_B: fx.dt(self.c) == -fx.div(flux)
                })
            },
            "boundaries": {
                flux: {"right": 100.0} # Massive flux
            },
            "initial_conditions": {self.c: 0.0}
        }

# ==============================================================================
# Model 2: Standard Equation on Region (The Bug Suspect)
# ==============================================================================
class StandardRegionBoundaryModel(fx.PDE):
    cell = fx.Domain(bounds=(0, 10.0), resolution=11)
    reg = cell.region(bounds=(5.0, 10.0), resolution=6, name="reg")
    
    c_reg = fx.State(domain=reg, name="c_reg") # Bound to REGION
    
    def math(self):
        flux = -fx.grad(self.c_reg)
        return {
            "equations": {
                # Standard equation (not piecewise!)
                self.c_reg: fx.dt(self.c_reg) == -fx.div(flux)
            },
            "boundaries": {
                flux: {"right": 100.0} # Massive flux
            },
            "initial_conditions": {self.c_reg: 0.0}
        }

# ==============================================================================
# Model 3: The Mask Workaround (Proving the fix)
# ==============================================================================
class WorkaroundRegionBoundaryModel(fx.PDE):
    cell = fx.Domain(bounds=(0, 10.0), resolution=11)
    reg = cell.region(bounds=(5.0, 10.0), resolution=6, name="reg")
    
    c_reg = fx.State(domain=reg, name="c_reg")
    
    def math(self):
        # 1. Create a spatial mask that is 1.0 ONLY at the rightmost node
        # Cell dx = 1.0. Region bounds are 5.0 to 10.0. Rightmost node is > 9.0.
        mask = (self.reg.coords > 9.0)
        
        dx = 1.0
        V_node = 0.5 * dx
        
        # 2. Inject the boundary flux manually into the divergence equation
        div_flux_corrected = fx.div(-fx.grad(self.c_reg)) + mask * (100.0 / V_node)
        
        return {
            "equations": {
                self.c_reg: fx.dt(self.c_reg) == -div_flux_corrected
            },
            "boundaries": {},
            "initial_conditions": {self.c_reg: 0.0}
        }

# ==============================================================================
# Tests
# ==============================================================================

def test_piecewise_boundary_evaluates_correctly():
    """Proves that Piecewise loops evaluate global bounds correctly."""
    engine = Engine(model=PiecewiseBoundaryModel(), target="cpu", mock_execution=False)
    y, ydot = np.zeros(engine.layout.n_states), np.zeros(engine.layout.n_states)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    # Expected residual at the right boundary: F = ydot - (-div(flux))
    # div(flux) = (flux_out - flux_in) / V_node = (100.0 - 0.0) / 0.5 = 200.0
    assert res[-1] == pytest.approx(200.0), "Piecewise boundary failed!"


def test_standard_region_boundary_is_ignored():
    """
    THE ORACLE: Proves that binding a standard equation to a region 
    causes the compiler to silently ignore the right-hand boundary condition.
    """
    engine = Engine(model=StandardRegionBoundaryModel(), target="cpu", mock_execution=False)
    y, ydot = np.zeros(engine.layout.n_states), np.zeros(engine.layout.n_states)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    print(f"\nResidual Array: {res}")
    
    # If the bug exists, the compiler skips the boundary check, 
    # evaluates standard gradient (which is 0.0 since y=0), and returns 0.0.
    bug_is_present = (abs(res[-1]) < 1e-5)
    
    assert not bug_is_present, "The bug is present. The boundary was not correctly evaluated (Residual is 0.0)."


def test_spatial_mask_workaround_succeeds():
    """Proves the FVM spatial mask workaround successfully injects the flux."""
    engine = Engine(model=WorkaroundRegionBoundaryModel(), target="cpu", mock_execution=False)
    y, ydot = np.zeros(engine.layout.n_states), np.zeros(engine.layout.n_states)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    # The mask should inject exactly 200.0 into the final node.
    assert res[-1] == pytest.approx(200.0), "The manual mask workaround failed!"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])