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
import numpy as np
import ion_flux as fx
import shutil
import platform



# ==============================================================================
# Model 1: Piecewise Equation (Works correctly)
# ==============================================================================
class PiecewiseBoundaryModel(fx.PDE):
    cell = fx.Domain(bounds=(0, 10.0), resolution=10)
    reg_A = cell.region(bounds=(0, 5.0), resolution=5, name="reg_A")
    reg_B = cell.region(bounds=(5.0, 10.0), resolution=5, name="reg_B")
    
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
    cell = fx.Domain(bounds=(0, 10.0), resolution=10)
    reg_A = cell.region(bounds=(0, 5.0), resolution=5, name="reg_A")
    reg = cell.region(bounds=(5.0, 10.0), resolution=5, name="reg")
    
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
    cell = fx.Domain(bounds=(0, 10.0), resolution=10)
    reg_A = cell.region(bounds=(0, 5.0), resolution=5, name="reg_A")
    reg = cell.region(bounds=(5.0, 10.0), resolution=5, name="reg")
    
    c_reg = fx.State(domain=reg, name="c_reg")
    
    def math(self):
        # 1. Create a spatial mask that is 1.0 ONLY at the rightmost node
        # Cell dx = 10.0/9.0. Rightmost node is 10.0.
        mask = (self.reg.coords > 9.0)
        
        dx = 10.0 / 9.0
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
    engine = fx.Engine(model=PiecewiseBoundaryModel(), target="cpu", mock_execution=False)
    y, ydot = np.zeros(engine.layout.n_states), np.zeros(engine.layout.n_states)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    dx = 10.0 / 9.0
    expected_res = 100.0 / (0.5 * dx)
    assert res[-1] == pytest.approx(expected_res), "Piecewise boundary failed!"




def test_standard_region_boundary_is_ignored():
    """
    THE ORACLE: Proves that binding a standard equation to a region 
    causes the compiler to silently ignore the right-hand boundary condition.
    """
    engine = fx.Engine(model=StandardRegionBoundaryModel(), target="cpu", mock_execution=False)
    y, ydot = np.zeros(engine.layout.n_states), np.zeros(engine.layout.n_states)
    
    res = engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters={})
    
    print(f"\nResidual Array: {res}")
    
    # If the bug exists, the compiler skips the boundary check, 
    # evaluates standard gradient (which is 0.0 since y=0), and returns 0.0.
    bug_is_present = (abs(res[-1]) < 1e-5)
    
    assert not bug_is_present, "The bug is present. The boundary was not correctly evaluated (Residual is 0.0)."


"""
test_piecewise_micro_routing_oracle.py

Compiler Bug Oracle: Global Piecewise to Micro-Domain Index Routing

This suite proves that when a micro-domain surface boundary (c_surf) is evaluated 
inside a global Piecewise block (e.g. for the electrolyte c_e), the loop variable `idx` 
is a global topological index (e.g. 81-143). The compiler incorrectly routes this 
directly into the micro-domain's local flat array, resulting in severe data corruption.
"""



class PiecewiseMicroRoutingOracle(fx.PDE):
    # Mimics O'Regan LG M50 topology
    cell = fx.Domain(bounds=(0, 100), resolution=144)
    x_n = cell.region(bounds=(0, 40), resolution=71, name="x_n")
    x_s = cell.region(bounds=(40, 50), resolution=10, name="x_s")
    x_p = cell.region(bounds=(50, 100), resolution=63, name="x_p")
    
    r_p = fx.Domain(bounds=(0, 5), resolution=10, name="r_p")
    
    # State bound to the sub-mesh composite domain (Size: 63 * 10 = 630)
    c_s_p = fx.State(domain=x_p * r_p, name="c_s_p")
    
    # State bound to the global cell (Size: 144)
    c_e = fx.State(domain=cell, name="c_e")
    
    def math(self):
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p)
        
        return {
            "equations": {
                # Lock original state
                self.c_s_p: fx.dt(self.c_s_p) == 0.0,
                
                # Expose the evaluation of c_surf_p inside a global Piecewise loop
                self.c_e: fx.Piecewise({
                    self.x_n: fx.dt(self.c_e) == 0.0,
                    self.x_s: fx.dt(self.c_e) == 0.0,
                    
                    # Inside the x_p loop, the C++ 'idx' goes from 81 to 143.
                    # We store c_surf_p directly into c_e's derivative to extract it.
                    self.x_p: fx.dt(self.c_e) == c_surf_p
                })
            },
            "boundaries": {},
            "initial_conditions": {
                self.c_s_p: 0.0,
                self.c_e: 0.0
            }
        }



@REQUIRES_COMPILER
def test_piecewise_micro_domain_routing():
    model = PiecewiseMicroRoutingOracle()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y0, ydot0, _, _, _ = engine._extract_metadata()
    y0 = np.array(y0)
    
    # Create a distinct value for EVERY micro node so we know exactly which one is read
    off_csp, size_csp = engine.layout.state_offsets["c_s_p"]
    y0[off_csp : off_csp + size_csp] = np.arange(size_csp)
    
    # The true physical surface values are at local indices 9, 19, 29, ..., 629
    true_surface_vals = np.arange(9, model.x_p.resolution * 10, 10)
    
    res = engine.evaluate_residual(y0.tolist(), ydot0, parameters={})
    
    # Extract the values evaluated during the Piecewise loop
    off_ce, size_ce = engine.layout.state_offsets["c_e"]
        
    # We must extract using the exact start_idx calculated by the compiler
    start = model.x_p.start_idx
    end = start + model.x_p.resolution
    
    # The residual equation is F = ydot - rhs = 0.0 - c_surf_p -> rhs = -F
    extracted_surfaces = -np.array(res[off_ce + start : off_ce + end])
    
    is_corrupted = not np.allclose(extracted_surfaces, true_surface_vals)
    
    assert not is_corrupted, \
        f"\nBUG CONFIRMED: The AST compiler incorrectly routed the macro-to-micro " \
        f"surface extraction inside a Piecewise loop!\n" \
        f"Expected: {true_surface_vals[:5]}...\n" \
        f"Actual:   {extracted_surfaces[:5]}...\n" \
        f"This destroys the electrolyte concentration gradient inside the cathode."



# ==============================================================================
# Models for Isolation
# ==============================================================================

class UnboundIntegralModel(fx.PDE):
    """
    Integrates over a domain `x` that has no `State` bound to it.
    If the compiler bug is present, `_lower_integral` fails to find `x` 
    in the state_map, falling back to `dx_default = 1.0`.
    """
    x = fx.Domain(bounds=(0, 0.1), resolution=11, name="x")
    
    # 0D state. No states are bound to `x`.
    V = fx.State(domain=None, name="V")

    def math(self):
        return {
            "equations": {
                # Integrate the coordinate `x` from 0 to 0.1
                self.V: fx.dt(self.V) == fx.integral(self.x.coords, over=self.x)
            },
            "boundaries": {},
            "initial_conditions": {self.V: 0.0}
        }



class BoundIntegralModel(fx.PDE):
    """
    Control Model: Introduces a dummy state bound to `x`.
    The compiler will successfully find `x` in the state_map, inject the correct
    context, and scale the coordinates accurately.
    """
    x = fx.Domain(bounds=(0, 0.1), resolution=11, name="x")
    
    # Dummy state acts as a topological anchor for the compiler context
    dummy_anchor = fx.State(domain=x, name="dummy_anchor")
    V = fx.State(domain=None, name="V")

    def math(self):
        return {
            "equations": {
                self.dummy_anchor: fx.dt(self.dummy_anchor) == 0.0,
                self.V: fx.dt(self.V) == fx.integral(self.x.coords, over=self.x)
            },
            "boundaries": {},
            "initial_conditions": {
                self.dummy_anchor: 0.0,
                self.V: 0.0
            }
        }



# ==============================================================================
# Tests
# ==============================================================================

@REQUIRES_COMPILER
def test_unbound_domain_integral_context_resolution():
    """
    Proves that `fx.integral()` correctly resolves the spatial context and 
    evaluates `domain.coords` accurately, even if no State is explicitly 
    bound to that domain.
    
    Analytical Integral of x dx from 0 to 0.1 = 0.5 * (0.1)^2 = 0.005.
    
    """
    engine = fx.Engine(model=UnboundIntegralModel(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y, ydot = np.zeros(N).tolist(), np.zeros(N).tolist()
    
    # Evaluate the instantaneous residual: res = ydot - (integral) = 0 - integral
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    off_v, _ = engine.layout.state_offsets["V"]
    evaluated_integral = -res[off_v]
    
    # We assert the CORRECT mathematical truth. 
    # If the compiler bug exists (dx_default=1.0), this will evaluate to 0.5 and fail.
    assert evaluated_integral == pytest.approx(0.005), \
        f"Compiler Bug: Expected exact analytical integral 0.005, but got {evaluated_integral}. " \
        "Context injection failed for unbound domain."




@REQUIRES_COMPILER
def test_bound_domain_integral_correctness():
    """
    Proves that anchoring a State to the domain currently serves as a valid 
    workaround to restore mathematical exactness to the integral.
    """
    engine = fx.Engine(model=BoundIntegralModel(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y, ydot = np.zeros(N).tolist(), np.zeros(N).tolist()
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    off_v, _ = engine.layout.state_offsets["V"]
    evaluated_integral = -res[off_v]
    
    # The integral correctly evaluates to the analytical truth of 0.005
    assert evaluated_integral == pytest.approx(0.005), \
        f"Expected exact value 0.005, got {evaluated_integral}. Context injection failed."
