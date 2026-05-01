"""
Compiler Bug Oracle: Unbound Domain Integral Context Failure

This suite isolates the AST coordinate broadcasting bug. It proves that calling 
`domain.coords` inside an `fx.integral()` over a domain that has no associated 
`State` must correctly bind the spatial context.

These tests enforce the mathematical truth and will fail until the compiler 
is updated to extract domain data directly from the AST payload rather than 
relying on the state_map.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

# ==============================================================================
# Environment Setup
# ==============================================================================

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

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
    engine = Engine(model=UnboundIntegralModel(), target="cpu", mock_execution=False)
    
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
    engine = Engine(model=BoundIntegralModel(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y, ydot = np.zeros(N).tolist(), np.zeros(N).tolist()
    
    res = engine.evaluate_residual(y, ydot, parameters={})
    
    off_v, _ = engine.layout.state_offsets["V"]
    evaluated_integral = -res[off_v]
    
    # The integral correctly evaluates to the analytical truth of 0.005
    assert evaluated_integral == pytest.approx(0.005), \
        f"Expected exact value 0.005, got {evaluated_integral}. Context injection failed."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])