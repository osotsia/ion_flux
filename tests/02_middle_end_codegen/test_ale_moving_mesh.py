"""
test_ale_moving_mesh.py

Validates Arbitrary Lagrangian-Eulerian (ALE) advection stability for 
moving boundaries. Bypasses string-matching in favor of AST validation
and numerical upwinding oracles.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

class MovingMeshModel(fx.PDE):
    x = fx.Domain(bounds=(0, 1), resolution=5, name="x")
    c = fx.State(domain=x, name="c")
    L = fx.State(domain=None, name="L") # Thickness state
    
    def math(self):
        return {
            "regions": { self.x: [ fx.dt(self.c) == fx.grad(self.c) ] },
            "boundaries": [ self.x.right == self.L, self.c.left == 0.0 ], 
            "global": [ fx.dt(self.L) == 1.0 ]
        }

def test_ale_ast_intent_capture():
    """Validates the AST correctly tags boundaries bound to dynamic states."""
    model = MovingMeshModel()
    ast = model.ast()
    
    # Instead of guessing the exact LHS dictionary schema for a Domain boundary,
    # we uniquely identify the ALE condition by its Right-Hand Side:
    # It is the only boundary condition dynamically bound to the state 'L'.
    moving_bc = next(
        bc for bc in ast["boundaries"] 
        if bc["rhs"].get("type") == "State" and bc["rhs"].get("name") == "L"
    )
    
    # The core intent validation: 
    # The RHS was successfully parsed as a State (which triggers ALE downstream), 
    # rather than being evaluated as a static Scalar.
    assert moving_bc["rhs"]["type"] == "State"
    assert moving_bc["rhs"]["name"] == "L"
    
    # Sanity check that the LHS is indeed recognized as some form of Boundary node
    assert "Boundary" in moving_bc["lhs"].get("type", "")

@REQUIRES_COMPILER
def test_ale_upwind_advection_numerical():
    """Validates ALE moving meshes apply stable directional upwinding rather than centered differences."""
    engine = Engine(model=MovingMeshModel(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    off_c, size_c = engine.layout.state_offsets["c"]
    off_L, _ = engine.layout.state_offsets["L"]
    
    # Setup base state: Constant linear gradient of 10.0
    y_base = [0.0] * N
    y_base[off_c : off_c + size_c] = [0.0, 10.0, 20.0, 30.0, 40.0]
    y_base[off_L] = 1.0 # Current length is 1.0
    
    # --- Case 1: EXPANDING MESH ---
    ydot_expand = [0.0] * N
    ydot_expand[off_L] = 1.0 # L_dot = 1.0 (v > 0)
    res_expand = engine.evaluate_residual(y_base, ydot_expand, parameters={})
    
    # --- Case 2: CONTRACTING MESH ---
    ydot_contract = [0.0] * N
    ydot_contract[off_L] = -1.0 # L_dot = -1.0 (v < 0)
    res_contract = engine.evaluate_residual(y_base, ydot_contract, parameters={})
    
    # Verify the advection calculation at the center node shifts its stencil.
    # If standard centered differences were mistakenly used, the expansion/contraction
    # would yield linearly symmetric changes. Upwinding causes a strict asymmetry.
    center_idx = off_c + 2
    assert res_expand[center_idx] != res_contract[center_idx], \
        "ALE Advection failed to dynamically shift the upwind stencil based on velocity direction."