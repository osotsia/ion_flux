"""
Frontend DSL: Semantic Validation

Tests the strict validation barriers applied to the AST before compilation.
Ensures structural integrity to prevent fatal unwraps downstream.
"""

import pytest
import ion_flux as fx
from ion_flux.dsl.ast import validate_ast

# --- Fixtures & Mock Models ---

class SharedScopeDiffusion(fx.PDE):
    """Models a simple 1D diffusion to test semantic buckets and shared Python scoping."""
    c = fx.State()
    D = fx.Parameter(1.5)
    
    def math(self):
        flux = -self.D * fx.grad(self.c)
        return {
            "global": [
                fx.dt(self.c) == -fx.div(flux),
                self.c.t0 == 0.5
            ],
            "boundaries": [
                flux.left == 0.0,
                flux.right == 2.5
            ]
        }

class TerminalInjectionModel(fx.PDE):
    """Models a simple zero-dimensional algebraic constraint to test hardware multiplexing."""
    i_app = fx.State()
    V_cell = fx.State()
    R_internal = fx.Parameter(0.05)
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        return {
            "global": [
                self.V_cell == 4.2 - self.R_internal * self.i_app,
                self.V_cell.t0 == 4.2
            ]
        }

# --- Test Cases ---

def test_semantic_bucket_parsing():
    """Validates that the base PDE class correctly unpacks the TypedDict into the AST payload."""
    model = SharedScopeDiffusion()
    ast_payload = model.ast()
    
    assert set(ast_payload.keys()) == {"regions", "global", "boundaries"}
    assert len(ast_payload["global"]) == 2
    assert len(ast_payload["boundaries"]) == 2
    
    pde_eq = ast_payload["global"][0]
    assert pde_eq["lhs"]["type"] == "UnaryOp"
    assert pde_eq["lhs"]["op"] == "dt"
    assert pde_eq["lhs"]["child"]["name"] == "c"

def test_shared_scope_resolution():
    """Validates that intermediate math variables (like flux) resolve cleanly across buckets."""
    model = SharedScopeDiffusion()
    ast_payload = model.ast()
    
    pde_rhs = ast_payload["global"][0]["rhs"]
    assert pde_rhs["type"] == "UnaryOp"
    assert pde_rhs["op"] == "neg"
    assert pde_rhs["child"]["op"] == "div"
    
    bc_lhs = ast_payload["boundaries"][0]["lhs"]
    assert bc_lhs["type"] == "Boundary"
    assert bc_lhs["side"] == "left"
    assert bc_lhs["child"]["type"] == "BinaryOp" 
    assert bc_lhs["child"]["op"] == "mul"

def test_terminal_multiplexer_injection():
    model = TerminalInjectionModel()
    ast_payload = model.ast()
    
    # User declared 2 globals (1 eq, 1 IC). Terminal injects 1 more constraint.
    assert len(ast_payload["global"]) == 3
    
    user_eq = ast_payload["global"][0]
    injected_eq = ast_payload["global"][2]
    
    assert user_eq["lhs"]["name"] == "V_cell"
    assert injected_eq["lhs"]["name"] == "i_app"
    
    # Verify the multiplexer structure exists in the RHS of the injected equation
    rhs = injected_eq["rhs"]
    assert rhs["type"] == "BinaryOp"
    assert rhs["op"] == "add" 

def test_ast_validation_rejects_missing_equality():
    """Prevents researchers from accidentally passing assignments instead of equations."""
    class BadEqModel(fx.PDE):
        c = fx.State()
        def math(self):
            return {"global": [self.c.t0]} 
            
    with pytest.raises(ValueError, match="Equations in 'global' must use '=='"):
        BadEqModel().ast()

def test_compiler_payload_integrity():
    """Tests the strict pre-compilation validation barrier."""
    assert validate_ast({"regions": {"x": [{"lhs": {"type": "State"}, "rhs": {"type": "Scalar"}}]}}) is True
    
    with pytest.raises(ValueError, match="Invalid equation bucket: 'pdes'"):
        validate_ast({"pdes": [{"lhs": {"type": "State"}, "rhs": {"type": "Scalar"}}]})
        
    with pytest.raises(ValueError, match="must contain a 'lhs' and 'rhs'"):
        validate_ast({"boundaries": [{"left_side": {"type": "State"}, "right_side": {"type": "Scalar"}}]})