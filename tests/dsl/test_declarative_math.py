import ion_flux as fx

def test_pde_ast_capture(heat_model):
    ast_dict = heat_model.math()
    
    # Assert Time Derivative Key
    dt_T = fx.dt(heat_model.T)
    assert dt_T in ast_dict
    
    # Assert Spatial Operators in Expression
    rhs_expr = ast_dict[dt_T]
    assert isinstance(rhs_expr, fx.dsl.Addition)
    assert isinstance(rhs_expr.left, fx.dsl.Divergence)
    
    # Assert Boundary Conditions
    assert heat_model.T.left in ast_dict
    assert ast_dict[heat_model.T.left] == 0.0

def test_dae_algebraic_constraint_capture(dae_model):
    ast_dict = dae_model.math()
    
    # fx.dt(c) exists, therefore c is a differential state
    assert fx.dt(dae_model.c) in ast_dict
    
    # V exists plainly as a key, indicating an algebraic constraint (0 = RHS)
    assert dae_model.V in ast_dict
    assert fx.dt(dae_model.V) not in ast_dict
    
    # Evaluate the topology of the constraint
    alg_expr = ast_dict[dae_model.V]
    assert isinstance(alg_expr, fx.dsl.Subtraction)
    assert alg_expr.right == dae_model.c.right

def test_invalid_boundary_condition_raises_error():
    class InvalidModel(fx.PDE):
        rod = fx.Domain(bounds=(0.0, 1.0), resolution=10)
        T = fx.State(domain=rod)
        def math(self):
            return {
                fx.dt(self.T): fx.grad(self.T),
                self.T.top: 0.0 # 'top' is invalid for 1D domain
            }
    
    model = InvalidModel()
    import pytest
    with pytest.raises(fx.errors.TopologyError, match="Invalid boundary 'top' for 1D domain"):
        model.math()