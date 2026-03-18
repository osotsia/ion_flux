import ion_flux as fx

def test_pde_ast_capture(heat_model):
    ast = heat_model.ast()
    
    # Check that 4 equations were captured (PDE, 2 BCs, 1 IC)
    assert len(ast) == 4
    
    # Find the time derivative equation
    pde_eq = next(eq for eq in ast if eq["lhs"].get("op") == "dt")
    assert pde_eq["lhs"]["child"]["name"] == "T"
    
    # Check that the RHS subtraction was captured correctly
    assert pde_eq["rhs"]["type"] == "BinaryOp"
    assert pde_eq["rhs"]["op"] == "add" # -div(flux) + source