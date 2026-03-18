from ion_flux import Engine

def test_engine_emits_enzyme_cpp(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    
    cpp = engine.cpp_source
    
    # Verify Enzyme forward declarations exist
    assert "extern void __enzyme_autodiff" in cpp
    
    # Verify C-ABI hooks for SUNDIALS exist
    assert "void evaluate_residual" in cpp
    assert "void evaluate_jacobian" in cpp
    
    # Verify the math was successfully lowered to C++ assignment
    assert "res[0] = ydot[0] - " in cpp
    assert "DIV(((-p_k) * GRAD(y_T)))" in cpp
    assert "std::abs((X_COORD - 1.000000))" in cpp
    
    # Verify boundary conditions were mapped
    assert "res[1] = 0.000000;" in cpp
    assert "left Boundary for T" in cpp

def test_solve_applies_runtime_parameters(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    
    # Test the mock solver returns different values based on parameters
    res_default = engine.solve(t_span=(0, 1))
    res_override = engine.solve(t_span=(0, 1), parameters={"k": 2.0})
    
    peak_T_default = max(res_default["T"].data[-1])
    peak_T_override = max(res_override["T"].data[-1])
    
    assert peak_T_override < peak_T_default