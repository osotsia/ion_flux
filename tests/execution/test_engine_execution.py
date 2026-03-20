from ion_flux import Engine

def test_engine_emits_enzyme_cpp(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    
    cpp = engine.cpp_source
    
    # Verify Enzyme forward declarations exist. 
    assert "extern void __enzyme_fwddiff" in cpp
    assert "void evaluate_residual" in cpp
    assert "void evaluate_jacobian" in cpp

def test_solve_applies_runtime_parameters(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    
    res_default = engine.solve(t_span=(0, 1))
    res_override = engine.solve(t_span=(0, 1), parameters={"k": 2.0})
    
    peak_T_default = max(res_default["T"].data[-1])
    peak_T_override = max(res_override["T"].data[-1])
    
    assert peak_T_override < peak_T_default

def test_simulation_result_serialization(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    res = engine.solve()
    
    json_payload = res.to_dict(variables=["Voltage [V]", "Time [s]"])
    
    assert "Voltage [V]" in json_payload
    assert "Time [s]" in json_payload
    assert "T" not in json_payload
    assert isinstance(json_payload["Voltage [V]"], list)
