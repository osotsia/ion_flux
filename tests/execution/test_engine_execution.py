import pytest
from ion_flux import Engine

def test_engine_initializes_cpu_target(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    assert engine.target == "cpu"
    assert engine.is_compiled is True

def test_engine_initializes_openmp_target(heat_model):
    engine = Engine(model=heat_model, target="cpu:omp", threads_per_model=4)
    assert engine.target == "cpu:omp"
    assert engine.threads_per_model == 4

def test_solve_applies_runtime_parameters(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    
    # Solve with default parameter (k=0.75)
    res_default = engine.solve(t_span=(0, 1))
    
    # Solve with override (k=2.0)
    res_override = engine.solve(t_span=(0, 1), parameters={"k": 2.0})
    
    # Higher diffusivity should result in lower peak temperature
    peak_T_default = max(res_default["T"].data[-1])
    peak_T_override = max(res_override["T"].data[-1])
    
    assert peak_T_override < peak_T_default

def test_solve_batch_tasks_parallelism(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    
    param_sets = [{"k": 0.5}, {"k": 1.0}, {"k": 1.5}]
    
    results = engine.solve_batch(
        t_span=(0, 1), 
        parameters=param_sets,
        max_workers=3
    )
    
    assert len(results) == 3
    assert results[0].parameters["k"] == 0.5
    assert results[2].parameters["k"] == 1.5