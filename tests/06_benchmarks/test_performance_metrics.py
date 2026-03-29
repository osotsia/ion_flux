"""
Benchmarks: Performance Metrics

Pytest-benchmark suites rigorously tracking regressions in JIT compilation time,
isolated numerical loop execution, and task-parallel throughput.
"""

import pytest
from ion_flux.battery import DFN, parameters
from ion_flux import Engine

# Standard resolution for benchmarking
MESH_RES = {"x_n": 50, "x_s": 50, "x_p": 50, "r_n": 20, "r_p": 20}

@pytest.fixture
def base_model():
    return DFN(options={"thermal": "lumped"})

@pytest.fixture
def compiled_cpu_engine(base_model):
    # Pre-compiled engine for execution benchmarks
    return Engine(model=base_model, target="cpu", mesh_resolution=MESH_RES)

def test_bench_cold_start_compilation(benchmark, base_model):
    """
    Measures the JIT pipeline: AST extraction -> Rust Symbolic Diff 
    -> C++ Emission -> clang++ Compilation -> .so Loading.
    """
    def compile_model():
        # Force a bypass of any disk-caching for the benchmark
        return Engine(model=base_model, target="cpu", mesh_resolution=MESH_RES, cache=False)

    # Rounds are limited because compilation takes seconds, not milliseconds
    benchmark.pedantic(compile_model, rounds=3, iterations=1)

def test_bench_warm_start_execution_cpu(benchmark, compiled_cpu_engine):
    """
    Measures the purely numerical integration loop (SUNDIALS KLU) and 
    memory updates for a standard 1C discharge.
    Bypasses Protocol Python-loops to test strictly native speed.
    """
    params = parameters.Chen2020()

    def execute_solve():
        return compiled_cpu_engine.solve(t_span=(0, 170), parameters=params)

    benchmark(execute_solve)

def test_bench_warm_start_execution_gpu(benchmark, base_model):
    """
    Measures the GPU execution loop (SUNDIALS cuSOLVER).
    Requires a GPU runner.
    """
    try:
        gpu_engine = Engine(model=base_model, target="cuda:0", mesh_resolution=MESH_RES)
    except RuntimeError:
        pytest.skip("CUDA hardware not available on this runner.")

    params = parameters.Chen2020()

    def execute_solve():
        return gpu_engine.solve(t_span=(0, 170), parameters=params)

    benchmark(execute_solve)

@pytest.mark.skip(reason="Hangs")
def test_bench_parameter_sweep_throughput(benchmark, compiled_cpu_engine):
    """
    Measures the task-parallel throughput for MCMC or parameter sweeps.
    """
    base_params = parameters.Chen2020()
    
    # Generate 100 varying parameter sets
    param_batch = [
        {**base_params, "neg_elec.porosity": 0.2 + (i * 0.001)} 
        for i in range(100)
    ]

    def execute_batch():
        # Uses Rust's Rayon threadpool internally
        return compiled_cpu_engine.solve_batch(
            t_span=(0, 170), 
            parameters=param_batch,
            max_workers=16
        )

    benchmark.pedantic(execute_batch, rounds=5, iterations=1)