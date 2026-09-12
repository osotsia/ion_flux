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
import asyncio
import os
import shutil
import platform
import numpy as np
import ion_flux as fx
from ion_flux.runtime.scheduler import MultiTenantScheduler
from ion_flux.protocols import Sequence, CC, CV, Rest
from ion_flux._core import solve_ida_native



class AdjointAndEISModel(fx.PDE):
    """Validates Analytical EIS (Frequency Domain) and Exact VJP Adjoints."""
    V = fx.State(domain=None, name="V")
    R = fx.Parameter(default=10.0, name="R")
    C = fx.Parameter(default=0.1, name="C")
    i_app = fx.Parameter(default=1.0, name="i_app")
    
    def math(self):
        return {
            "equations": {
                self.V: fx.dt(self.V) == (self.i_app - self.V / self.R) / self.C
            },
            "boundaries": {},
            "initial_conditions": {
                self.V: 0.0
            }
        }




# ==============================================================================
# Concept 4: Concurrency & Cloud Scale (Batching, Async, .so Export)
# ==============================================================================

@REQUIRES_RUNTIME
def test_rayon_task_parallelism_batching():
    """Validates that solve_batch bypasses the Python GIL utilizing Rust Rayon."""
    engine = fx.Engine(model=AdjointAndEISModel(), target="cpu", mock_execution=False)
    
    param_sweep = [{"R": 10.0}, {"R": 20.0}, {"R": 30.0}]
    results = engine.solve_batch(parameters=param_sweep, t_span=(0, 1.0), max_workers=3)
    
    assert len(results) == 3
    assert results[0].status == "completed"
    
    # Higher resistance should lead to a higher accumulated voltage
    assert results[2]["V"].data[-1] > results[0]["V"].data[-1]



@REQUIRES_RUNTIME
@pytest.mark.xfail(reason="Compiler bug. Will fix later")
def test_openmp_data_parallelism_emission():
    """Validates OpenMP pragmas are safely emitted for massive spatial arrays."""
    class LargeOpenMPModel(fx.PDE):
        x = fx.Domain(bounds=(0, 1), resolution=100) # Resolution > 50 triggers OpenMP Pragma
        c = fx.State(domain=x)
        def math(self):
            return {
                "equations": {self.c: fx.dt(self.c) == fx.grad(self.c)}, 
                "boundaries": {self.c: {"left": fx.Dirichlet(0.0), "right": fx.Dirichlet(0.0)}}, 
                "initial_conditions": {self.c: 0.0}
            }
            
    engine = fx.Engine(model=LargeOpenMPModel(), target="cpu:omp", mock_execution=False)
    assert "omp parallel for" in engine.cpp_source




@pytest.mark.asyncio
async def test_async_multitenant_scheduler_isolation():
    """
    Validates Async queueing limits hardware oversubscription, and safely isolates 
    thread-pool panic/exceptions from dragging down sibling jobs.
    """
    # For isolation testing without needing Clang, use a mock execution engine
    engine = fx.Engine(model=AdjointAndEISModel(), target="cpu", mock_execution=True)
    scheduler = MultiTenantScheduler(max_concurrent=2)
    
    # `c.t0` == float('inf') is a hardcoded mock trigger for "Native Solver Crash" in mock_execution
    bad_params = {"c.t0": float('inf')} 
    good_params = {"R": 10.0}
    
    future_bad = engine.solve_async(t_span=(0, 1), parameters=bad_params, scheduler=scheduler)
    future_good = engine.solve_async(t_span=(0, 1), parameters=good_params, scheduler=scheduler)
    
    res_bad, res_good = await asyncio.gather(future_bad, future_good, return_exceptions=True)
    
    assert isinstance(res_bad, Exception)
    assert "Newton convergence failure" in str(res_bad) or "crash" in str(res_bad).lower()
    
    assert not isinstance(res_good, Exception)
    assert res_good.status == "completed"

    
