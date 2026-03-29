"""
Runtime Execution: Hardware Concurrency

Validates safe asynchronous task queueing and limits concurrent solves to
prevent hardware oversubscription (OOM) and scheduler collision conditions.
"""

import pytest
import asyncio
from ion_flux import Engine
from ion_flux.runtime import MultiTenantScheduler

@pytest.mark.asyncio
async def test_scheduler_handles_concurrent_solves(heat_model):
    engine = Engine(model=heat_model, target="cpu")
    scheduler = MultiTenantScheduler(max_concurrent=2)
    
    async def run_task(k_val):
        return await engine.solve_async(
            t_span=(0, 1), 
            parameters={"k": k_val}, 
            scheduler=scheduler
        )
    
    tasks = [run_task(k) for k in [0.1, 0.2, 0.3, 0.4, 0.5]]
    results = await scheduler.gather(tasks)
    
    assert len(results) == 5
    for res in results:
        assert res.status == "completed"

@pytest.mark.asyncio
async def test_scheduler_isolates_solver_failures(dae_model):
    engine = Engine(model=dae_model, target="cpu")
    scheduler = MultiTenantScheduler(max_concurrent=2)
    
    bad_params = {"p_fail": 0.0} 
    good_params = {"p_fail": 1.0}
    
    future_bad = engine.solve_async(t_span=(0, 1), parameters=bad_params, scheduler=scheduler)
    future_good = engine.solve_async(t_span=(0, 1), parameters=good_params, scheduler=scheduler)
    
    res_bad, res_good = await asyncio.gather(future_bad, future_good, return_exceptions=True)
    
    # Bad params trigger solver singularity via divide-by-zero, throwing an exception through the threadpool
    assert isinstance(res_bad, Exception)
    assert "Singular Jacobian" in str(res_bad) or "Newton" in str(res_bad) or "NaN" in str(res_bad) or "Step collapsed" in str(res_bad)
    
    # Concurrency barrier successfully prevented the native panic from dragging down the healthy task
    assert not isinstance(res_good, Exception)
    assert res_good.status == "completed"