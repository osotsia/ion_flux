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
    
    bad_params = {"c.t0": float('inf')} 
    good_params = {"c.t0": 1.0}
    
    future_bad = engine.solve_async(t_span=(0, 1), parameters=bad_params, scheduler=scheduler)
    future_good = engine.solve_async(t_span=(0, 1), parameters=good_params, scheduler=scheduler)
    
    res_bad, res_good = await asyncio.gather(future_bad, future_good, return_exceptions=True)
    
    assert isinstance(res_bad, Exception)
    assert "Newton convergence failure" in str(res_bad)
    
    assert not isinstance(res_good, Exception)
    assert res_good.status == "completed"
