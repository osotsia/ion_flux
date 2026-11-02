import pytest
import numpy as np
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class TrajectoryParityPDE(fx.PDE):
    soc = fx.State(domain=None, name="soc")
    V = fx.State(domain=None, name="V")
    i_app = fx.State(domain=None, name="i_app")
    
    terminal = fx.Terminal(current=i_app, voltage=V)
    R = fx.Parameter(default=0.05, name="R")
    
    def math(self):
        return {
            "equations": {
                self.soc: fx.dt(self.soc) == -self.i_app / 3600.0,
                self.V: self.V == 4.0 + self.soc - self.i_app * self.R
            },
            "boundaries": {},
            "initial_conditions": {
                self.soc: 1.0, 
                self.V: 4.5, 
                self.i_app: 0.0
            }
        }

def test_solve_vs_solve_batch_trajectory_alignment():
    """
    Asserts exact index-by-index trajectory parity between single-engine solve()
    and Rayon solve_batch() across multi-stage Sequence protocols.
    
    Catches:
      1. Missing t=0.0 point in solve(protocol=...).
      2. Trajectory length mismatch (N vs N+1).
      3. Discrepancies in initial algebraic state at t=0.0 under closed-circuit load.
    """
    model = TrajectoryParityPDE()
    engine = fx.Engine(model=model, target="cpu", mock_execution=False)
    
    protocol = Sequence([
        CC(rate=10.0, time=3.0),
        Rest(time=2.0)
    ])
    
    res_single = engine.solve(protocol=protocol, show_progress=False)
    res_batch = engine.solve_batch(parameters=[{}], protocols=[protocol], show_progress=False)[0]
    
    t_single = res_single["Time [s]"].data
    t_batch = res_batch["Time [s]"].data
    
    # Invariant 1: Trajectories must start at t=0.0
    assert t_single[0] == pytest.approx(0.0, abs=1e-12), (
        f"Single solve() trajectory dropped initial timepoint: starts at {t_single[0]}"
    )
    assert t_batch[0] == pytest.approx(0.0, abs=1e-12), (
        f"solve_batch() trajectory dropped initial timepoint: starts at {t_batch[0]}"
    )
    
    # Invariant 2: Identical trajectory lengths across identical protocols
    assert len(t_single) == len(t_batch), (
        f"Trajectory length mismatch: solve() produced {len(t_single)} points, "
        f"solve_batch() produced {len(t_batch)} points."
    )
    
    # Invariant 3: Identical time grids and state values
    np.testing.assert_allclose(t_single, t_batch, atol=1e-12)
    np.testing.assert_allclose(res_single["V"].data, res_batch["V"].data, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_single["i_app"].data, res_batch["i_app"].data, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_single["soc"].data, res_batch["soc"].data, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])