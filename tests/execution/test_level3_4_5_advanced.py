import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE
from ion_flux.protocols import Sequence, CC, CV, Rest
from ion_flux.dsl.core import Condition

# --- Test Models ---

class SimpleBatteryProtocol(fx.PDE):
    soc = fx.State()
    V = fx.State(domain=None) 
    i_app = fx.State(domain=None)
    
    terminal = fx.Terminal(current=i_app, voltage=V)
    R = fx.Parameter(default=0.05)
    
    def math(self):
        return {
            fx.dt(self.soc): -self.i_app / 3600.0,
            self.soc.t0: 1.0,
            
            # Physics mapping only. The Terminal handles the sequence constraints.
            self.V: 4.0 + self.soc - self.i_app * self.R,
            self.V.t0: 4.5,
            self.i_app.t0: 10.0
        }

class ParallelRCCircuit(fx.PDE):
    V = fx.State(name="V")
    R = fx.Parameter(default=10.0)
    C = fx.Parameter(default=0.1)
    i_app = fx.Parameter(default=1.0)
    def math(self):
        return {
            fx.dt(self.V): (self.i_app - self.V / self.R) / self.C,
            self.V.t0: 0.0
        }

class AdjointDecay(fx.PDE):
    y = fx.State(name="y")
    k = fx.Parameter(default=2.0)
    def math(self):
        return {
            fx.dt(self.y): -self.k * self.y,
            self.y.t0: 1.0
        }


# --- Tests ---

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_stateful_session_hil_control():
    """Validates the SolverHandle safely caching persistent memory structures without OS file read operations."""
    engine = Engine(model=SimpleBatteryProtocol(), target="cpu", mock_execution=False)
    if engine.mock_execution: pytest.skip("Compilation environment absent.")
    
    # Init at Rest (0 current) using hardware terminal defaults
    session = engine.start_session(parameters={"_term_i_target": 0.0, "_term_mode": 1.0})
    assert session.get("V") == pytest.approx(5.0)
    
    session.step(dt=1800.0, inputs={"_term_i_target": 1.0}) 
    assert session.time == 1800.0
    assert session.get("soc") == pytest.approx(0.5)
    assert session.get("V") == pytest.approx(4.45) # 4.0 + 0.5 - 1.0 * 0.05

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_protocol_hot_swapping_cccv():
    """Validates Python bisection root-finding triggers mapping the native DAE to CC/CV constraint arrays iteratively."""
    engine = Engine(model=SimpleBatteryProtocol(), target="cpu", mock_execution=False)
    if engine.mock_execution: pytest.skip("Compilation environment absent.")
    
    protocol = Sequence([
        CC(rate=10.0, until="V <= 3.2"),
        CV(voltage=3.2, time=5.0)
    ])
    
    res = engine.solve(protocol=protocol)
    assert res.status == "completed"
    
    V_traj = res["V"].data
    assert len(V_traj) > 0
    assert V_traj[-1] == pytest.approx(3.2, rel=1e-3)

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_native_eis_frequency_domain_solver():
    """Validates the JIT pulling analytical steady state Jacobian to evaluate (jwM + J)^-1 B."""
    engine = Engine(model=ParallelRCCircuit(), target="cpu", mock_execution=False)
    if engine.mock_execution: pytest.skip("Compilation environment absent.")
    
    session = engine.start_session(parameters={"R": 10.0, "C": 0.1, "i_app": 1.0})
    
    session.reach_steady_state()
    assert session.get("V") == pytest.approx(10.0)
    
    w_arr = np.array([0.01, 1.0, 100.0]) * 2 * np.pi
    Z_res = session.solve_eis(np.array([0.01, 1.0, 100.0]), input_var="i_app", output_var="V")
    
    # Reconstruct the complex array from the Differentiable SimulationResult output
    Z_sim = Z_res["Z_real"].data + 1j * Z_res["Z_imag"].data
    
    # Analytical Z(w) = R / (1 + j w R C)
    Z_analytical = 10.0 / (1 + 1j * w_arr * 1.0)
    
    np.testing.assert_allclose(np.real(Z_sim), np.real(Z_analytical), rtol=1e-4)
    np.testing.assert_allclose(np.imag(Z_sim), np.imag(Z_analytical), rtol=1e-4)

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_discrete_adjoint_backward_propagation():
    """
    Validates reverse tracking of continuous parameter sensitivities via Enzyme VJPs 
    and an unconditionally stable implicit backward integration scheme.
    """
    engine = Engine(model=AdjointDecay(), target="cpu", mock_execution=False)
    if engine.mock_execution: pytest.skip("Compilation environment absent.")
    
    res = engine.solve(t_span=(0, 1.0), parameters={"k": 2.0}, requires_grad=["k"])
       
    loss = fx.metrics.rmse(res["y"], np.zeros_like(res["y"].data), engine=engine, state_name="y")
    grads = loss.backward()
    
    grad_k = grads.get("k")
    assert isinstance(grad_k, float)
    assert not np.isnan(grad_k)
    assert grad_k < 0.0 # High k accelerates curve towards 0 target, lowering overall error

@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires compiled Rust backend.")
def test_rayon_task_parallelism_batch():
    """Level 5: Validates bypassing the Python GIL using Rust Rayon mapping execution threads natively."""
    engine = Engine(model=AdjointDecay(), target="cpu", mock_execution=False)
    if engine.mock_execution: pytest.skip("Compilation environment absent.")
    
    param_sweep = [{"k": 1.0}, {"k": 2.0}, {"k": 3.0}]
    results = engine.solve_batch(parameters=param_sweep, t_span=(0, 1.0))
    
    assert len(results) == 3
    # Decay constant 3.0 should result in lower ending y than constant 1.0
    assert results[2]["y"].data[-1] < results[0]["y"].data[-1]

def test_omp_data_parallelism_emission():
    """Level 5: Validates OpenMP #pragma mapping injected into AST codegen for spatial arrays."""
    class LargeSpatial(fx.PDE):
        x = fx.Domain(bounds=(0, 1), resolution=5000)
        c = fx.State(domain=x, name="c")
        def math(self): return { fx.dt(self.c): fx.grad(self.c) }
        
    engine = Engine(model=LargeSpatial(), target="cpu:omp", mock_execution=True)
    assert "#pragma omp parallel for" in engine.cpp_source