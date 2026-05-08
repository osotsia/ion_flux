import pytest
import numpy as np
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class MinimalTestPDE(fx.PDE):
    """
    A minimal 1D Diffusion DAE model.
    Compiles extremely fast but strictly enforces spatial arrays, 
    algebraic constraints (DAEs), and parameter sensitivities to 
    stress-test the FFI boundary and Adjoint tape.
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=5, name="x")
    u = fx.State(domain=x, name="u")
    v = fx.State(domain=None, name="v")      # 0D Algebraic state
    i_app = fx.State(domain=None, name="i_app")
    
    # Must be strictly named 'terminal' for the AST compiler to inject the cycler equations
    terminal = fx.Terminal(current=i_app, voltage=v)

    k = fx.Parameter(default=2.0, name="k")  # Target for gradient testing

    def math(self):
        flux = -self.k * fx.grad(self.u, axis=self.x)
        return {
            "equations": {
                # 1D PDE
                self.u: fx.dt(self.u) == -fx.div(flux, axis=self.x),
                # 0D Spatial DAE
                self.v: self.v == self.u.right - self.i_app * 0.1
            },
            "boundaries": {
                flux: {"left": 0.0, "right": self.i_app}
            },
            "initial_conditions": {
                self.u: 0.0,
                self.v: 0.0,
                self.i_app: 0.0
            }
        }

@pytest.fixture(scope="module")
def engine():
    """
    Module-level fixture to compile the C++ binary exactly once 
    per test session, keeping the test suite lightning fast.
    """
    model = MinimalTestPDE()
    return fx.Engine(model=model, target="cpu:serial", solver_backend="native")

def test_continuous_adjoint_accuracy(engine):
    """
    Guards against: Broken Enzyme VJP bindings or corrupted DL/DY mappings.
    Method: Compares the exact O(1) analytical gradient against a central 
    Finite Difference approximation.
    """
    t_eval = np.linspace(0, 5.0, 50)
    
    # Override default terminal parameters to drive a constant current of 1.0
    p_base = {"_term_mode": 1.0, "_term_i_target": 1.0}

    # 1. Generate target "Lab Data"
    res_target = engine.solve(t_eval=t_eval, parameters={**p_base, "k": 3.0})
    v_target = res_target["v"].data

    # 2. Compute Exact Analytical Gradient via Adjoint (Reverse-Mode AD)
    guess_k = 2.0
    res_ad = engine.solve(t_eval=t_eval, parameters={**p_base, "k": guess_k}, requires_grad=["k"])
    loss_ad = fx.metrics.rmse(predicted=res_ad["v"], target=v_target)
    
    grads = loss_ad.backward()
    grad_ad = grads["k"]

    # 3. Compute Numerical Gradient via Central Finite Difference
    eps = 1e-5
    res_plus = engine.solve(t_eval=t_eval, parameters={**p_base, "k": guess_k + eps})
    loss_plus = fx.metrics.rmse(predicted=res_plus["v"], target=v_target).value
    
    res_minus = engine.solve(t_eval=t_eval, parameters={**p_base, "k": guess_k - eps})
    loss_minus = fx.metrics.rmse(predicted=res_minus["v"], target=v_target).value

    grad_fd = (loss_plus - loss_minus) / (2 * eps)

    # 4. Assert mathematical parity
    # Note: BDF is an adaptive solver. The FD runs (`+eps` and `-eps`) will take slightly 
    # different time-steps/orders, injecting numerical grid noise into the FD derivative. 
    # We use a 2% rtol to comfortably account for the adaptation noise floor.
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=2e-2, atol=1e-5, 
                               err_msg="Analytical Adjoint gradient drifted from Finite Difference baseline.")

def test_protocol_bisection_tape_integrity(engine):
    """
    Guards against: Memory leaks, shape mismatches (`np.vstack`), or 
    corrupted AD tapes caused by the speculative bisection root-finder.
    """
    model = engine.model
    protocol = Sequence([
        CC(rate=1.0, until=model.v >= 0.5, time=10.0),
        Rest(time=2.0)
    ])
    
    # 1. Baseline Target
    res_target = engine.solve(protocol=protocol, parameters={"k": 2.5}, show_progress=False)
    v_target = res_target["v"].data
    
    # 2. Optimization Pass (Stresses the suspend_history context manager)
    try:
        res_ad = engine.solve(protocol=protocol, parameters={"k": 2.0}, requires_grad=["k"], show_progress=False)
        loss_ad = fx.metrics.rmse(predicted=res_ad["v"], target=v_target)
        
        # If the tape arrays (micro_y, micro_t) are mismatched or contain garbage 
        # from the bisection rejections, this backward pass will throw a hard FFI exception.
        grads = loss_ad.backward()
        
    except Exception as e:
        pytest.fail(f"Adjoint backpropagation crashed across a Protocol Sequence boundary: {e}")

    assert "k" in grads, "Gradient dictionary missing requested parameter."
    assert np.isfinite(grads["k"]), "Gradient evaluated to NaN/Inf due to tape corruption."
    assert abs(grads["k"]) > 1e-10, "Gradient is zero. The BDF tape was likely dropped or disconnected."

def test_ffi_batch_memory_safety(engine):
    """
    Guards against: Segmentation faults or buffer overwrites when passing 
    zero-copy PyReadonlyArray pointers concurrently via Rayon.
    """
    # Override default terminal parameters to drive a constant current of 1.0
    parameters = [{"k": float(p), "_term_mode": 1.0, "_term_i_target": 1.0} for p in np.linspace(1.5, 3.5, 10)]
    
    try:
        results = engine.solve_batch(
            parameters=parameters, 
            t_span=(0, 5), 
            max_workers=4
        )
    except Exception as e:
        pytest.fail(f"Thread-parallel batch execution crashed: {e}")
        
    assert len(results) == 10
    
    # Verify deterministic output order matches input parameter distribution
    final_voltages = [res["v"].data[-1] for res in results]
    
    # Because a higher 'k' diffuses the flux faster, the gradient is less steep, 
    # causing the depletion at the right boundary to be less severe.
    # Therefore, a higher 'k' (index -1) results in a mathematically higher (less negative) voltage.
    assert final_voltages[0] < final_voltages[-1], "Batch execution yielded physically inconsistent data sorting."


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])