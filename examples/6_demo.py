"""
ion_flux vs. Graph-Based Architectures (e.g., PyBaMM/CasADi)

This script highlights three use-cases that severely degrade in legacy 
Symbolic-to-Numeric architectures due to high memory footprints, GIL locking, 
and lack of continuous time-domain automatic differentiation.

By leveraging Ahead-of-Time (AOT) LLVM compilation, Rust-based memory isolation, 
and continuous Adjoint sensitivity tracking, ion_flux handles these with zero 
boilerplate, zero cold-starts, and mathematical exactness.
"""

import time
import os
import numpy as np
from scipy.optimize import minimize
import ion_flux as fx

# ==============================================================================
# Base Model Definition
# ==============================================================================
class FastSPM(fx.PDE):
    """A compact Single Particle Model mapping physical parameters to cycler voltage."""
    r = fx.Domain(bounds=(0, 5e-6), resolution=15, coord_sys="spherical")
    
    c_s = fx.State(domain=r)
    V_cell = fx.State(domain=None)
    i_app = fx.State(domain=None)
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    # Target Parameters for sweep and optimization
    D_s = fx.Parameter(default=1e-14)
    R_internal = fx.Parameter(default=0.02)
    
    def math(self):
        flux = -self.D_s * fx.grad(self.c_s, axis=self.r)
        
        # ---------------------------------------------------------------------
        # Explicit Equation Targeting
        # ---------------------------------------------------------------------
        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(flux, axis=self.r),
                self.V_cell: self.V_cell == 4.2 - 0.0001 * self.c_s.right - self.R_internal * self.i_app
            },
            "boundaries": {
                flux: {"left": 0.0, "right": self.i_app / 96485.0}
            },
            "initial_conditions": {
                self.c_s: 500.0,
                self.V_cell: 4.15,
                self.i_app: 0.0
            }
        }

# ==============================================================================
# SCENARIO 1: Stateless Fleet-Scale SaaS Batching & Global Sensitivity
# ==============================================================================
def scenario_1_saas_fleet_batching():
    print("\n--- SCENARIO 1: Stateless Fleet-Scale SaaS Batching & Global Sensitivity ---")
    
    # 1. Compile exactly once (simulating a CI/CD build step)
    compiler_engine = fx.Engine(model=FastSPM(), target="cpu:serial")
    compiler_engine.export_binary("spm_prod.so")
    
    # 2. Instantaneous Serverless Load (0ms Cold Start)
    # PyBaMM Pain Point: Unpickling massive ASTs and evaluating CasADi graphs. 
    # ion_flux simply links a `.so` via FFI instantly.
    stateless_engine = fx.Engine.load("spm_prod.so", target="cpu:serial")
    
    # 3. Generate massive payload of concurrent requests
    # PyBaMM Pain Point: Holding 10,000 `Simulation` objects in RAM causes OOM. 
    # ion_flux maps parameters to flat C-arrays; Python memory overhead is zero.
    fleet_size = 10_000
    
    D_s_samples = np.random.uniform(1e-14, 5e-14, fleet_size)
    R_samples = np.random.uniform(0.01, 0.05, fleet_size)
    
    param_payloads = [
        {
            "D_s": D_s_samples[i], 
            "R_internal": R_samples[i],
            "_term_mode": 1.0,      # Constant Current mode
            "_term_i_target": 10.0  # 10A discharge
        } 
        for i in range(fleet_size)
    ]
    
    print(f"Executing {fleet_size} parallel battery models...")
    start_time = time.perf_counter()
    
    # 4. Rayon Task-Parallel Execution
    # PyBaMM Pain Point: Python Multiprocessing serialization bottlenecks.
    # ion_flux drops into Rust, entirely bypassing the Python GIL.
    results = stateless_engine.solve_batch(
        parameters=param_payloads, 
        t_span=(0, 3600), 
        max_workers=os.cpu_count()
    )
    
    elapsed = time.perf_counter() - start_time
    print(f"✅ Solved {fleet_size} models in {elapsed:.2f} seconds ({(fleet_size/elapsed):.1f} solves/sec).")

    # 5. Global Sensitivity Analysis (Variance-Based)
    print("\nExtracting Global Sensitivity Metrics (Variance Explained)...")
    # Target: The final cell voltage at t=3600s
    qoi_voltage = np.array([res["V_cell"].data[-1] for res in results])
    
    # Utilizing Pearson correlation squared (R^2) as a proxy for first-order global 
    # variance contribution.
    from scipy.stats import pearsonr
    
    r_ds, _ = pearsonr(D_s_samples, qoi_voltage)
    r_res, _ = pearsonr(R_samples, qoi_voltage)
    
    var_ds = (r_ds ** 2) * 100
    var_res = (r_res ** 2) * 100
    
    print(f"   D_s        : {var_ds:.1f}%")
    print(f"   R_internal : {var_res:.1f}%")
    print("   (Note: Residual variance indicates higher-order coupling or non-linearities not captured by R^2).")


# ==============================================================================
# SCENARIO 2: Real-Time BMS Hardware-in-the-Loop (Micro-stepping)
# ==============================================================================
def scenario_2_real_time_bms_control():
    print("\n--- SCENARIO 2: Microsecond-Latency BMS Co-Simulation ---")
    
    engine = fx.Engine(model=FastSPM(), target="cpu:serial")
    
    # 1. Start a "Hot" Native Session
    # PyBaMM Pain Point: `sim.step()` wraps massive amounts of Python overhead.
    # ion_flux pins the native Jacobian, sparse memory, and BDF integration history 
    # strictly in C++ RAM. Only scalar FFI pointers cross back to Python.
    session = engine.start_session(parameters={"D_s": 1e-14})
    
    print("Initiating 10ms BMS Control Loop for 60 simulated seconds...")
    dt = 0.01  # 10 millisecond loop
    
    start_time = time.perf_counter()
    steps_taken = 0
    
    # Mock BMS Logic: Charge at 5A, but aggressively taper current to prevent 
    # voltage from exceeding a hard 4.2V safety ceiling.
    target_v = 4.2
    Kp = 50.0  # Proportional gain
    
    while session.time < 60.0:
        # Introspect state with zero overhead (directly indexing the C-array)
        current_v = session.get("V_cell")
        
        # Calculate dynamic BMS current response
        if current_v < 4.15:
            i_req = -5.0 # Charge
        else:
            i_req = -max(0.0, Kp * (target_v - current_v))
            
        # 2. Inject dynamic parameters and advance native solver seamlessly
        # PyBaMM Pain Point: Dynamically changing inputs forces expensive re-discretization.
        session.step(dt=dt, inputs={"_term_i_target": i_req, "_term_mode": 1.0})
        steps_taken += 1
        
    elapsed = time.perf_counter() - start_time
    print(f"✅ Completed {steps_taken} control loops in {elapsed:.4f} seconds.")
    print(f"   Average latency per step: {(elapsed/steps_taken)*1e6:.1f} microseconds.")
    print(f"   Final Voltage: {session.get('V_cell'):.4f} V, Final Current: {session.get('i_app'):.2f} A")


# ==============================================================================
# SCENARIO 3: End-to-End Adjoint Gradient Optimization
# ==============================================================================
def scenario_3_adjoint_gradient_optimization():
    print("\n--- SCENARIO 3: End-to-End Adjoint Gradient Optimization ---")
    
    engine = fx.Engine(model=FastSPM(), target="cpu:serial")
    
    # 1. Generate Synthetic "Lab Data" (The Ground Truth)
    true_D_s = 3.0e-14
    true_R = 0.045
    p_true = {"D_s": true_D_s, "R_internal": true_R, "_term_mode": 1.0, "_term_i_target": 10.0}
    
    t_eval = np.linspace(0, 150, 100) # 150-second 10A Constant Current discharge pulse
    
    print("1. Generating Synthetic 'Lab Data' (Ground Truth)...")
    res_true = engine.solve(t_eval=t_eval, parameters=p_true, show_progress=False)
    v_target = res_true["V_cell"].data
    
    # 2. Optimization Routine using L-BFGS-B (First-Order Exact Gradients)
    init_D_s = 1.0e-14
    init_R = 0.02
    
    print("2. Handing exact analytical gradients to L-BFGS-B Optimizer...")
    # print(f"   Ground Truth  -> D_s: {true_D_s:.2e}, R: {true_R:.4f}")
    # print(f"   Initial Guess -> D_s: {init_D_s:.2e}, R: {init_R:.4f}\n")
    
    # Scale normalization prevents L-BFGS-B from breaking due to scale disparities
    SCALE_D = 1e-14
    SCALE_R = 0.1
    iteration = 0
    start_time = time.perf_counter()

    def objective(x):
        nonlocal iteration
        iteration += 1
        
        D_s_guess = x[0] * SCALE_D
        R_guess = x[1] * SCALE_R
        
        p_guess = {
            "D_s": D_s_guess, 
            "R_internal": R_guess, 
            "_term_mode": 1.0, 
            "_term_i_target": 10.0
        }
        
        # --- THE MAGIC HAPPENS HERE ---
        # Forward Pass (Records the highly non-linear integration trajectory)
        res = engine.solve(t_eval=t_eval, parameters=p_guess, requires_grad=["D_s", "R_internal"], show_progress=False)
        
        # Backward Pass (Triggers the VJP Adjoint loop through the native Rust solver)
        loss = fx.metrics.rmse(predicted=res["V_cell"], target=v_target)
        grads = loss.backward()
        
        # Chain Rule: Map physical gradients back to normalized optimizer scales
        grad_x0 = grads["D_s"] * SCALE_D
        grad_x1 = grads["R_internal"] * SCALE_R
        
        # Keep output concise
        if iteration % 10 == 0 or iteration == 1:
            print(f"   [Iter {iteration:02d}] Loss: {loss.value:.3e} | Guess -> D_s: {D_s_guess:.2e}, R: {R_guess:.4f}")
        
        return loss.value, np.array([grad_x0, grad_x1])

    x0 = np.array([init_D_s / SCALE_D, init_R / SCALE_R]) 
    
    # PyBaMM Pain Point: Extracting dLoss/dParameter requires massive numerical FD loops 
    # or derivative-free optimizers. ion_flux provides exact jacobians in O(1) backward pass.
    res_opt = minimize(
        objective, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        bounds=[(0.1, 10.0), (0.01, 1.0)] 
    )
    
    elapsed = time.perf_counter() - start_time
    final_D = res_opt.x[0] * SCALE_D
    final_R = res_opt.x[1] * SCALE_R
    
    print(f"\n✅ Optimization Complete in {elapsed:.2f} seconds ({res_opt.nit} iterations).")
    print("   Results:")
    print(f"   Initial Guess : D_s = {init_D_s:.2e}, R = {init_R:.4f}")
    print(f"   Recovered     : D_s = {final_D:.2e}, R = {final_R:.4f}")
    print(f"   Ground Truth  : D_s = {true_D_s:.2e}, R = {true_R:.4f}")
    
    # Clean up the artifact from Scenario 1
    if os.path.exists("spm_prod.so"): os.remove("spm_prod.so")
    if os.path.exists("spm_prod.so.meta.json"): os.remove("spm_prod.so.meta.json")

if __name__ == "__main__":
    scenario_1_saas_fleet_batching()
    scenario_2_real_time_bms_control()
    scenario_3_adjoint_gradient_optimization()