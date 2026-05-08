"""
Exact-Gradient GITT Parameter Inversion using ion_flux
------------------------------------------------------
Inspired by recent work from the Battery Intelligence Lab:
- Jackowska et al. (2026): Fitting solid-state transport limitations in SC-NCM via GITT.
- Kuhn et al. (2025): The computational bottleneck of fitting voltage relaxations 
  using surrogate models and Bayesian inference.

Why this matters:
Extracting exact gradients (dLoss/dParam) through a stiff DFN solver is 
notoriously difficult, forcing parameterization libraries to rely on slow 
finite-differences or derivative-free optimizers (like SciPy's differential_evolution).

ion_flux solves this using Ahead-of-Time compilation and the Enzyme LLVM AD plugin. 
It calculates the exact, continuous Vector-Jacobian Product (VJP) of the entire 
voltage relaxation curve in a single backward pass. 
This allows hyper-fast fitting using first-order optimizers like L-BFGS-B.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

from Chen2020_DFN import Chen2020_DFN

def run_gitt_inversion_demo():
    print("Initializing ion_flux DFN Engine...")
    
    # 1. Compile the DFN to a native binary (bypassing Python overhead)
    engine = fx.Engine(model=Chen2020_DFN(), target="cpu:serial", solver_backend="native")
    
    # Define a standard GITT protocol: 10 min 1C discharge, followed by 1 hour rest.
    # We use t_eval to ensure identical array shapes for the RMSE loss function.
    t_pulse = 600
    t_rest = 3600
    protocol = Sequence([
        CC(rate=5.0, time=t_pulse),  # ~1C for the 5Ah M50 cell
        Rest(time=t_rest)
    ])

    # ==============================================================================
    # STEP 1: Generate Synthetic "Ground Truth" Experimental Data
    # ==============================================================================
    print("\n[Step 1] Generating Synthetic GITT Lab Data...")
    true_D_s_n = 3.3e-14  # Ground truth anode diffusion coefficient
    
    res_true = engine.solve(
        protocol=protocol, 
        parameters={"D_s_n": true_D_s_n}, 
        show_progress=False
    )
    v_target = res_true["V_cell"].data
    t_target = res_true["Time [s]"].data

    # ==============================================================================
    # STEP 2: The Exact-Gradient Optimization Loop
    # ==============================================================================
    print("\n[Step 2] Running L-BFGS-B Optimizer with Exact Analytical Adjoints...")
    
    # We start with a deliberately poor guess (almost an order of magnitude off)
    init_D_s_n = 0.5e-14  
    
    # Normalization scale to keep the optimizer mathematically stable
    SCALE_D = 1e-14
    iteration = 0
    start_time = time.perf_counter()

    def objective(x):
        nonlocal iteration
        iteration += 1
        
        # Un-normalize the parameter guess
        D_s_n_guess = x[0] * SCALE_D
        
        # --- THE CORE ion_flux AD CAPABILITY ---
        # 1. Forward Pass: Solves the DFN and records the non-linear integration trajectory
        res = engine.solve(
            protocol=protocol, 
            parameters={"D_s_n": D_s_n_guess}, 
            requires_grad=["D_s_n"], # Flags Enzyme to compute sensitivities for this param
            show_progress=False
        )
        
        # 2. Compute Loss: Mean Squared Error against our "Lab Data"
        loss = fx.metrics.rmse(predicted=res["V_cell"], target=v_target)
        
        # 3. Backward Pass: Triggers exact reverse-mode AD through the C++ / Rust solver
        grads = loss.backward()
        
        # Extract gradient and apply chain rule for the scale factor
        grad_x0 = grads["D_s_n"] * SCALE_D
        
        print(f"   [Iter {iteration:02d}] Loss: {loss.value:.4e} | D_s_n Guess: {D_s_n_guess:.3e} | dLoss/dD: {grads['D_s_n']:.2e}")
        
        return loss.value, np.array([grad_x0])

    # Run the SciPy optimizer using the exact Jacobian
    x0 = np.array([init_D_s_n / SCALE_D])
    res_opt = minimize(
        objective, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        bounds=[(0.1, 10.0)], # Bounded between 0.1e-14 and 10e-14
        options={
            'ftol': 1e-6,  # Function tolerance aligned with BDF rel_tol
            'gtol': 1e-4   # Gradient tolerance relaxed to prevent noise-floor thrashing
        }
    )
    
    elapsed = time.perf_counter() - start_time
    final_D_s_n = res_opt.x[0] * SCALE_D

    # ==============================================================================
    # STEP 3: Results & Visualization
    # ==============================================================================
    print(f"\n✅ Optimization converged in {elapsed:.2f} seconds ({res_opt.nit} iterations).")
    print(f"   Initial Guess : {init_D_s_n:.3e}")
    print(f"   Recovered     : {final_D_s_n:.3e}")
    print(f"   Ground Truth  : {true_D_s_n:.3e}")

    res_init = engine.solve(protocol=protocol, parameters={"D_s_n": init_D_s_n}, show_progress=False)
    res_final = engine.solve(protocol=protocol, parameters={"D_s_n": final_D_s_n}, show_progress=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle("GITT Parameter Inversion via Analytical Adjoints (Chen2020 DFN)", fontsize=14, fontweight="bold")
    
    t_mins = t_target / 60.0
    
    # --- Top Plot: Absolute Voltage ---
    ax1.plot(t_mins, v_target, 'k-', linewidth=3, label="Ground Truth (Lab Data)")
    ax1.plot(res_init["Time [s]"].data / 60.0, res_init["V_cell"].data, 'r--', linewidth=2, label=f"Initial Guess ($D_{{s,n}}$ = {init_D_s_n:.1e})")
    ax1.plot(res_final["Time [s]"].data / 60.0, res_final["V_cell"].data, 'g-.', linewidth=2, label=f"Optimized Fit ($D_{{s,n}}$ = {final_D_s_n:.1e})")
    
    ax1.axvline(t_pulse / 60.0, color='gray', linestyle=':', label="Current Interrupt (Start Rest)")
    ax1.set_ylabel("Terminal Voltage [V]", fontsize=12)
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # --- Bottom Plot: Residuals (Error in mV) ---
    err_init = (res_init["V_cell"].data - v_target) * 1000.0
    err_final = (res_final["V_cell"].data - v_target) * 1000.0
    
    ax2.plot(t_mins, err_init, 'r--', linewidth=2, label="Initial Guess Error")
    ax2.plot(t_mins, err_final, 'g-.', linewidth=2, label="Optimized Fit Error")
    ax2.axhline(0, color='k', linewidth=1)
    ax2.axvline(t_pulse / 60.0, color='gray', linestyle=':')
    
    ax2.set_xlabel("Time [minutes]", fontsize=12)
    ax2.set_ylabel("Error [mV]", fontsize=12)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_gitt_inversion_demo()