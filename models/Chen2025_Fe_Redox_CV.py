"""
================================================================================
Differentiable Electrochemistry: End-to-End Parameter Inversion
Reference: Chen et al., 2025 (DiffEC) - Figures 3a through 3e
================================================================================
This script implements the full inverse problem pipeline to recover heterogeneous 
electron transfer kinetics and mass transport properties from Cyclic Voltammetry.

This architecture utilizes continuous exact-gradient Adjoints (via Enzyme AD) to 
feed quasi-Newton optimizers (L-BFGS-B), which greatly reduces convergence time 
and memory requirements.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.lines import Line2D
import ion_flux as fx

# ==============================================================================
# Declarative Model Formulation
# ==============================================================================

class Fe_Redox_CV_Differentiable(fx.PDE):
    """
    1D Diffusion with Butler-Volmer Kinetics.
    The cyclic voltage sweep is formulated as a continuous algebraic constraint 
    driven by a time ODE, preserving a single differentiable computational graph.
    """
    
    # --- 1. Topology ---
    # The diffusion layer extends to ~60 um at 200 mV/s. 
    # Envelop the entire active gradient in the dense mesh to prevent FVM truncation error.
    x = fx.Domain(bounds=(0, 2e-3), name="x")
    x_dense = x.region(bounds=(0, 100e-6), resolution=150, name="x_dense")
    x_sparse = x.region(bounds=(100e-6, 2e-3), resolution=50, name="x_sparse")
    
    # --- 2. States ---
    c_ox  = fx.State(domain=x, name="c_ox")
    c_red = fx.State(domain=x, name="c_red")
    time_s = fx.State(domain=None, name="time_s") 
    
    # PROMOTED TO STATE: Forces metrics.py to track reverse-mode Adjoints.
    I_CV_uA = fx.State(domain=None, name="I_CV_uA") 
    
    # --- 3. Discoverable Parameters ---
    k_0   = fx.Parameter(default=6.54e-5, name="k_0")     
    alpha = fx.Parameter(default=0.248, name="alpha")
    beta  = fx.Parameter(default=0.612, name="beta")
    D_avg = fx.Parameter(default=5.33e-10, name="D_avg")  
    
    # --- 4. Control Variables ---
    scan_rate = fx.Parameter(default=0.2, name="scan_rate") 
    
    def math(self):
        F = 96485.3
        R = 8.314
        T = 298.15
        E_ref = 0.4336  
        Area = np.pi * (0.85e-3)**2
        
        # CV Protocol Bounds
        E_start = 0.8
        E_rev = 0.1
        
        # --- The Algebraic Waveform ---
        T_half = (E_start - E_rev) / self.scan_rate
        E_app = E_start - self.scan_rate * (T_half - fx.abs(self.time_s - T_half))
        
        # --- Mass Transport ---
        # fx.grad automatically maps across the non-uniform regional grid strides
        flux_ox  = -self.D_avg * fx.grad(self.c_ox, axis=self.x)
        flux_red = -self.D_avg * fx.grad(self.c_red, axis=self.x)
        
        # --- Interfacial Kinetics (Butler-Volmer) ---
        overpotential = E_app - E_ref
        f_RT = F / (R * T)
        
        J_BV = self.k_0 * (
            fx.exp(-self.alpha * f_RT * overpotential) * self.c_ox.left - 
            fx.exp( self.beta  * f_RT * overpotential) * self.c_red.left
        )
        
        # --- Nernstian Equilibrium Initial Condition ---
        # Dynamically calculate the exact concentration of C_red required to force J_BV = 0.0 at t=0.
        # This eliminates the non-equilibrium cold-start transient ("lip").
        initial_eta = E_start - E_ref
        c_red_eq = 4.85 * fx.exp(-(self.alpha + self.beta) * f_RT * initial_eta)
        
        return {
            "equations": {
                # 1 second per second. This is necessary to get E_app working above in that continuous formulation.
                self.time_s:  fx.dt(self.time_s) == 1.0, 

                self.c_ox:    fx.dt(self.c_ox)   == -fx.div(flux_ox, axis=self.x),
                self.c_red:   fx.dt(self.c_red)  == -fx.div(flux_red, axis=self.x),
                
                # Spatial DAE Constraint. Solved concurrently with the bulk PDEs.
                self.I_CV_uA: self.I_CV_uA == -1e6 * F * Area * J_BV
            },
            "boundaries": {
                flux_ox:  {"left": -J_BV, "right": 0.0},
                flux_red: {"left":  J_BV, "right": 0.0}
            },
            "initial_conditions": {
                self.time_s: 0.0,
                self.c_ox:   4.85, 
                self.c_red:  c_red_eq,
                self.I_CV_uA: 0.0
            }
        }


# ==============================================================================
# Inversion Pipeline
# ==============================================================================

def generate_ground_truth(engine: fx.Engine, scan_rates_mV: list, true_params: dict) -> list:
    """Generates synthetic empirical data using the target physiological parameters."""
    print("Generating synthetic laboratory data (Ground Truth)...")
    
    ground_truth = []
    
    for rate_mV in scan_rates_mV:
        rate_V_s = rate_mV / 1000.0
        t_half = (0.8 - 0.1) / rate_V_s
        t_end = 2.0 * t_half
        t_eval = np.linspace(0, t_end, int(t_end * 250))
        
        params = {**true_params, "scan_rate": rate_V_s}
        res = engine.solve(t_eval=t_eval, parameters=params, show_progress=False)
        ground_truth.append((rate_V_s, t_eval, res["I_CV_uA"].data))
        
    return ground_truth


def execute_parameter_inversion(engine: fx.Engine, ground_truth: list, init_params: dict):
    """
    Drives the L-BFGS-B optimizer using exact analytical Vector-Jacobian Products (VJPs).
    Hyperspace parameters are scaled to O(1) to ensure Hessian stability.
    """
    print("\nExecuting Exact-Gradient Parameter Inversion...")
    
    # 1. Normalization Scales
    SCALE_K = 1e-5
    SCALE_D = 1e-10
    
    # 2. Initial Guesses
    init_k_0 = init_params["k_0"]
    init_alpha = init_params["alpha"]
    init_beta = init_params["beta"]
    init_D = init_params["D_avg"]
    
    iteration = 0
    start_time = time.perf_counter()

    def objective_function(x):
        nonlocal iteration
        iteration += 1
        
        # Unpack from normalized hyperspace
        k_0_guess = x[0] * SCALE_K
        alpha_guess = x[1]
        beta_guess = x[2]
        D_guess = x[3] * SCALE_D
        
        total_loss = 0.0
        total_grads = np.zeros(4)
        
        for rate_V_s, t_eval, i_target in ground_truth:
            params = {
                "k_0": k_0_guess,
                "alpha": alpha_guess,
                "beta": beta_guess,
                "D_avg": D_guess,
                "scan_rate": rate_V_s
            }
            
            # Forward Pass
            res = engine.solve(
                t_eval=t_eval, 
                parameters=params, 
                requires_grad=["k_0", "alpha", "beta", "D_avg"], 
                show_progress=False
            )
            
            # Backward Pass (Adjoint Sensitivities)
            loss = fx.metrics.rmse(predicted=res["I_CV_uA"], target=i_target)
            grads = loss.backward()
            
            total_loss += loss.value
            
            # Chain Rule mapping physical gradients back to normalized hyperspace
            total_grads[0] += grads["k_0"] * SCALE_K
            total_grads[1] += grads["alpha"]
            total_grads[2] += grads["beta"]
            total_grads[3] += grads["D_avg"] * SCALE_D

        print(f"  [Iter {iteration:02d}] Loss: {total_loss:.4e} | Grad Norm: {np.linalg.norm(total_grads):.2e}")

        return total_loss, total_grads

    x0 = np.array([init_k_0 / SCALE_K, init_alpha, init_beta, init_D / SCALE_D])
    bounds = [(0.1, 100.0), (0.01, 1.0), (0.01, 1.0), (0.1, 50.0)]
    
    opt_result = minimize(
        objective_function, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'ftol': 1e-7}
    )
    
    elapsed = time.perf_counter() - start_time
    
    recovered_params = {
        "k_0": opt_result.x[0] * SCALE_K,
        "alpha": opt_result.x[1],
        "beta": opt_result.x[2],
        "D_avg": opt_result.x[3] * SCALE_D
    }
    
    print(f"\nConvergence Reached in {elapsed:.2f} seconds.")
    print(f"  Optimization Steps (nit): {opt_result.nit}")
    print(f"  Function Evaluations (nfev): {opt_result.nfev}")
    return recovered_params


# ==============================================================================
# Execution & Visualization (2x2 Grid)
# ==============================================================================

def visualize_results(plot_data: list, scan_rates_mV: list):
    """Generates a 2x2 diagnostic dashboard for the CV inversion."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0.8, 0.0, len(scan_rates_mV)))
    
    # --------------------------------------------------------------------------
    # 1. Data Mapping
    # --------------------------------------------------------------------------
    
    # Panel [0, 0]: Representative Macro View (Fastest scan rate only)
    rep = plot_data[0] 
    axs[0, 0].plot(rep["v_arr"], rep["i_target"], 'k-', lw=4, alpha=0.3, label="Ground Truth")
    axs[0, 0].plot(rep["v_arr"], rep["i_init"], 'r:', lw=2, label="Initial Guess")
    axs[0, 0].plot(rep["v_arr"], rep["i_opt"], 'g--', lw=2, label="Optimized Fit")
    axs[0, 0].invert_xaxis()
    axs[0, 0].legend(loc='upper right')

    # Panels [0, 1] & [1, 0]: All CVs and Residual Errors
    for data, color in zip(plot_data, colors):
        label = f"{data['rate_mV']} mV/s"
        
        # [0, 1] Target vs. Recovery
        axs[0, 1].plot(data["v_arr"], data["i_target"], color=color, ls='--', lw=2, alpha=0.7)
        axs[0, 1].plot(data["v_arr"], data["i_opt"], color=color, ls='-', lw=1.5, label=label)
        
        # [1, 0] Residuals
        axs[1, 0].plot(data["progress_pct"], data["err_opt"], color=color, ls='-', lw=2, label=label)

    axs[0, 1].invert_xaxis()
    axs[0, 1].legend(loc='upper left')
    
    axs[1, 0].axhline(0, color='k', lw=1.5, alpha=0.8)
    axs[1, 0].legend(loc='upper right')

    # Panel [1, 1]: Boundary Layer Depletion Zone at Peak Current (0.38V)
    res_obj = rep["res_opt_obj"]
    engine = res_obj.engine
    peak_idx = np.argmin(res_obj["I_CV_uA"].data)
    
    # 1. Extract the exact normalized node centers from the engine's memory cache
    # and scale them by the physical domain length (2e-3 meters = 2000 um)
    n_nodes = 200
    centers_offset = engine.layout.mesh_offsets["x"]["w_centers"]
    x_exact_um = np.array([
        engine.layout.mesh_cache[centers_offset + i] for i in range(n_nodes)
    ]) * 2000.0
    
    # 2. Extract the full concentration profiles
    c_ox = res_obj["c_ox"].data[peak_idx, :]
    c_red = res_obj["c_red"].data[peak_idx, :]
    
    # 3. Plot using the exact mesh coordinates
    axs[1, 1].plot(x_exact_um, c_ox, 'b-', lw=2, label=r"Reactant ($Fe^{3+}$)")
    axs[1, 1].plot(x_exact_um, c_red, 'r-', lw=2, label=r"Product ($Fe^{2+}$)")
    
    axs[1, 1].axvline(0, color='k', lw=2, label='Electrode Surface')
    axs[1, 1].legend(loc='center right')
    axs[1, 1].set_xlim(-2, 60)
    
    axs[1, 1].text(0.05, 0.08, 
                   "Tafel approximations fail here\nbecause the reactant is\nseverely depleted at the surface.", 
                   transform=axs[1, 1].transAxes, fontsize=10, style='italic', 
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

    # --------------------------------------------------------------------------
    # 2. Axis Formatting
    # --------------------------------------------------------------------------
    
    def format_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.3)

    format_ax(axs[0, 0], f"Macro View: Trajectory ({rep['rate_mV']} mV/s)", "Potential, V vs. SCE", r"Current, $\mu$A")
    format_ax(axs[0, 1], "All CVs: Target vs. Recovery", "Potential, V vs. SCE", r"Current, $\mu$A")
    format_ax(axs[1, 0], "Micro View: Residual Error Tolerance", "Scan Progress (%)", r"Error: $I_{sim} - I_{target}$ ($\mu$A)")
    
    peak_voltage = rep["v_arr"][peak_idx]
    format_ax(axs[1, 1], f"Boundary Depletion Zone @ Peak Current ({peak_voltage:.2f}V)", r"Distance from Electrode, $\mu$m", "Concentration, mM")

    plt.tight_layout()
    plt.show()


def main():
    print("Compiling AST via Ahead-of-Time (AOT) LLVM/Enzyme Pipeline...")
    model = Fe_Redox_CV_Differentiable()
    engine = fx.Engine(model, target="cpu:serial")
    
    scan_rates_mV = [200, 100, 50] 
    
    true_params = {
        "k_0": 6.54e-5,
        "alpha": 0.248,
        "beta": 0.612,
        "D_avg": 5.33e-10
    }
    
    init_params = {
        "k_0": 1.0e-5,
        "alpha": 0.5,
        "beta": 0.5,
        "D_avg": 1.0e-10
    }
    
    ground_truth = generate_ground_truth(engine, scan_rates_mV, true_params)
    recovered_params = execute_parameter_inversion(engine, ground_truth, init_params)
    
    # Pre-calculate plotting data
    plot_data = []
    for rate_V_s, t_eval, i_target in ground_truth:
        res_init = engine.solve(
            t_eval=t_eval, 
            parameters={**init_params, "scan_rate": rate_V_s}, 
            show_progress=False
        )
        res_opt = engine.solve(
            t_eval=t_eval, 
            parameters={**recovered_params, "scan_rate": rate_V_s}, 
            show_progress=False
        )
        
        t_half = 0.7 / rate_V_s
        v_arr = 0.8 - rate_V_s * (t_half - np.abs(t_eval - t_half))
        progress_pct = np.linspace(0, 100, len(t_eval))
        
        plot_data.append({
            "rate_mV": int(rate_V_s * 1000),
            "v_arr": v_arr,
            "progress_pct": progress_pct,
            "i_target": i_target,
            "i_init": res_init["I_CV_uA"].data,
            "i_opt": res_opt["I_CV_uA"].data,
            "err_opt": res_opt["I_CV_uA"].data - i_target,
            # We explicitly store the full PDE result for the 200 mV/s scan to extract spatial insights
            "res_opt_obj": res_opt 
        })

    visualize_results(plot_data, scan_rates_mV)

if __name__ == "__main__":
    main()