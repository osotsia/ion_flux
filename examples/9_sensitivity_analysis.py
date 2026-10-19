"""
Global and Local Sensitivity Analysis (DFN)
-------------------------------------------
Demonstrates input feature importance mapping using Ahead-of-Time (AOT) 
differentiability and Rayon task-parallelism.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sobol_indices, uniform
from tqdm import tqdm

# Add the 'models' directory to the path to inherit the baseline model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from Chen2020_DFN import Chen2020_DFN # type: ignore

import ion_flux as fx
from ion_flux.protocols import Sequence, CC

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG = {
    "c_rates": [0.5, 1.0, 2.0], # 3x3 Grid Multipliers
    "v_cutoff": 2.5,            # End-of-Discharge Voltage
    "sobol_n": 64,              # Power of 2 (Total evaluations = N * (D + 2))
    "variance": 0.20,           # +/- 20% uniform variance for global sweep
    "target": "cpu:serial"      # JIT Compilation Target
}

# ==============================================================================
# Computational Pipelines
# ==============================================================================

def compute_local_sensitivity(engine: fx.Engine, params_list: list, nominal_params: dict, c_rate: float):
    """
    Computes exact local sensitivities via analytical adjoints: 
    "How many extra seconds of discharge do I get if I increase a parameter by 1%?"
    """
    protocol = Sequence([
        CC(rate=5.0 * c_rate, until=engine.model.V_cell <= CONFIG["v_cutoff"], time=7200)
    ])
    
    # Forward Pass: Records the highly non-linear trajectory tape
    res = engine.solve(
        protocol=protocol, 
        parameters=nominal_params, 
        requires_grad=params_list, 
        show_progress=False
    )
    
    # Backward Pass: Adjoint VJP evaluating the exact trigger event
    loss = fx.metrics.runtime_to_event(res, trigger_state="V_cell")
    grads = loss.backward()
    
    # Normalize gradients to engineering units: (Seconds gained) per (1% parameter increase)
    sensitivities = {}
    for p in params_list:
        delta_p_1_percent = nominal_params[p] * 0.01
        sensitivities[p] = grads[p] * delta_p_1_percent
        
    return sensitivities, res


def compute_global_sobol(engine: fx.Engine, params_list: list, nominal_params: dict, c_rate: float):
    """
    Computes global feature variance using Total-Order Sobol Indices (ST).
    "What fraction of the total variability in battery runtime is driven by uncertainty in this parameter?"
    """
    # Define uniform variance distributions for the parameter hyperspace
    dists = [
        uniform(loc=nominal_params[p] * (1.0 - CONFIG["variance"]), 
                scale=nominal_params[p] * 2 * CONFIG["variance"])
        for p in params_list
    ]
        
    protocol = Sequence([CC(rate=5.0 * c_rate, until=engine.model.V_cell <= CONFIG["v_cutoff"], time=7200)])
    all_trajectories = []
    
    total_evals = CONFIG["sobol_n"] * (len(params_list) + 2)
    pbar = tqdm(total=total_evals, desc=f"Sobol Sweep ({c_rate}C)", leave=False)

    def batch_evaluator(x_array):
        """Closure invoked by scipy.sobol_indices to evaluate parameter matrices."""
        payloads = [{p: val for p, val in zip(params_list, row)} for row in x_array.T]
        kpis = []
        
        # Chunk execution to limit concurrent memory footprint
        chunk_size = min(64, (os.cpu_count() or 4) * 4) 
        
        for i in range(0, len(payloads), chunk_size):
            chunk = payloads[i:i+chunk_size]
            res_chunk = engine.solve_batch(
                parameters=chunk, 
                t_span=(0, 7200), 
                protocols=protocol, 
                max_workers=os.cpu_count(), 
                show_progress=False
            )
            
            for r in res_chunk:
                if len(all_trajectories) < 250:
                    all_trajectories.append({
                        "Time [s]": r["Time [s]"].data.copy(),
                        "V_cell": r["V_cell"].data.copy()
                    })
                kpis.append(r["Time [s]"].data[-1])
                
            pbar.update(len(chunk))
            
        return np.array(kpis)

    res = sobol_indices(func=batch_evaluator, n=CONFIG["sobol_n"], dists=dists)
    pbar.close()
    
    importance = {p: res.total_order[i] for i, p in enumerate(params_list)}
    return all_trajectories, importance

# ==============================================================================
# Visualization
# ==============================================================================

def plot_sensitivity_dashboard(results_data: dict, params_list: list):
    """Renders the 3x3 multi-variate diagnostic dashboard."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Multi-Variate Sensitivity Analysis (Chen2020 DFN)", fontsize=18, fontweight="bold")
    
    for row_idx, c_rate in enumerate(CONFIG["c_rates"]):
        data = results_data[c_rate]
        ax_env, ax_loc, ax_glob = axes[row_idx]
        
        # --- Column 1: Trajectory Envelope ---
        for res_dict in data["mc_results"]:
            t_mins = res_dict["Time [s]"] / 60.0
            ax_env.plot(t_mins, res_dict["V_cell"], color="tab:blue", alpha=0.05, linewidth=1.5)
            
        t_nom_mins = data["nominal_res"]["Time [s]"].data / 60.0
        ax_env.plot(t_nom_mins, data["nominal_res"]["V_cell"].data, color="black", linewidth=2, label="Nominal")
        ax_env.axhline(CONFIG["v_cutoff"], color="k", linestyle="--", linewidth=1.5, label="Cutoff")
        
        ax_env.set_ylabel(f"{c_rate}C\nTerminal Voltage [V]", fontweight="bold")
        ax_env.set_xlim(0, max(t_nom_mins))
        
        if row_idx == 0: 
            ax_env.set_title(f"Uncertainty Envelope (Sobol Samples)\n($\\pm${int(CONFIG['variance']*100)}% Variance)", fontsize=12)
            ax_env.legend(loc="upper right")
        if row_idx == 2: 
            ax_env.set_xlabel("Time [minutes]")
        
        # Establish Common Y-Axis Sorting (Driven by Global Importance)
        sorted_global = sorted(data["global_importance"].items(), key=lambda item: item[1])
        sorted_keys = [item[0] for item in sorted_global]
        vals_global = [item[1] for item in sorted_global]
        
        # --- Column 2: Local Sensitivity ---
        vals_local = [data["local_sens"][k] for k in sorted_keys]
        colors = ["tab:red" if v < 0 else "tab:green" for v in vals_local]
        
        ax_loc.barh(sorted_keys, vals_local, color=colors, height=0.6)
        ax_loc.axvline(0, color="k", linewidth=1)
        
        if row_idx == 0: 
            ax_loc.set_title("Exact Local Adjoint Sensitivities\n(Impact on Discharge Runtime)", fontsize=12)
        if row_idx == 2: 
            ax_loc.set_xlabel(f"Δ Time to {CONFIG['v_cutoff']}V [Seconds] per 1% Increase")
        
        for i, v in enumerate(vals_local):
            offset = max(abs(max(vals_local)), abs(min(vals_local)), 1e-6) * 0.05
            x_text = v + offset if v >= 0 else v - offset
            ha = 'left' if v >= 0 else 'right'
            ax_loc.text(x_text, i, f"{v:.2f}s", va='center', ha=ha, fontsize=9)
        
        # --- Column 3: Global Feature Importance ---
        ax_glob.barh(sorted_keys, vals_global, color="tab:purple", height=0.6)
        
        for i, v in enumerate(vals_global):
            ax_glob.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=10)
            
        ax_glob.set_xlim(0, max(vals_global) * 1.25)
        
        if row_idx == 0: 
            ax_glob.set_title(f"Global Feature Importance\n(Sobol $S_T$ vs Time-to-{CONFIG['v_cutoff']}V)", fontsize=12)
        if row_idx == 2: 
            ax_glob.set_xlabel("Total-Order Sobol Index ($S_T$)")

    plt.tight_layout()
    plt.show()

# ==============================================================================
# Main Orchestration
# ==============================================================================

def main():
    model = Chen2020_DFN()
    engine = fx.Engine(model=model, target=CONFIG["target"])
    
    target_params = ["D_s_n", "D_s_p", "k_n", "k_p"]
    nominal_params = {p: engine.parameters[p].value for p in target_params}
    
    results_data = {}
    
    for c_rate in CONFIG["c_rates"]:
        print(f"\n--- Processing {c_rate}C ---")
        local_sens, nominal_res = compute_local_sensitivity(engine, target_params, nominal_params, c_rate)
        mc_results, global_importance = compute_global_sobol(engine, target_params, nominal_params, c_rate)
        
        results_data[c_rate] = {
            "local_sens": local_sens,
            "nominal_res": nominal_res,
            "mc_results": mc_results,
            "global_importance": global_importance
        }
        
    plot_sensitivity_dashboard(results_data, target_params)

if __name__ == "__main__":
    main()