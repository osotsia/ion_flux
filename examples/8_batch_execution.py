"""
Batch Execution & Unique Protocol Mapping Demo
----------------------------------------------
This script demonstrates how to utilize `engine.solve_batch()` to distribute 
computationally heavy DFN models across multiple vCPUs, completely bypassing 
the Python Global Interpreter Lock (GIL).

It highlights the ability to map a unique `Sequence` protocol to each 
individual parameter payload in the batch 1:1.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the 'models' directory to the path so we can import Chen2020_DFN
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest
from Chen2020_DFN import Chen2020_DFN

def run_batch_demo():
    print("Compiling Chen2020_DFN to Native C++ Binary...")
    # Bypassing Python overhead; translating AST -> C++ -> LLVM -> Rust FFI
    engine = fx.Engine(model=Chen2020_DFN(), target="cpu:serial")

    # Define a test matrix of varying C-rates
    # 1C for the LG M50 cell is roughly 5.0A
    c_rates = [0.5, 1.0, 1.5, 2.0]
    base_D_s_n = 3.3e-14
    
    param_payloads = []
    protocol_payloads = []

    for c in c_rates:
        # 1. Perturb the parameters per battery
        # (e.g., simulating degraded diffusion coefficients at higher rates)
        degraded_D = base_D_s_n * (1.0 - 0.1 * c)
        param_payloads.append({"D_s_n": degraded_D})
        
        # 2. Assign a completely unique protocol sequence per battery
        # They will all stop dynamically based on their individual voltage triggers
        prot = Sequence([
            CC(rate=5.0 * c, until=engine.model.V_cell <= 2.5, time=10000),
            Rest(time=1800)
        ])
        protocol_payloads.append(prot)

    print(f"\nExecuting {len(param_payloads)} independent models concurrently...")
    print("Dropping into Rust Rayon thread-pool (Bypassing Python GIL).")
    
    # --- THE BATCH DISPATCH ---
    # max_workers dictates the size of the native thread pool.
    results = engine.solve_batch(
        parameters=param_payloads,
        protocols=protocol_payloads,
        max_workers=os.cpu_count(),
        show_progress=True
    )

    print("\nBatch execution complete. Generating plots...")

    # =========================================================================
    # Visualization
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Chen2020 DFN: Batch Execution with Unique Protocols", fontsize=15, fontweight="bold")
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    for c_rate, res, color in zip(c_rates, results, colors):
        t_hours = res["Time [s]"].data / 3600.0
        v_cell = res["V_cell"].data
        i_app = res["i_app"].data
        
        # Mask to cleanly plot capacity (only during active discharge)
        discharge_mask = i_app > 0.1
        capacity_ah = (res["Time [s]"].data[discharge_mask] * (5.0 * c_rate)) / 3600.0
        
        # Left Plot: Voltage vs Time (Shows the CC + Rest sequence)
        ax1.plot(t_hours, v_cell, label=f"{c_rate}C Discharge", color=color, linewidth=2)
        
        # Right Plot: Voltage vs Capacity
        ax2.plot(capacity_ah, v_cell[discharge_mask], label=f"{c_rate}C", color=color, linewidth=2)

    ax1.set_title("Terminal Voltage vs. Time", fontsize=12)
    ax1.set_xlabel("Time [hours]", fontsize=11)
    ax1.set_ylabel("Voltage [V]", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best")

    ax2.set_title("Terminal Voltage vs. Capacity", fontsize=12)
    ax2.set_xlabel("Discharge Capacity [Ah]", fontsize=11)
    ax2.set_ylabel("Voltage [V]", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_batch_demo()