#!/usr/bin/env python3
"""
10_latency_profiling.py
=======================
Head-to-head deterministic real-time latency benchmarking for battery models.
Profiles Full-Order P2D (Chen2020_DFN) and Asymptotic Reduced (Brosa2021_TSPMe)
under an identical closed-loop multi-rate co-simulation drive cycle.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root and models directory are importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in [REPO_ROOT, os.path.join(REPO_ROOT, "models")]:
    if path not in sys.path:
        sys.path.insert(0, path)

import ion_flux as fx
from Chen2020_DFN import Chen2020_DFN  # type: ignore
from Brosa2021_TSPMe import ExactTSPMe  # type: ignore


# ==============================================================================
# Domain Models & Data Structures
# ==============================================================================

@dataclass
class ModelProfileResult:
    name: str
    n_states: int
    latencies_us: np.ndarray
    voltages: np.ndarray
    applied_currents: np.ndarray
    stats: Dict[str, float]
    peak_cpu_load_pct: float
    scientific_voltages: Optional[np.ndarray] = None


def is_physically_sane(v: np.ndarray) -> np.ndarray:
    """Filters non-finite numbers and runaway unphysical voltage states."""
    return np.isfinite(v) & (v >= -1.0) & (v <= 10.0)


# ==============================================================================
# Drive Cycle Synthesis
# ==============================================================================

def generate_synthetic_drive_cycle(total_steps: int, dt: float) -> np.ndarray:
    """
    Synthesizes an automotive traction drive cycle scaled for a 5Ah cell at 100% SOC:
      - Mean discharge demand: +4.0 A (~0.8C)
      - Dynamic acceleration peaks: up to +8.5 A (~1.7C)
      - Mild regenerative braking pulses: down to -2.0 A (avoids overvoltage ceiling)
    """
    rng = np.random.default_rng(seed=42)
    time_arr = np.arange(total_steps) * dt

    base_discharge = 4.0 + 1.5 * np.sin(2 * np.pi * 0.05 * time_arr)
    accel_bursts = 3.0 * np.maximum(0.0, np.sin(2 * np.pi * 0.15 * time_arr))
    regen_pulses = -2.5 * np.maximum(0.0, np.sin(2 * np.pi * 0.10 * time_arr + np.pi))
    ripple = 0.3 * np.sin(2 * np.pi * 10.0 * time_arr)

    raw_noise = rng.normal(0.0, 0.4, total_steps)
    filtered_noise = np.zeros(total_steps)
    for i in range(1, total_steps):
        filtered_noise[i] = 0.1 * raw_noise[i] + 0.9 * filtered_noise[i - 1]

    return base_discharge + accel_bursts + regen_pulses + ripple + filtered_noise


# ==============================================================================
# Reference Baseline & Real-Time Profiler
# ==============================================================================

def compute_scientific_reference(
    engine: fx.Engine,
    model_name: str,
    current_profile: np.ndarray,
    dt: float = 0.01
) -> np.ndarray:
    """
    Computes a high-fidelity reference trajectory using adaptive VSVO BDF
    in 'scientific' mode to serve as ground truth for physics verification.
    """
    print(f"  [Ref] Computing scientific baseline for {model_name}...")
    session = engine.start_session(mode="scientific")

    # Prime session matching real-time initial warmup
    session.terminal.set_current(current_profile[0])
    session.step(dt=dt)

    n_steps = len(current_profile)
    voltages = np.zeros(n_steps, dtype=np.float64)

    for i in range(n_steps):
        session.terminal.set_current(current_profile[i])
        session.step(dt=dt)
        voltages[i] = session.get_scalar("V_cell")

    return voltages


def profile_battery_model(
    engine: fx.Engine,
    model_name: str,
    current_profile: np.ndarray,
    dt: float = 0.01,
    include_scientific_ref: bool = True
) -> ModelProfileResult:
    """
    Executes an unadorned 100 Hz co-simulation stepping loop:
      - Direct step timing via time.perf_counter_ns()
      - Isolates real-time timing from scientific reference evaluations
    """
    n_states = engine.layout.n_states
    freq_hz = int(round(1.0 / dt))
    print(f"  [Run] Profiling {model_name} (N = {n_states} states) at {freq_hz} Hz (dt = {dt*1000:.1f} ms)...")

    session = engine.start_session(mode="realtime")

    # Prime native JIT caches at the exact operational step size
    session.terminal.set_current(current_profile[0])
    session.step(dt=dt)

    n_steps = len(current_profile)
    budget_us = dt * 1_000_000.0

    latencies_us = np.zeros(n_steps, dtype=np.float64)
    voltages = np.zeros(n_steps, dtype=np.float64)

    for i in range(n_steps):
        i_app = current_profile[i]

        t0 = time.perf_counter_ns()
        session.terminal.set_current(i_app)
        session.step(dt=dt)
        t1 = time.perf_counter_ns()

        latencies_us[i] = (t1 - t0) / 1000.0
        voltages[i] = session.get_scalar("V_cell")

    stats = {
        "Min": float(np.min(latencies_us)),
        "Median": float(np.median(latencies_us)),
        "Mean": float(np.mean(latencies_us)),
        "99th": float(np.percentile(latencies_us, 99)),
        "WCET": float(np.max(latencies_us)),
    }

    scientific_voltages = None
    if include_scientific_ref:
        scientific_voltages = compute_scientific_reference(engine, model_name, current_profile, dt)

    return ModelProfileResult(
        name=model_name,
        n_states=n_states,
        latencies_us=latencies_us,
        voltages=voltages,
        applied_currents=current_profile,
        stats=stats,
        peak_cpu_load_pct=(stats["WCET"] / budget_us) * 100.0,
        scientific_voltages=scientific_voltages
    )


# ==============================================================================
# Visualization Dashboard (3x2 Layout)
# ==============================================================================

def render_comparative_dashboard(
    results: List[ModelProfileResult],
    current_profile: np.ndarray,
    dt: float = 0.01
) -> None:
    """Renders structured head-to-head physical and computational benchmarks."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    freq_hz = int(round(1.0 / dt))
    fig.suptitle(f"Real-Time HIL Performance: Chen2020 DFN vs Brosa2021 TSPMe ({freq_hz} Hz)", fontsize=14, fontweight="bold")

    time_arr = np.arange(len(current_profile)) * dt
    budget_us = dt * 1_000_000.0
    colors = {"Chen2020_DFN": "tab:blue", "ExactTSPMe": "tab:orange"}

    # --------------------------------------------------------------------------
    # Panel [0, 0]: Input Current Profile
    # --------------------------------------------------------------------------
    ax_i = axs[0, 0]
    ax_i.plot(time_arr, current_profile, color="black", linewidth=1.5)
    ax_i.set_title("Input Current Profile (Standard Pulse-Discharge)", fontsize=11, fontweight="bold")
    ax_i.set_ylabel("Applied Current [A]")
    ax_i.grid(True, linestyle="--", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel [0, 1]: Terminal Voltage Tracking (Physical Verification)
    # --------------------------------------------------------------------------
    ax_v = axs[0, 1]
    finite_voltages = []

    for r in results:
        ax_v.plot(time_arr, r.voltages, label=f"{r.name} [Realtime] (N={r.n_states})", color=colors[r.name], linewidth=1.8)
        if r.scientific_voltages is not None:
            ax_v.plot(time_arr, r.scientific_voltages, label=f"{r.name} [Scientific Ref]", color=colors[r.name], linestyle="--", linewidth=1.5, alpha=0.85)
            s_sane = r.scientific_voltages[is_physically_sane(r.scientific_voltages)]
            if len(s_sane) > 0:
                finite_voltages.extend(s_sane)
        rt_sane = r.voltages[is_physically_sane(r.voltages)]
        if len(rt_sane) > 0:
            finite_voltages.extend(rt_sane)

    ax_v.set_title("Terminal Voltage Tracking (Physics Verification)", fontsize=11, fontweight="bold")
    ax_v.set_ylabel("Voltage [V]")
    if finite_voltages:
        v_min = max(2.5, float(np.percentile(finite_voltages, 0.5)) - 0.05)
        v_max = min(4.5, float(np.percentile(finite_voltages, 99.5)) + 0.05)
        ax_v.set_ylim(v_min, v_max)
    else:
        ax_v.set_ylim(2.5, 4.5)

    ax_v.legend(loc="lower left", frameon=True)
    ax_v.grid(True, linestyle="--", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel [1, 0]: Tracking Error vs Time (Realtime - Scientific in mV)
    # --------------------------------------------------------------------------
    ax_err = axs[1, 0]
    ax_err.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    all_valid_errors = []
    for r in results:
        if r.scientific_voltages is not None:
            valid_mask = is_physically_sane(r.voltages) & is_physically_sane(r.scientific_voltages)
            err_mv = np.full_like(r.voltages, np.nan)
            err_mv[valid_mask] = (r.voltages[valid_mask] - r.scientific_voltages[valid_mask]) * 1000.0

            n_valid = int(np.sum(valid_mask))
            if n_valid > 0:
                valid_errs = err_mv[valid_mask]
                rmse = np.sqrt(np.mean(np.square(np.clip(valid_errs, -1e6, 1e6))))
                max_e = np.max(np.abs(valid_errs))
                lbl = f"{r.name} (RMS: {rmse:.2f} mV, Max: {max_e:.2f} mV)"
                all_valid_errors.extend(valid_errs)
            else:
                lbl = f"{r.name} (All steps diverged/NaN)"

            ax_err.plot(time_arr, err_mv, label=lbl, color=colors[r.name], linewidth=1.6)

            # Annotate point of divergence if any occurred
            if n_valid < len(r.voltages):
                invalid_indices = np.where(~valid_mask)[0]
                if len(invalid_indices) > 0:
                    t_div = time_arr[invalid_indices[0]]
                    ax_err.axvline(t_div, color=colors[r.name], linestyle=":", linewidth=2, alpha=0.8)
                    ax_err.annotate(
                        f"{r.name} diverged\n(t={t_div:.2f}s)",
                        xy=(t_div, 0),
                        xytext=(t_div + 0.3, 5),
                        arrowprops=dict(arrowstyle="->", color=colors[r.name]),
                        fontsize=8,
                        color=colors[r.name],
                        fontweight="bold"
                    )

    ax_err.set_title("Real-Time Tracking Error vs Scientific Baseline (ΔV in mV)", fontsize=11, fontweight="bold")
    ax_err.set_ylabel("Error: $V_{RT} - V_{Sci}$ [mV]")
    ax_err.set_xlabel("Time [s]")
    if all_valid_errors:
        p99_err = float(np.percentile(np.abs(all_valid_errors), 99))
        err_bound = max(p99_err * 1.5, 10.0)
        err_bound = min(err_bound, 500.0)  # Bound to 500 mV so converged models remain readable
        ax_err.set_ylim(-err_bound, err_bound)

    ax_err.legend(loc="upper right", frameon=True)
    ax_err.grid(True, linestyle="--", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel [1, 1]: Step Latency Traces vs Real-Time Deadline
    # --------------------------------------------------------------------------
    ax_lat = axs[1, 1]
    for r in results:
        ax_lat.plot(time_arr, r.latencies_us, label=r.name, color=colors[r.name], alpha=0.8, linewidth=1.2)
    ax_lat.set_title(f"Step Latency vs {freq_hz} Hz Deadline ({budget_us:,.0f} µs)", fontsize=11, fontweight="bold")
    ax_lat.set_xlabel("Time [s]")
    ax_lat.set_ylabel("Step Latency [µs]")
    ax_lat.set_ylim(0, max(np.percentile(results[0].latencies_us, 99.5) * 1.5, 50.0))
    ax_lat.legend(loc="upper right", frameon=True)
    ax_lat.grid(True, linestyle="--", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel [2, 0]: Latency Distributions & WCET Metrics
    # --------------------------------------------------------------------------
    ax_hist = axs[2, 0]
    p995_lat = max(float(np.percentile(r.latencies_us, 99.5)) for r in results)
    x_max_lat = max(p995_lat * 1.65, 110.0)
    lat_bins = np.linspace(0.0, x_max_lat, 45)
    box_y_positions = [0.96, 0.60]

    for idx, r in enumerate(results):
        c = colors.get(r.name, "gray")
        ax_hist.hist(
            r.latencies_us, bins=lat_bins, color=c,
            alpha=0.5, edgecolor="black", linewidth=0.6
        )
        ax_hist.axvline(r.stats["Median"], color=c, linestyle="--", linewidth=1.3, alpha=0.85)

        stat_box = (
            f"{r.name}\n"
            f"Median : {r.stats['Median']:7.1f} µs\n"
            f"99th % : {r.stats['99th']:7.1f} µs\n"
            f"WCET   : {r.stats['WCET']:7.1f} µs\n"
            f"Load   : {r.peak_cpu_load_pct:6.2f}%"
        )
        y_pos = box_y_positions[idx] if idx < len(box_y_positions) else max(0.24, 0.96 - 0.36 * idx)
        ax_hist.text(
            0.98, y_pos, stat_box, transform=ax_hist.transAxes,
            ha="right", va="top", fontfamily="monospace", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=c, linewidth=1.4, alpha=0.92)
        )

    ax_hist.set_title("Latency Distributions & WCET Metrics (99.5% View)", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Latency [µs]")
    ax_hist.set_ylabel("Step Count")
    ax_hist.set_xlim(0.0, x_max_lat)
    ax_hist.grid(True, linestyle="--", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel [2, 1]: Voltage Error Distribution Histogram
    # --------------------------------------------------------------------------
    ax_err_hist = axs[2, 1]
    ax_err_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    valid_err_series = []
    for r in results:
        if r.scientific_voltages is not None:
            valid_mask = is_physically_sane(r.voltages) & is_physically_sane(r.scientific_voltages)
            if np.any(valid_mask):
                v_err = (r.voltages[valid_mask] - r.scientific_voltages[valid_mask]) * 1000.0
                valid_err_series.append((r, v_err))

    if valid_err_series:
        p995_err = max(float(np.percentile(np.abs(ve), 99.5)) for _, ve in valid_err_series)
        err_lim = max(p995_err * 1.65, 0.05)
        err_bins = np.linspace(-err_lim, err_lim, 45)

        for idx, (r, v_err) in enumerate(valid_err_series):
            c = colors.get(r.name, "gray")
            ax_err_hist.hist(
                v_err, bins=err_bins, color=c,
                alpha=0.5, edgecolor="black", linewidth=0.6
            )

            mean_e = float(np.mean(v_err))
            med_e = float(np.median(v_err))
            std_e = float(np.std(v_err))
            max_e = float(np.max(np.abs(v_err)))

            err_box = (
                f"{r.name}\n"
                f"Mean   : {mean_e:+7.3f} mV\n"
                f"Median : {med_e:+7.3f} mV\n"
                f"StdDev : {std_e:7.3f} mV\n"
                f"Max    : {max_e:7.3f} mV"
            )
            y_pos = box_y_positions[idx] if idx < len(box_y_positions) else max(0.24, 0.96 - 0.36 * idx)
            ax_err_hist.text(
                0.98, y_pos, err_box, transform=ax_err_hist.transAxes,
                ha="right", va="top", fontfamily="monospace", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=c, linewidth=1.4, alpha=0.92)
            )

        ax_err_hist.set_xlim(-err_lim, err_lim)

    ax_err_hist.set_title("Voltage Error Distribution (Realtime Fidelity)", fontsize=11, fontweight="bold")
    ax_err_hist.set_xlabel("Tracking Error [mV]")
    ax_err_hist.set_ylabel("Step Count")
    ax_err_hist.grid(True, linestyle="--", alpha=0.6)

    plt.show()


# ==============================================================================
# Main Orchestration
# ==============================================================================

def main() -> None:
    dt = 0.001         # 1 kHz fixed time step (1.0 ms)
    n_steps = 10000    # 10.0 seconds total simulation (10,000 steps)

    print("=" * 82)
    print(" Deterministic HIL Benchmarking: Chen2020_DFN vs Brosa2021_TSPMe")
    print(f" Execution Profile: 1 kHz Synchronous Stepping (1,000 µs Deadline)")
    print("=" * 82)

    # 1. Standard GITT-style pulse profile:
    # 0-3s: 1C (5A) | 3-6s: Rest (0A) | 6-8s: 1.5C (7.5A) | 8-10s: Rest (0A)
    time_arr = np.arange(n_steps) * dt
    current_profile = np.zeros(n_steps, dtype=np.float64)
    current_profile[(time_arr >= 0.0) & (time_arr < 3.0)] = 5.0
    current_profile[(time_arr >= 3.0) & (time_arr < 6.0)] = 0.0
    current_profile[(time_arr >= 6.0) & (time_arr < 8.0)] = 7.5
    current_profile[(time_arr >= 8.0) & (time_arr <= 10.0)] = 0.0

    # 2. Compile models Ahead-Of-Time
    print("\n[1/3] Initializing compiled models...")
    engine_dfn = fx.Engine(model=Chen2020_DFN(), target="cpu:serial")
    engine_tspme = fx.Engine(model=ExactTSPMe(), target="cpu:serial")

    # 3. Execute standardized benchmark loops
    freq_hz = int(round(1.0 / dt))
    print(f"\n[2/3] Executing {freq_hz} Hz stepping loops ({n_steps:,} steps per model)...")
    res_dfn = profile_battery_model(engine_dfn, "Chen2020_DFN", current_profile, dt)
    res_tspme = profile_battery_model(engine_tspme, "ExactTSPMe", current_profile, dt)

    # 4. Summary table including verification fidelity
    print("\n[3/3] Benchmark & Verification Summary:")
    print("=" * 96)
    print(f"{'Model Name':<16} | {'States':<6} | {'Median':<10} | {'WCET':<10} | {'1kHz CPU':<10} | {'Valid Steps':<12} | {'RMSE [mV]':<10} | {'Max Err [mV]':<12}")
    print("-" * 96)
    for r in [res_dfn, res_tspme]:
        valid_steps_str = f"{len(r.voltages)}/{len(r.voltages)}"
        rmse_str = "N/A"
        max_err_str = "N/A"
        if r.scientific_voltages is not None:
            v_mask = is_physically_sane(r.voltages) & is_physically_sane(r.scientific_voltages)
            n_valid = int(np.sum(v_mask))
            valid_steps_str = f"{n_valid}/{len(r.voltages)}"
            if n_valid > 0:
                v_err = (r.voltages[v_mask] - r.scientific_voltages[v_mask]) * 1000.0
                rmse_str = f"{np.sqrt(np.mean(np.square(np.clip(v_err, -1e6, 1e6)))):.2f}"
                max_err_str = f"{np.max(np.abs(v_err)):.2f}"
            else:
                rmse_str = "DIVERGED"
                max_err_str = "DIVERGED"

        print(
            f"{r.name:<16} | {r.n_states:<6} | {r.stats['Median']:7.1f} µs | "
            f"{r.stats['WCET']:7.1f} µs | {r.peak_cpu_load_pct:8.2f}% | "
            f"{valid_steps_str:<12} | {rmse_str:<10} | {max_err_str:<12}"
        )
    print("=" * 96 + "\n")

    render_comparative_dashboard([res_dfn, res_tspme], current_profile, dt)


if __name__ == "__main__":
    main()