#!/usr/bin/env python3
"""
plan/benchmark_suite.py
=======================
Automated benchmarking harness for ion_flux.
Measures execution latency across 0D, 1D, 1D-1D multi-scale, ALE, and Adjoint regimes.

BUG: The benchmark label ffi_overhead_ms is misleading for fast models: For small models with high step counts, that time is spent in the native Rust BDF stepping logic, not in Python marshalling.
"""

import os
import sys
import time
import math
import argparse
import json
from typing import Dict, Any, List, Callable, Optional
import numpy as np

# Ensure repository root and submodules are importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "models"))
    sys.path.insert(0, os.path.join(REPO_ROOT, "examples"))

import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

# Import Standard Models
from Chen2020_DFN import Chen2020_DFN
from Brosa2021_TSPMe import ExactTSPMe
from Paten2026_DC1 import Paten2026_DC1

# Import Example Models dynamically
from importlib.machinery import SourceFileLoader

ecm_mod = SourceFileLoader("ecm", os.path.join(REPO_ROOT, "examples", "1_equivalent_circuit.py")).load_module()
spm_mod = SourceFileLoader("spm", os.path.join(REPO_ROOT, "examples", "3_single_particle.py")).load_module()
ale_mod = SourceFileLoader("ale", os.path.join(REPO_ROOT, "examples", "4_swelling_spm.py")).load_module()

ThermoCoupledECM = ecm_mod.ThermoCoupledECM
SingleParticleModel = spm_mod.SingleParticleModel
SwellingSPM = ale_mod.SwellingSPM


# ==============================================================================
# In-Memory Test Models (Unique Test Oracles)
# ==============================================================================

class StiffAdjointDiffusionPDE(fx.PDE):
    """Extracted from tests/06_e2e_physics/test_adjoint_gradients.py."""
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    u = fx.State(domain=x, name="u")
    v = fx.State(domain=None, name="v")
    i_app = fx.State(domain=None, name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=v)
    k = fx.Parameter(default=2.0, name="k")

    def math(self):
        flux = -self.k * fx.grad(self.u, axis=self.x)
        return {
            "equations": {
                self.u: fx.dt(self.u) == -fx.div(flux, axis=self.x),
                self.v: self.v == self.u.right - self.i_app * 0.1
            },
            "boundaries": {
                flux: {"left": 0.0, "right": self.i_app}
            },
            "initial_conditions": {
                self.u: 0.0, self.v: 0.0, self.i_app: 0.0
            }
        }


# ==============================================================================
# Benchmark Definition Schema & Registry
#
# Lifecycle:
#   1. Model compilation (measures JIT compile latency)
#   2. Warmup executions (primes CPU caches and native factorizations)
#   3. Timed sample iterations (records wall-clock min, median, p95 latencies)
#   4. Telemetry harvest (extracts pure solver core time, FFI overhead, steps, RSS)
# ==============================================================================

class BenchmarkCase:
    def __init__(
        self,
        name: str,
        category: str,
        model_factory: Callable[[], fx.PDE],
        run_fn: Callable[[fx.Engine], None],
        iterations: int = 5,
        warmup: int = 1
    ):
        self.name = name
        self.category = category
        self.model_factory = model_factory
        self.run_fn = run_fn
        self.iterations = iterations
        self.warmup = warmup


def get_benchmark_registry() -> List[BenchmarkCase]:
    cases = []

    # --- Category A: 0D / Lumped Electrical-Thermal Networks ---
    cases.append(BenchmarkCase(
        name="ThermoCoupled_ECM_3600s",
        category="0D Lumped",
        model_factory=lambda: ThermoCoupledECM(),
        run_fn=lambda eng: eng.solve(
            protocol=Sequence([CC(rate=50.0, time=3600)]), 
            show_progress=False
        ),
        iterations=10,
        warmup=2
    ))

    # --- Category B: 1D Continuum Models ---
    cases.append(BenchmarkCase(
        name="SingleParticleModel_1800s",
        category="1D Continuum",
        model_factory=lambda: SingleParticleModel(),
        run_fn=lambda eng: eng.solve(
            protocol=Sequence([CC(rate=10.0, time=1800)]), 
            show_progress=False
        ),
        iterations=8,
        warmup=2
    ))

    # --- Category C: Moving Boundary / ALE Kinematics ---
    cases.append(BenchmarkCase(
        name="SwellingSPM_MovingMesh_1800s",
        category="ALE Kinematics",
        model_factory=lambda: SwellingSPM(),
        run_fn=lambda eng: eng.solve(
            protocol=Sequence([CC(rate=10.0, time=1800)]), 
            show_progress=False
        ),
        iterations=8,
        warmup=2
    ))

    # --- Category D: Multi-Scale 1D-1D Stiff DAEs ---
    cases.append(BenchmarkCase(
        name="Paten2026_DC1_Discharge_1200s",
        category="Multi-Scale DAE",
        model_factory=lambda: Paten2026_DC1(),
        run_fn=lambda eng: eng.solve(
            protocol=Sequence([CC(rate=5.0, time=1200)]), 
            show_progress=False
        ),
        iterations=5,
        warmup=1
    ))

    cases.append(BenchmarkCase(
        name="Brosa2021_TSPMe_Discharge_1200s",
        category="Multi-Scale DAE",
        model_factory=lambda: ExactTSPMe(),
        run_fn=lambda eng: eng.solve(
            protocol=Sequence([CC(rate=5.0, time=1200)]), 
            show_progress=False
        ),
        iterations=4,
        warmup=1
    ))

    cases.append(BenchmarkCase(
        name="Chen2020_DFN_Pulse_300s",
        category="Multi-Scale DAE",
        model_factory=lambda: Chen2020_DFN(),
        run_fn=lambda eng: eng.solve(
            protocol=Sequence([CC(rate=5.0, time=300)]), 
            show_progress=False
        ),
        iterations=3,
        warmup=1
    ))

    # --- Category E: Reverse-Mode Discrete Adjoint Sensitivities ---
    def run_adjoint_pass(eng: fx.Engine):
        t_eval = np.linspace(0, 10.0, 50)
        res = eng.solve(
            t_eval=t_eval, 
            parameters={"_term_mode": 1.0, "_term_i_target": 1.0, "k": 2.0},
            requires_grad=["k"],
            show_progress=False
        )
        target_v = np.zeros_like(res["v"].data)
        loss = fx.metrics.rmse(predicted=res["v"], target=target_v)
        grads = loss.backward()
        assert "k" in grads
        return res

    cases.append(BenchmarkCase(
        name="Discrete_Adjoint_VJP_Sensitivity",
        category="Adjoint Sensitivities",
        model_factory=lambda: StiffAdjointDiffusionPDE(),
        run_fn=run_adjoint_pass,
        iterations=10,
        warmup=2
    ))

    return cases


# ==============================================================================
# Benchmarking Engine
# ==============================================================================

def execute_benchmarks(cases: List[BenchmarkCase]) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(" 🚀 ion_flux Automated Performance Benchmark Suite")
    print("=" * 90)

    results: Dict[str, Any] = {"cases": {}, "categories": {}}

    for case in cases:
        print(f"\n▶ Compiling & Initializing: {case.name} [{case.category}]")
        
        t0_compile = time.perf_counter()
        model = case.model_factory()
        engine = fx.Engine(model=model, target="cpu:serial", solver_backend="native")
        t_compile = (time.perf_counter() - t0_compile) * 1000.0
        
        n_states = engine.layout.n_states if engine.layout else 0

        # Warmup executions
        for _ in range(case.warmup):
            case.run_fn(engine)

        # Timed executions
        latencies = []
        last_res = None
        for _ in range(case.iterations):
            t0 = time.perf_counter()
            last_res = case.run_fn(engine)
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        lat_arr = np.array(latencies)
        min_ms = float(np.min(lat_arr))
        median_ms = float(np.median(lat_arr))
        p95_ms = float(np.percentile(lat_arr, 95))
        std_ms = float(np.std(lat_arr))

        # Harvest native diagnostics and compiler topology
        diag = getattr(last_res, "diagnostics", {}) if last_res else {}
        tel = engine.telemetry if hasattr(engine, "telemetry") else None

        solver_time_ms = (
            diag.get("residual_time_us", 0) +
            diag.get("jacobian_assembly_time_us", 0) +
            diag.get("linear_solve_time_us", 0)
        ) / 1000.0
        ffi_overhead_ms = max(0.0, median_ms - solver_time_ms) if solver_time_ms > 0 else 0.0
        peak_rss_mb = diag.get("peak_rss_bytes", 0) / (1024.0 * 1024.0)

        results["cases"][case.name] = {
            "category": case.category,
            "n_states": n_states,
            "compile_time_ms": t_compile,
            "min_ms": min_ms,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "std_ms": std_ms,
            "iterations": case.iterations,
            "solver_time_ms": solver_time_ms,
            "ffi_overhead_ms": ffi_overhead_ms,
            "total_steps": diag.get("total_steps", 0),
            "newton_per_step": diag.get("newton_iterations_per_step", 0.0),
            "rejection_ratio": diag.get("step_rejection_ratio", 0.0),
            "peak_rss_mb": peak_rss_mb,
            "nnz": tel.nnz if tel else 0,
            "sparsity": tel.sparsity if tel else 0.0,
        }

        print(
            f"  Done: Median: {median_ms:8.2f} ms (Core: {solver_time_ms:6.2f} ms | FFI: {ffi_overhead_ms:5.2f} ms) "
            f"| P95: {p95_ms:8.2f} ms | RSS: {peak_rss_mb:5.1f} MB"
        )

    # Compute Geometric Means per Category
    categories: Dict[str, List[float]] = {}
    for case_data in results["cases"].values():
        categories.setdefault(case_data["category"], []).append(case_data["median_ms"])

    for cat_name, medians in categories.items():
        log_sum = sum(math.log(m) for m in medians)
        geomean = math.exp(log_sum / len(medians))
        results["categories"][cat_name] = {
            "geomean_median_ms": geomean,
            "count": len(medians)
        }

    return results


# ==============================================================================
# Reporting & Output
# ==============================================================================

def print_summary_table(results: Dict[str, Any]) -> None:
    print("\n\n" + "=" * 125)
    print(
        f"{'Benchmark Target':<32} | {'Category':<16} | {'N':<5} | "
        f"{'Wall (ms)':<10} | {'Core (ms)':<10} | {'FFI (ms)':<9} | "
        f"{'Steps':<6} | {'Newton/St':<9} | {'Peak RSS':<9}"
    )
    print("-" * 125)

    for name, d in results["cases"].items():
        core_str = f"{d.get('solver_time_ms', 0.0):10.2f}" if d.get('solver_time_ms', 0) > 0 else "       N/A"
        ffi_str = f"{d.get('ffi_overhead_ms', 0.0):9.2f}" if d.get('solver_time_ms', 0) > 0 else "      N/A"
        rss_str = f"{d.get('peak_rss_mb', 0.0):7.1f} MB" if d.get('peak_rss_mb', 0) > 0 else "      N/A"
        print(
            f"{name:<32} | {d['category']:<16} | {d['n_states']:<5} | "
            f"{d['median_ms']:10.2f} | {core_str} | {ffi_str} | "
            f"{d.get('total_steps', 0):<6} | {d.get('newton_per_step', 0.0):9.2f} | {rss_str}"
        )

    print("=" * 125)
    print(f"{'Category Geometric Means (Scale-Invariant Summary)':<70} | {'Geomean Median':<15} | {'Models':<6}")
    print("-" * 125)

    for cat_name, cat_data in results["categories"].items():
        print(f"{cat_name:<70} | {cat_data['geomean_median_ms']:12.2f} ms | {cat_data['count']:<6}")

    print("=" * 125 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ion_flux performance benchmark harness.")
    parser.add_argument("--export", type=str, default="", help="Path to export results as JSON.")
    args = parser.parse_args()

    registry = get_benchmark_registry()

    results = execute_benchmarks(registry)
    print_summary_table(results)

    if args.export:
        export_path = os.path.abspath(args.export)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Exported benchmark artifacts to: {export_path}")


if __name__ == "__main__":
    main()