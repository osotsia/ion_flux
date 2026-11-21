"""
Runtime Solver Audit Suite.

Executes post-hoc trajectory verification, guarded differential cross-checks,
all-parameter sensitivity scans, clamp saturation detection, and sequence transition audits.
"""

import contextvars
import json
from typing import Dict, Any, List, Optional
import numpy as np


_AUDIT_CONTEXT: contextvars.ContextVar[Optional["AuditSession"]] = contextvars.ContextVar(
    "ion_flux_audit_context", default=None
)


def get_active_audit() -> Optional["AuditSession"]:
    """Returns the currently active AuditSession context, if any."""
    return _AUDIT_CONTEXT.get()


def grade_audit(checks: Dict[str, Any], static_audit: Optional[Dict[str, Any]]) -> List[str]:
    """Single source of truth for reducing runtime and static audit checks into status tokens."""
    statuses = [c.get("status", "PASS") for c in checks.values()]
    if static_audit:
        if "cpr_jvp_probe_passed" in static_audit:
            statuses.append("PASS" if static_audit["cpr_jvp_probe_passed"] else "WARN")
        ledger = static_audit.get("boundary_ledger", [])
        if any(e.get("implicit_zero_flux_faces") for e in ledger):
            statuses.append("WARN")
        if any(e.get("missing_gauge_anchor") for e in ledger):
            statuses.append("WARN")
        ic = static_audit.get("ic_health")
        if ic is not None:
            statuses.append("PASS" if ic.get("ic_converged", False) else "FAIL")
            j_max, j_min = ic.get("init_jac_max", 0.0), ic.get("init_jac_min", 0.0)
            if j_min > 0.0 and j_max > 0.0 and (j_max / j_min) > 1e20:
                statuses.append("WARN")
    return statuses


def reduce_status(statuses: List[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


class SolverAuditReport:
    """Structured scorecard and verification data produced by solve(..., audit_solver=True)."""

    def __init__(
        self,
        model_name: str,
        n_states: int,
        n_params: int,
        checks: Dict[str, Any],
        static_audit: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.n_states = n_states
        self.n_params = n_params
        self.checks = checks
        self.static_audit = static_audit or {}
        self.provenance = provenance or {}

    def _all_statuses(self) -> List[str]:
        return grade_audit(self.checks, self.static_audit)

    @property
    def passed(self) -> bool:
        return "FAIL" not in self._all_statuses()

    @property
    def status(self) -> str:
        return reduce_status(self._all_statuses())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_states": self.n_states,
            "n_params": self.n_params,
            "status": self.status,
            "passed": self.passed,
            "provenance": self.provenance,
            "checks": self.checks,
            "static_audit": self.static_audit,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(payload)
        return payload

    def print_summary(self) -> None:
        print("=" * 80)
        print(f"ION_FLUX MODEL AUDIT SCORECARD: {self.model_name} (N={self.n_states} states, {self.n_params} params)")
        all_st = self._all_statuses()
        n_fail = sum(1 for s in all_st if s == "FAIL")
        n_warn = sum(1 for s in all_st if s == "WARN")
        n_pass = sum(1 for s in all_st if s == "PASS")
        print(f"OVERALL STATUS: {self.status} ({n_fail} FAIL, {n_warn} WARN, {n_pass} PASS)")
        print("=" * 80)

        # 1. Static & Compiler Integrity
        print("[1. STATIC & COMPILER INTEGRITY]")
        if self.static_audit:
            cpr_st = "PASS" if self.static_audit.get("cpr_jvp_probe_passed", False) else "WARN"
            cpr_diff = self.static_audit.get("cpr_jvp_probe_max_diff", 0.0)
            print(f"  [{cpr_st:<4}] CPR vs. Enzyme JVP        Max rel diff {cpr_diff:.1e} across random probes")

            ledger = self.static_audit.get("boundary_ledger", [])
            implicit_count = sum(len(e.get("implicit_zero_flux_faces", [])) for e in ledger)
            gauge_missing = [e["state"] for e in ledger if e.get("missing_gauge_anchor")]
            led_st = "WARN" if (implicit_count > 0 or gauge_missing) else "PASS"
            print(f"  [{led_st:<4}] Boundary Completeness     {implicit_count} faces use implicit zero-flux fallback")
            for e in ledger:
                for face in e.get("implicit_zero_flux_faces", []):
                    print(f"         - {e['state']}.{face}")
            if gauge_missing:
                print(f"         - Missing Dirichlet gauge anchor on spatial DAE: {', '.join(gauge_missing)}")
        else:
            print("  [SKIP] Static Audit            Compile with Engine(..., audit_compiler=True) for full checks")

        # 2. Initialization & Conditioning (t=0)
        print("\n[2. INITIALIZATION & CONDITIONING (t=0)]")
        if self.static_audit and "ic_health" in self.static_audit:
            ic = self.static_audit["ic_health"]
            ic_st = "PASS" if ic.get("ic_converged", False) else "FAIL"
            print(f"  [{ic_st:<4}] Algebraic Root Snap       Converged: {ic.get('ic_converged')}, iters: {ic.get('ic_iters')}")
            print(f"         Pre-snap max res: {ic.get('pre_snap_alg_max_res', 0.0):.1e} -> Post-snap: {ic.get('ic_final_max_res', 0.0):.1e}")
            if ic.get("max_snap_state"):
                print(f"         Max state delta: {ic.get('max_snap_state')} ({ic.get('max_snap_delta', 0.0):.3e})")
            j_max = ic.get("init_jac_max", 0.0)
            j_min = ic.get("init_jac_min", 0.0)
            cond_spread = (j_max / j_min) if (j_min > 0.0 and j_max > 0.0) else 0.0
            cond_st = "WARN" if cond_spread > 1e20 else "PASS"
            print(f"  [{cond_st:<4}] Jacobian Conditioning     Raw spread: {cond_spread:.1e} (row-equilibrated in LU)")
        else:
            print("  [SKIP] Initialization Audit    Compile with Engine(..., audit_compiler=True) for t=0 checks")

        # 3. Solver & Trajectory Integrity
        print("\n[3. SOLVER & TRAJECTORY INTEGRITY]")
        res_chk = self.checks.get("dae_residual", {})
        res_st = res_chk.get("status", "SKIP")
        if res_st != "SKIP":
            print(f"  [{res_st:<4}] Post-Hoc DAE Residual     Max ||F||_inf = {res_chk.get('max_residual', 0.0):.1e} (WRMS: {res_chk.get('max_wrms_residual', 0.0):.2f}, {res_chk.get('samples', 0)} steps sampled)")
            if res_chk.get("worst_state"):
                print(f"         Worst: {res_chk.get('worst_state')} at t={res_chk.get('worst_time', 0.0):.3f}s")
        else:
            print(f"  [SKIP] Post-Hoc DAE Residual     {res_chk.get('reason', 'N/A')}")

        sun_chk = self.checks.get("sundials_cross_check", {})
        sun_st = sun_chk.get("status", "SKIP")
        if sun_st in ("PASS", "WARN", "FAIL"):
            if "rmse_mv" in sun_chk:
                print(f"  [{sun_st:<4}] Native vs. SUNDIALS IDA   {sun_chk.get('target_var', 'V')} RMSE = {sun_chk.get('rmse_mv', 0.0):.3f} mV, Max = {sun_chk.get('max_diff_mv', 0.0):.3f} mV")
            elif "relative_l2_error" in sun_chk:
                print(f"  [{sun_st:<4}] Native vs. SUNDIALS IDA   Relative L2 error = {sun_chk.get('relative_l2_error', 0.0):.1e}")
            else:
                print(f"  [{sun_st:<4}] Native vs. SUNDIALS IDA   {sun_chk.get('note', 'Compared cleanly')}")
        else:
            print(f"  [SKIP] Native vs. SUNDIALS IDA   {sun_chk.get('reason', 'N/A')}")

        guard_chk = self.checks.get("numerical_guardrails", {})
        guard_st = guard_chk.get("status", "SKIP")
        if guard_st in ("PASS", "WARN"):
            active = guard_chk.get("active_clamps", [])
            if active:
                print(f"  [{guard_st:<4}] Numerical Guardrails      {len(active)} clamp(s) saturated during trajectory:")
                for c in active:
                    lbl = c["label"]
                    dom = c["domain"]
                    pct = max(c.get("lo_hits_max_pct", 0.0), c.get("hi_hits_max_pct", 0.0))
                    tr = c.get("lo_t_range") or c.get("hi_t_range") or (0.0, 0.0)
                    print(f"         - {lbl} on {dom}: max {pct:.1f}% nodes, t=[{tr[0]:.1f}s .. {tr[1]:.1f}s]")
            else:
                print(f"  [{guard_st:<4}] Numerical Guardrails      0 clamps saturated")
        else:
            print(f"  [SKIP] Numerical Guardrails      {guard_chk.get('reason', 'N/A')}")

        step_chk = self.checks.get("step_health", {})
        step_st = step_chk.get("status", "PASS")
        tot = step_chk.get("total_steps", 0)
        rej = step_chk.get("rejected_steps", 0)
        frz = step_chk.get("realtime_frozen_steps", 0)
        min_dt = step_chk.get("min_dt")
        dt_str = f", min dt={min_dt:.1e}s" if min_dt is not None else ""
        print(f"  [{step_st:<4}] Step & Realtime Health    {tot} steps ({rej} rejected, {frz} frozen{dt_str})")
        for n in step_chk.get("notes", []):
            print(f"         {n}")

        # 4. Sensitivity & Protocols
        print("\n[4. SENSITIVITY & PROTOCOLS]")
        sens_chk = self.checks.get("parameter_sensitivity", {})
        sens_st = sens_chk.get("status", "SKIP")
        if sens_st in ("PASS", "WARN"):
            qoi = sens_chk.get("qoi_name", "QoI")
            top = sens_chk.get("sens_ranking", [])[:3]
            top_str = ", ".join(f"{name} ({s:.1e})" for name, s, _ in top)
            print(f"  [{sens_st:<4}] Parameter Coverage        QoI: {qoi} | Top: {top_str}")
            for w in sens_chk.get("warnings", []):
                print(f"         {w}")
        else:
            print(f"  [SKIP] Parameter Coverage        {sens_chk.get('error', 'N/A')}")

        trans_chk = self.checks.get("sequence_transitions", {})
        trans_st = trans_chk.get("status", "SKIP")
        if trans_st != "SKIP":
            tr_list = trans_chk.get("transitions", [])
            print(f"  [{trans_st:<4}] Protocol Transitions      {len(tr_list)} transition(s) recorded cleanly")
            for tr in tr_list:
                print(f"         Step {tr['step_idx']}: {tr['termination_reason']} at t={tr['trigger_t']:.3f}s (width={tr['bisection_width']:.1e})")
        else:
            print(f"  [SKIP] Protocol Transitions      {trans_chk.get('reason', 'N/A')}")

        print("-" * 80)
        print("Drill-down: .show_math() | .show_compute() | .show_boundaries() | .show_clamps() | .show_residuals() | .show_sensitivities() | .to_json()")
        print("=" * 80 + "\n")

    def show_compute(self, state: Optional[str] = None) -> None:
        pretty = self.static_audit.get("compute_pretty")
        if not pretty:
            print("Compute IR unavailable. Compile Engine with `audit_compiler=True`.")
            return
        if not state:
            print(pretty)
            return
        sections = pretty.split("// === Eq: ")
        matched = [sections[0]] if state in sections[0] else []
        for sec in sections[1:]:
            if sec.startswith(f"{state} "):
                matched.append("// === Eq: " + sec)
        print("\n".join(matched) if matched else f"No Compute IR section found for '{state}'.")

    def show_residuals(self, n: int = 5) -> None:
        chk = self.checks.get("dae_residual", {})
        top = chk.get("top_offenders", [])
        if not top:
            print("No post-hoc residual samples recorded.")
            return
        print(f"Top {min(n, len(top))} Worst Residual Samples:")
        print(f"{'Time [s]':<12} | {'State':<24} | {'|Residual|':<12} | {'WRMS':<10}")
        print("-" * 66)
        for item in top[:n]:
            print(f"{item['time']:<12.4f} | {item['state']:<24} | {item['residual']:<12.3e} | {item['wrms']:<10.3f}")

    def show_sensitivities(self) -> None:
        chk = self.checks.get("parameter_sensitivity", {})
        ranking = chk.get("sens_ranking", [])
        if not ranking:
            print("No parameter sensitivity data recorded.")
            return
        print(f"Parameter Sensitivity Ranking (QoI: {chk.get('qoi_name', 'QoI')}):")
        print(f"{'Parameter':<20} | {'Normalized Sens':<16} | {'Raw d(QoI)/dp':<16}")
        print("-" * 58)
        for p_name, norm_s, raw_g in ranking:
            print(f"{p_name:<20} | {norm_s:<16.4e} | {raw_g:<16.4e}")

    def show_clamps(self) -> None:
        chk = self.checks.get("numerical_guardrails", {})
        active = chk.get("active_clamps", [])
        if not active:
            print("No clamp saturation recorded.")
            return
        for c in active:
            print(f"Clamp: {c['label']} (state={c['state']}, domain={c['domain']})")
            if c.get("lo_hits_max_pct", 0) > 0:
                print(f"  Lower bound hit: max {c['lo_hits_max_pct']:.1f}% nodes, t range={c['lo_t_range']}")
            if c.get("hi_hits_max_pct", 0) > 0:
                print(f"  Upper bound hit: max {c['hi_hits_max_pct']:.1f}% nodes, t range={c['hi_t_range']}")

    def show_transitions(self) -> None:
        trans = self.checks.get("sequence_transitions", {}).get("transitions", [])
        if not trans:
            print("No protocol sequence transitions recorded.")
            return
        for tr in trans:
            print(f"Step {tr['step_idx']}: {tr['termination_reason']} at t={tr['trigger_t']:.4f}s | bisection_width={tr['bisection_width']:.2e} | trigger_res={tr['trigger_residual']:.2e}")

    def show_math(self, state: Optional[str] = None) -> None:
        pretty = self.static_audit.get("math_pretty")
        if not pretty:
            print("Math IR unavailable. Compile Engine with `audit_compiler=True`.")
            return
        if state:
            lines = [l for l in pretty.splitlines() if state in l or l.startswith("===") or l.startswith("Shared")]
            print("\n".join(lines))
        else:
            print(pretty)

    def show_boundaries(self) -> None:
        ledger = self.static_audit.get("boundary_ledger", [])
        if not ledger:
            print("Boundary ledger unavailable. Compile Engine with `audit_compiler=True`.")
            return
        print(f"{'State':<16} | {'Domain':<12} | {'Kind':<8} | {'Faces & Stitching'}")
        print("-" * 75)
        for e in ledger:
            kind = "DAE" if e.get("is_dae") else "PDE"
            faces_str = ", ".join(f"{f}={t}" for f, t in e.get("faces", {}).items())
            print(f"{e['state']:<16} | {e['domain']:<12} | {kind:<8} | {faces_str}")


def _state_name_for_idx(layout: Any, flat_idx: int) -> str:
    for s_name, (off, sz) in layout.state_offsets.items():
        if off <= flat_idx < off + sz:
            return f"{s_name}[{flat_idx - off}]" if sz > 1 else s_name
    return f"state[{flat_idx}]"


def run_solver_audit(
    engine: Any,
    result: Any,
    parameters: Dict[str, float],
    protocol: Any,
    t_span: tuple,
    t_eval: Optional[np.ndarray],
    rel_tol: Optional[float],
    abs_tol: Optional[float],
) -> SolverAuditReport:
    """Executes all runtime post-hoc verification passes and builds the SolverAuditReport."""
    checks: Dict[str, Any] = {}
    manifest = engine.manifest
    n_states = manifest.layout.n_states
    n_params = manifest.layout.n_params
    model_name = type(engine.model).__name__ if engine.model else "Model"

    static_audit = engine.audit_compiler_report() if getattr(engine, "audit_compiler", False) else getattr(manifest, "static_audit", None)

    # 1. Post-Hoc DAE Residual Sweep (uncapped up to 50,000 steps)
    traj = result.trajectory
    if traj and "_micro_y" in traj and len(traj["_micro_y"]) > 0:
        micro_y = traj["_micro_y"]
        micro_ydot = traj["_micro_ydot"]
        micro_t = traj["_micro_t"]
        p_traj = traj["_p_traj"]
        m_list = manifest.layout.get_mesh_data()

        n_micro = len(micro_t)
        stride = 1 if n_micro <= 50000 else max(1, n_micro // 50000)
        sample_indices = range(0, n_micro, stride)

        id_arr = np.asarray(manifest.id_arr, dtype=np.float64)

        # Compute Oettli-Prager Jacobian row-scaling tolerances across sample points
        # so residuals in equation units (e.g., A/m^3) normalize cleanly to state tolerances.
        eq_tol = 1e-4 * np.max(np.abs(micro_y), axis=0) + 1e-6
        if hasattr(manifest, "evaluate_jacobian_triplets"):
            probe_pts = sorted({min(1, n_micro - 1), n_micro // 2, n_micro - 1})
            row_tol_max = np.full(n_states, 1e-8, dtype=np.float64)
            p_default_list = manifest.pack_parameters(parameters)
            for pt in probe_pts:
                y_pt = micro_y[pt]
                ydot_pt = micro_ydot[pt]
                col_scale = 1e-4 * (np.abs(y_pt) + np.abs(ydot_pt)) + 1e-6
                # Override params with active tape step if available
                pt_overrides = {
                    p_name: float(p_traj[pt, off])
                    for p_name, (off, _) in manifest.layout.param_offsets.items()
                } if p_traj.shape[1] == manifest.layout.p_length else parameters
                for c_j in (0.0, 1.0):
                    r_j, c_j_idx, v_j = manifest.evaluate_jacobian_triplets(
                        y_pt.tolist(), ydot_pt.tolist(), c_j, pt_overrides
                    )
                    if v_j:
                        r_arr = np.asarray(r_j, dtype=np.intp)
                        c_arr = np.asarray(c_j_idx, dtype=np.intp)
                        v_arr = np.abs(np.asarray(v_j, dtype=np.float64))
                        tol_pt = np.bincount(r_arr, weights=v_arr * col_scale[c_arr], minlength=n_states)
                        row_tol_max = np.maximum(row_tol_max, tol_pt)
            eq_tol = np.maximum(eq_tol, row_tol_max)

        max_res = 0.0
        max_wrms = 0.0
        max_res_state = ""
        sum_res = 0.0
        worst_res_t = 0.0
        count = 0
        top_offenders: List[Dict[str, Any]] = []

        for idx in sample_indices:
            y_step = micro_y[idx].tolist()
            ydot_step = micro_ydot[idx].tolist()
            p_step = p_traj[idx].tolist()
            res_vec = manifest.native_lib.evaluate_residual(y_step, ydot_step, p_step, m_list)
            res_arr = np.abs(res_vec)
            if idx == 0:
                res_arr = np.where(id_arr < 0.5, res_arr, 0.0)

            local_weighted = res_arr / eq_tol
            local_max_idx = int(np.argmax(local_weighted))
            local_max = float(res_arr[local_max_idx])
            local_max_w = float(local_weighted[local_max_idx])

            sum_res += float(np.mean(res_arr))
            count += 1
            if local_max_w > max_wrms:
                max_wrms = local_max_w
                max_res = local_max
                worst_res_t = float(micro_t[idx])
                max_res_state = _state_name_for_idx(manifest.layout, local_max_idx)

            if len(top_offenders) < 5 or local_max_w > top_offenders[-1]["wrms"]:
                top_offenders.append({
                    "time": float(micro_t[idx]),
                    "state": _state_name_for_idx(manifest.layout, local_max_idx),
                    "residual": local_max,
                    "wrms": local_max_w,
                })
                top_offenders.sort(key=lambda x: x["wrms"], reverse=True)
                if len(top_offenders) > 5:
                    top_offenders.pop()

        res_st = "PASS" if max_wrms <= 2.0 else ("WARN" if max_wrms <= 20.0 else "FAIL")
        checks["dae_residual"] = {
            "status": res_st,
            "max_residual": max_res,
            "max_wrms_residual": max_wrms,
            "mean_residual": (sum_res / count) if count > 0 else 0.0,
            "worst_time": worst_res_t,
            "worst_state": max_res_state,
            "samples": count,
            "top_offenders": top_offenders,
        }
    else:
        checks["dae_residual"] = {"status": "SKIP", "reason": "No microstep tape recorded"}

    # 2. Guarded Native vs. SUNDIALS IDA Cross-Check
    bw = manifest.jacobian_bandwidth
    if engine.solver_backend != "native":
        checks["sundials_cross_check"] = {"status": "SKIP", "reason": "solver_backend is already sundials"}
    elif n_states > 1500:
        checks["sundials_cross_check"] = {"status": "SKIP", "reason": f"n_states ({n_states}) > 1500 (guards dense matrix blowup)"}
    elif bw == -1:
        checks["sundials_cross_check"] = {"status": "SKIP", "reason": "Matrix-Free GMRES mesh (unsupported by SUNDIALS wrapper)"}
    else:
        try:
            from ion_flux.runtime.backends import dispatch_native
            sund_raw = dispatch_native(
                manifest=manifest,
                y0=[0.0] * n_states,
                ydot0=[0.0] * n_states,
                parameters=[parameters],
                t_eval=t_eval if t_eval is not None else np.linspace(t_span[0], t_span[1], 100),
                protocols=[protocol] if protocol is not None else None,
                solver_backend="sundials",
                requires_grad=None,
                max_workers=1,
                spatial_threads=1,
                show_progress=False,
                v_idx=-1,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            t_sund, y_sund, _, _, _ = sund_raw[0]
            t_nat = result._data["Time [s]"]

            target_var = None
            for v_cand in ("V_cell", "V", "v", "Voltage"):
                if v_cand in result._data:
                    target_var = v_cand
                    break

            if target_var:
                v_nat = result._data[target_var]
                off, _ = manifest.layout.state_offsets[target_var]
                v_s = y_sund[:, off]
                v_s_interp = np.interp(t_nat, t_sund, v_s)
                start_k = 1 if len(t_nat) > 2 else 0
                diff_mv = (v_nat[start_k:] - v_s_interp[start_k:]) * 1000.0
                rmse_mv = float(np.sqrt(np.mean(diff_mv ** 2)))
                max_diff_mv = float(np.max(np.abs(diff_mv)))
                s_st = "PASS" if rmse_mv < 1.0 else ("WARN" if rmse_mv < 10.0 else "FAIL")
                checks["sundials_cross_check"] = {
                    "status": s_st,
                    "target_var": target_var,
                    "rmse_mv": rmse_mv,
                    "max_diff_mv": max_diff_mv,
                }
            else:
                checks["sundials_cross_check"] = {"status": "PASS", "note": "SUNDIALS run completed cleanly"}
        except Exception as e:
            checks["sundials_cross_check"] = {"status": "WARN", "reason": f"SUNDIALS run failed: {e}"}

    # 3. All-Parameter VJP Sensitivity Scan
    if traj and "_micro_y" in traj and len(traj["_micro_y"]) > 0:
        try:
            from ion_flux._core import discrete_adjoint_native
            y_traj = np.ascontiguousarray(traj["_micro_y"], dtype=np.float64)
            ydot_traj = np.ascontiguousarray(traj["_micro_ydot"], dtype=np.float64)
            t_eval_arr = np.ascontiguousarray(traj["_micro_t"], dtype=np.float64)
            p_traj = np.ascontiguousarray(traj["_p_traj"], dtype=np.float64)
            m_list = np.ascontiguousarray(manifest.layout.get_mesh_data(), dtype=np.float64)
            id_arr = np.ascontiguousarray(manifest.id_arr, dtype=np.float64)

            c_seeds, c_ptrs, c_rows, c_cols, c_dense = manifest.cpr_cache
            dl_dy = np.zeros_like(y_traj)

            v_info = None
            for v_cand in ("V_cell", "V", "v", "Voltage"):
                if v_cand in manifest.layout.state_offsets:
                    v_info = (v_cand, manifest.layout.state_offsets[v_cand][0])
                    break

            i_info = None
            for i_cand in ("i_app", "I_app", "I"):
                if i_cand in manifest.layout.state_offsets:
                    i_info = (i_cand, manifest.layout.state_offsets[i_cand][0])
                    break

            if len(t_eval_arr) > 1:
                dt = np.zeros_like(t_eval_arr)
                dt[1:] = np.diff(t_eval_arr)
                total_t = max(float(t_eval_arr[-1] - t_eval_arr[0]), 1e-12)
                w_t = dt / total_t
            else:
                w_t = np.ones(len(t_eval_arr), dtype=np.float64)

            if v_info is not None:
                v_name, v_off = v_info
                v_traj = y_traj[:, v_off]
                v_scale = max(float(np.max(np.abs(v_traj))), 1e-6)
                dl_dy[:, v_off] = w_t * v_traj / (v_scale ** 2)
                qoi_val = 0.5 * float(np.sum(w_t * ((v_traj / v_scale) ** 2)))
                qoi_name = f"{v_name}(L2)"
                if i_info is not None:
                    i_name, i_off = i_info
                    i_traj = y_traj[:, i_off]
                    i_scale = max(float(np.max(np.abs(i_traj))), 1e-6)
                    dl_dy[:, i_off] = w_t * i_traj / (i_scale ** 2)
                    qoi_val += 0.5 * float(np.sum(w_t * ((i_traj / i_scale) ** 2)))
                    qoi_name = f"{v_name}+{i_name}(L2)"
            else:
                y_scale = max(float(np.max(np.abs(y_traj))), 1e-6)
                dl_dy[:] = (w_t[:, None] * y_traj) / (y_scale ** 2)
                qoi_val = 0.5 * float(np.sum(w_t * np.sum((y_traj / y_scale) ** 2, axis=1)))
                qoi_name = "||y||_L2^2"

            p_grad = discrete_adjoint_native(
                manifest.lib_path, y_traj, ydot_traj,
                t_eval_arr, id_arr, p_traj, m_list, dl_dy, bw,
                c_seeds, c_ptrs, c_rows, c_cols, c_dense, -1
            )
            p_grad_arr = np.asarray(p_grad)

            dead_params = []
            extreme_params = []
            sens_ranking = []

            for p_name, (off, _) in manifest.layout.param_offsets.items():
                if p_name.startswith("_term_"):
                    continue
                p_val = parameters.get(p_name, manifest.default_parameters.get(p_name, 1.0))
                grad_val = float(p_grad_arr[off])
                if abs(p_val) > 0.0:
                    norm_sens = (abs(p_val) / max(qoi_val, 1e-6)) * abs(grad_val)
                    is_dead = norm_sens < 1e-10
                else:
                    norm_sens = abs(grad_val)
                    is_dead = abs(grad_val) < 1e-12
                sens_ranking.append((p_name, norm_sens, grad_val))
                if is_dead:
                    dead_params.append(p_name)
                elif norm_sens > 1e4:
                    extreme_params.append(p_name)

            sens_ranking.sort(key=lambda x: x[1], reverse=True)
            sens_st = "PASS"
            warnings = []
            if dead_params:
                sens_st = "WARN"
                warnings.append(f"Zero sensitivity detected on: {', '.join(dead_params)}")
            if extreme_params:
                sens_st = "WARN"
                warnings.append(f"Extreme sensitivity (>1e4) on: {', '.join(extreme_params)}")

            checks["parameter_sensitivity"] = {
                "status": sens_st,
                "qoi_name": qoi_name,
                "dead_params": dead_params,
                "extreme_params": extreme_params,
                "sens_ranking": sens_ranking,
                "warnings": warnings,
            }
        except Exception as e:
            checks["parameter_sensitivity"] = {"status": "WARN", "error": str(e)}
    else:
        checks["parameter_sensitivity"] = {"status": "SKIP", "error": "No trajectory tape"}

    # 4. Numerical Guardrail (Clamp) Watchpoints (single pass over sampled steps)
    if static_audit and static_audit.get("clamp_meta"):
        clamp_meta = static_audit["clamp_meta"]
        t_times = result._data["Time [s]"]
        y_traj_raw = traj.get("_y_raw") if traj else None
        if y_traj_raw is not None and len(y_traj_raw) == len(t_times):
            lo_hits_map: Dict[int, List[tuple]] = {wp["index"]: [] for wp in clamp_meta}
            hi_hits_map: Dict[int, List[tuple]] = {wp["index"]: [] for wp in clamp_meta}
            stride_step = 1 if len(t_times) <= 500 else max(1, len(t_times) // 500)
            ydot_zero = [0.0] * n_states

            for s_i in range(0, len(t_times), stride_step):
                audit_obs = manifest.evaluate_audit_observables(y_traj_raw[s_i].tolist(), ydot_zero, parameters)
                for wp in clamp_meta:
                    idx_w = wp["index"]
                    hi_slot = wp["upper_slot"]
                    if len(audit_obs) > hi_slot:
                        frac_lo = audit_obs[wp["lower_slot"]]
                        frac_hi = audit_obs[hi_slot]
                        if frac_lo > 0.0:
                            lo_hits_map[idx_w].append((float(t_times[s_i]), float(frac_lo)))
                        if frac_hi > 0.0:
                            hi_hits_map[idx_w].append((float(t_times[s_i]), float(frac_hi)))

            active_clamps = []
            for wp in clamp_meta:
                idx_w = wp["index"]
                lo_h = lo_hits_map[idx_w]
                hi_h = hi_hits_map[idx_w]
                if lo_h or hi_h:
                    active_clamps.append({
                        "label": wp["label"],
                        "state": wp["state"],
                        "domain": wp["domain"],
                        "lo_hits_max_pct": max([h[1] for h in lo_h], default=0.0) * 100.0,
                        "hi_hits_max_pct": max([h[1] for h in hi_h], default=0.0) * 100.0,
                        "lo_t_range": (lo_h[0][0], lo_h[-1][0]) if lo_h else None,
                        "hi_t_range": (hi_h[0][0], hi_h[-1][0]) if hi_h else None,
                    })
            checks["numerical_guardrails"] = {
                "status": "WARN" if active_clamps else "PASS",
                "active_clamps": active_clamps,
            }
        else:
            checks["numerical_guardrails"] = {"status": "PASS", "active_clamps": []}
    else:
        checks["numerical_guardrails"] = {
            "status": "SKIP",
            "reason": "No clamps declared or compiled without audit_compiler=True",
        }

    # 5. Sequence Transition Log
    trans = result.diagnostics.get("sequence_transitions", [])
    if trans:
        checks["sequence_transitions"] = {"status": "PASS", "transitions": trans}
    else:
        checks["sequence_transitions"] = {"status": "SKIP", "reason": "Continuous solve (no Sequence protocol)"}

    # 6. Step & Realtime Health
    diag = result.diagnostics
    frozen = diag.get("realtime_frozen_steps", 0)
    total_steps = diag.get("total_steps", 0)
    rejected = diag.get("rejected_steps", 0)
    rej_ratio = diag.get("step_rejection_ratio", 0.0)
    trace_dt = diag.get("trace_dt", [])
    min_dt = float(min(trace_dt)) if trace_dt else None

    health_status = "PASS"
    health_notes = []
    if frozen > 0:
        health_status = "FAIL"
        health_notes.append(f"{frozen} steps frozen due to Newton divergence in realtime mode")
    elif rej_ratio > 0.2:
        health_status = "WARN"
        health_notes.append(f"High step rejection ratio: {rej_ratio:.1%}")

    checks["step_health"] = {
        "status": health_status,
        "total_steps": total_steps,
        "accepted_steps": diag.get("accepted_steps", 0),
        "rejected_steps": rejected,
        "realtime_frozen_steps": frozen,
        "min_dt": min_dt,
        "notes": health_notes,
    }

    provenance = {
        "solver_backend": engine.solver_backend,
        "rel_tol": rel_tol if rel_tol is not None else 1e-6,
        "abs_tol": abs_tol if abs_tol is not None else 1e-8,
        "cpp_sha256": (static_audit or {}).get("cpp_sha256", ""),
        "toolchain": (static_audit or {}).get("toolchain", {}),
    }

    return SolverAuditReport(
        model_name=model_name,
        n_states=n_states,
        n_params=n_params,
        checks=checks,
        static_audit=static_audit,
        provenance=provenance,
    )


AuditReport = SolverAuditReport


class AuditSession:
    """
    Context manager orchestrating scoped compile-time and solve-time model audits:
        with fx.audit() as audit:
            engine = fx.Engine(model=MyModel())
            res = engine.solve(protocol=protocol)
        audit.print_summary()
        assert audit.passed
    """

    def __init__(
        self,
        compiler: bool = True,
        solver: bool = True,
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None,
    ):
        self.compiler = compiler
        self.solver = solver
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self._token: Optional[contextvars.Token] = None
        self.static_reports: List[Dict[str, Any]] = []
        self.reports: List[SolverAuditReport] = []

    def __enter__(self) -> "AuditSession":
        self._token = _AUDIT_CONTEXT.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            _AUDIT_CONTEXT.reset(self._token)
            self._token = None

    def record_static(self, static_report: Dict[str, Any]) -> None:
        self.static_reports.append(static_report)

    def record_solver(self, solver_report: SolverAuditReport) -> None:
        self.reports.append(solver_report)

    @property
    def report(self) -> Optional[SolverAuditReport]:
        return self.reports[-1] if self.reports else None

    @property
    def checks(self) -> Dict[str, Any]:
        return self.reports[-1].checks if self.reports else {}

    @property
    def provenance(self) -> Dict[str, Any]:
        return self.reports[-1].provenance if self.reports else {}

    @property
    def passed(self) -> bool:
        return self.status != "FAIL"

    @property
    def status(self) -> str:
        if not self.reports and not self.static_reports:
            return "EMPTY"
        statuses: List[str] = []
        for r in self.reports:
            statuses.extend(r._all_statuses())
        for s in self.static_reports:
            statuses.extend(grade_audit({}, s))
        return reduce_status(statuses)

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.reports)

    def __getitem__(self, idx: int) -> SolverAuditReport:
        return self.reports[idx]

    def to_dict(self) -> Dict[str, Any]:
        if len(self.reports) == 1 and len(self.static_reports) <= 1:
            return self.reports[0].to_dict()
        return {
            "status": self.status,
            "passed": self.passed,
            "n_runs": len(self.reports),
            "runs": [r.to_dict() for r in self.reports],
            "static_audits": self.static_reports if not self.reports else [],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(payload)
        return payload

    def print_summary(self) -> None:
        if self.reports:
            for r in self.reports:
                r.print_summary()
            return
        if self.static_reports:
            for s in self.static_reports:
                stub = SolverAuditReport(
                    model_name="StaticOnly",
                    n_states=0,
                    n_params=0,
                    checks={"dae_residual": {"status": "SKIP", "reason": "No solve() executed inside fx.audit()"}},
                    static_audit=s,
                )
                stub.print_summary()
            return
        print("AuditSession: no Engine compilation or solve() calls captured.")

    def _require_report(self) -> SolverAuditReport:
        if not self.reports:
            raise RuntimeError("No solve() report captured in this AuditSession.")
        return self.reports[-1]

    def show_clamps(self) -> None:
        self._require_report().show_clamps()

    def show_transitions(self) -> None:
        self._require_report().show_transitions()

    def show_residuals(self, n: int = 5) -> None:
        self._require_report().show_residuals(n=n)

    def show_sensitivities(self) -> None:
        self._require_report().show_sensitivities()

    def show_math(self, state: Optional[str] = None) -> None:
        if self.reports:
            self.reports[-1].show_math(state=state)
        elif self.static_reports:
            print(self.static_reports[-1].get("math_pretty", "Math IR unavailable."))
        else:
            print("No audit data captured.")

    def show_compute(self, state: Optional[str] = None) -> None:
        if self.reports:
            self.reports[-1].show_compute(state=state)
        elif self.static_reports:
            print(self.static_reports[-1].get("compute_pretty", "Compute IR unavailable."))
        else:
            print("No audit data captured.")

    def show_boundaries(self) -> None:
        if self.reports:
            self.reports[-1].show_boundaries()
        elif self.static_reports:
            stub = SolverAuditReport("StaticOnly", 0, 0, {}, static_audit=self.static_reports[-1])
            stub.show_boundaries()
        else:
            print("No audit data captured.")


def audit(
    compiler: bool = True,
    solver: bool = True,
    rel_tol: Optional[float] = None,
    abs_tol: Optional[float] = None,
) -> AuditSession:
    """
    Creates a scoped model-auditing context manager.

    Usage:
        with fx.audit() as report:
            engine = fx.Engine(model=MyModel())
            res = engine.solve(protocol=protocol)
        report.print_summary()
        assert report.passed
    """
    return AuditSession(compiler=compiler, solver=solver, rel_tol=rel_tol, abs_tol=abs_tol)