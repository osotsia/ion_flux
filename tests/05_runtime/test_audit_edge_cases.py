import os
import platform
import shutil
import numpy as np
import pytest

import ion_flux as fx
from ion_flux.protocols import Sequence, CC, CV, Rest
from ion_flux.runtime.audit import SolverAuditReport
from ion_flux.runtime.manifest import ExecutableManifest


def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        os.path.exists("/opt/homebrew/opt/llvm/bin/clang++")
        or os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac


try:
    from ion_flux._core import solve_native  # noqa: F401
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_COMPILER = pytest.mark.skipif(
    not _has_compiler(), reason="Requires native C++ toolchain."
)
REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE,
    reason="Requires native C++ toolchain and compiled Rust backend.",
)


# ==============================================================================
# Edge Case 1: `missing_gauge_anchor` ignored by SolverAuditReport / AuditSession
# ==============================================================================

@REQUIRES_RUNTIME
def test_edge_case_1_missing_gauge_anchor_escalates_audit_status():
    """
    A spatial DAE (`div(-grad(phi)) == 0`) with compatible zero-flux Neumann BCs
    on both ends has NO `implicit_zero_flux_faces` and converges trivially at
    `phi = 0.0`, yet has a floating potential gauge (`missing_gauge_anchor=True`).

    Expected contract: `SolverAuditReport` and `AuditSession` must escalate
    `missing_gauge_anchor=True` to `"WARN"` or `"FAIL"`, whereas adding a
    Dirichlet anchor must restore `"PASS"`.
    """
    class CompatibleFloatingPotentialDAE(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=8, name="x")
        phi = fx.State(domain=x, name="phi")

        def math(self):
            i_e = -fx.grad(self.phi)
            return {
                "equations": {self.phi: fx.div(i_e) == 0.0},
                "boundaries": {i_e: {"left": 0.0, "right": 0.0}},
                "initial_conditions": {self.phi: 0.0},
            }

    class GroundedPotentialDAE(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=8, name="x")
        phi = fx.State(domain=x, name="phi")

        def math(self):
            i_e = -fx.grad(self.phi)
            return {
                "equations": {self.phi: fx.div(i_e) == 0.0},
                "boundaries": {
                    self.phi: {"left": fx.Dirichlet(0.0)},
                    i_e: {"right": 0.0},
                },
                "initial_conditions": {self.phi: 0.0},
            }

    # Control: grounded spatial DAE passes static grading cleanly
    with fx.audit(solver=False) as grounded_sess:
        eng_g = fx.Engine(model=GroundedPotentialDAE(), target="cpu:serial")
        static_g = eng_g.audit_compiler_report()

    phi_g = next(e for e in static_g["boundary_ledger"] if e["state"] == "phi")
    assert phi_g["missing_gauge_anchor"] is False
    assert grounded_sess.status == "PASS"

    # Floating DAE: unanchored gauge must escalate status
    with fx.audit(solver=False) as sess:
        eng = fx.Engine(model=CompatibleFloatingPotentialDAE(), target="cpu:serial")
        static = eng.audit_compiler_report()

    phi_entry = next(e for e in static["boundary_ledger"] if e["state"] == "phi")
    assert phi_entry["is_dae"] is True
    assert phi_entry["implicit_zero_flux_faces"] == []
    assert phi_entry["missing_gauge_anchor"] is True

    static_report = SolverAuditReport(
        model_name="CompatibleFloatingPotentialDAE",
        n_states=eng.layout.n_states,
        n_params=eng.layout.n_params,
        checks={},
        static_audit=static,
    )
    assert static_report.status in ("WARN", "FAIL"), (
        f"missing_gauge_anchor=True was silently graded as '{static_report.status}'."
    )
    assert sess.status in ("WARN", "FAIL"), (
        f"AuditSession graded missing_gauge_anchor=True as '{sess.status}'."
    )


# ==============================================================================
# Edge Case 2: Final-time QoI sensitivity collapses on CV / Rest protocols
# ==============================================================================

class ProtocolSensitivityCell(fx.PDE):
    """
    `D` governs solid diffusion; `R_int` is a purely ohmic series resistance in `V_cell`.
    Under `CC -> Rest`, `R_int` alters `V_cell` during `CC` without altering `c`,
    so `dV_cell(T_final)/dR_int == 0` when `Rest` ends with `i_app == 0`.
    Under `CC -> CV`, `V_cell(T_final)` is pinned to `v_target`, so both
    `dV_cell(T_final)/dD == 0` and `dV_cell(T_final)/dR_int == 0`.
    """
    x = fx.Domain(bounds=(0.0, 1.0), resolution=10, name="x")
    c = fx.State(domain=x, name="c")
    V_cell = fx.State(domain=None, name="V_cell")
    i_app = fx.State(domain=None, name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    D = fx.Parameter(default=0.5, name="D")
    R_int = fx.Parameter(default=0.1, name="R_int")

    def math(self):
        flux = -self.D * fx.grad(self.c, axis=self.x)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.x) - 0.05 * self.c,
                self.V_cell: self.V_cell == 3.5 + 0.5 * self.c.right - self.R_int * self.i_app,
            },
            "boundaries": {
                flux: {"left": 0.0, "right": 0.2 * self.i_app},
            },
            "initial_conditions": {
                self.c: 2.0,
                self.V_cell: 4.5,
                self.i_app: 0.0,
            },
        }


@REQUIRES_RUNTIME
@pytest.mark.parametrize("steps_factory", [
    lambda m: [CC(rate=1.0, time=0.5), Rest(time=0.5)],
    lambda m: [CC(rate=1.0, until=m.V_cell <= 4.35, time=2.0), CV(voltage=4.35, time=0.5)],
])
def test_edge_case_2_cv_and_rest_protocols_avoid_zero_sensitivity_collapse(steps_factory):
    """
    Both `D` and `R_int` significantly impact the full protocol trajectory (verified
    directly via forward perturbations). An audit sensitivity scan must report
    meaningful normalized sensitivity (`norm_sens > 1e-6`, not just `>= 1e-15` LU
    roundoff noise) and `dead_params == []` regardless of whether the sequence ends
    in `Rest` or `CV`.
    """
    model = ProtocolSensitivityCell()
    protocol = Sequence(steps_factory(model))

    # 1. Ground truth: both parameters visibly alter the trajectory
    eng_ref = fx.Engine(model=model, target="cpu:serial")
    res_nom = eng_ref.solve(protocol=protocol, show_progress=False)
    res_d = eng_ref.solve(protocol=protocol, parameters={"D": 0.25}, show_progress=False)
    res_r = eng_ref.solve(protocol=protocol, parameters={"R_int": 0.2}, show_progress=False)

    d_impact = max(
        float(np.max(np.abs(res_d["V_cell"].data - res_nom["V_cell"].data))),
        float(np.max(np.abs(res_d["i_app"].data - res_nom["i_app"].data))),
    )
    r_impact = max(
        float(np.max(np.abs(res_r["V_cell"].data - res_nom["V_cell"].data))),
        float(np.max(np.abs(res_r["i_app"].data - res_nom["i_app"].data))),
    )
    assert d_impact > 1e-3
    assert r_impact > 1e-2

    # 2. Audit sensitivity scan must reflect non-negligible normalized sensitivities
    with fx.audit() as sess:
        eng = fx.Engine(model=model, target="cpu:serial")
        eng.solve(protocol=protocol, show_progress=False)

    sens = sess.checks["parameter_sensitivity"]
    assert sens["dead_params"] == [], (
        f"Active parameters falsely flagged as dead: {sens['dead_params']}"
    )
    assert sens["status"] == "PASS"

    ranking = {name: (norm_s, raw_g) for name, norm_s, raw_g in sens["sens_ranking"]}
    for p_name in ("D", "R_int"):
        norm_s, raw_g = ranking[p_name]
        assert norm_s > 1e-6, (
            f"Sensitivity collapse for '{p_name}' (norm_sens={norm_s:.3e}, raw_grad={raw_g:.3e}): "
            "audit QoI only reflects T_final rather than the full trajectory."
        )


# ==============================================================================
# Edge Case 3: `extract_domain_name` omits `MathClamp` on `fx.clamp(...).right`
# ==============================================================================

@REQUIRES_RUNTIME
def test_edge_case_3a_1d_clamp_before_boundary_ref_preserves_domain():
    """
    Taking `.right` on a clamped 1D state (`fx.clamp(self.c, ...).right`) without an
    explicit `domain=` relies on `extract_domain_name(MathClamp)`. Without a
    `MathClamp` branch, the index silently falls back to `0` (`c.left == 1.0`).
    """
    class Clamped1DBoundaryRefModel(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=5, name="x")
        c = fx.State(domain=x, name="c")
        v = fx.State(domain=None, name="v")

        def math(self):
            c_safe = fx.clamp(self.c, lower=0.1, upper=10.0, label="c_safe")
            return {
                "equations": {
                    self.c: fx.dt(self.c) == 0.0,
                    self.v: self.v == c_safe.right,
                },
                "boundaries": {},
                "initial_conditions": {
                    self.c: 1.0 + 4.0 * self.x.coords,  # left=1.0, right=5.0
                    self.v: 5.0,
                },
            }

    eng = fx.Engine(model=Clamped1DBoundaryRefModel(), target="cpu:serial")
    y0, ydot0 = eng.manifest.evaluate_ic({})
    res_t0 = eng.evaluate_residual(y0, ydot0)
    off_v = eng.layout.get_state_offset("v")

    assert res_t0[off_v] == pytest.approx(0.0, abs=1e-12), (
        f"c_safe.right evaluated wrong spatial index at t=0 (residual={res_t0[off_v]})."
    )

    sim = eng.solve(t_span=(0.0, 0.1), show_progress=False)
    assert sim["v"].data[-1] == pytest.approx(5.0, rel=1e-6), (
        f"Expected c_safe.right == 5.0, got {sim['v'].data[-1]}."
    )


@REQUIRES_RUNTIME
def test_edge_case_3b_composite_clamp_before_boundary_ref_preserves_inner_axis():
    """
    Taking `.right` on a clamped 2D composite state (`fx.clamp(c_s, ...).right` on
    `x * r`) inside a 1D macro equation on `x` must pin `r` to `res_r - 1` at every `x`.
    """
    class ClampedCompositeBoundaryRefModel(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=3, name="x")
        r = fx.Domain(bounds=(0.0, 1.0), resolution=4, coord_sys="spherical", name="r")
        c_s = fx.State(domain=x * r, name="c_s")
        v = fx.State(domain=x, name="v")

        def math(self):
            c_s_safe = fx.clamp(self.c_s, lower=0.1, upper=20.0, label="cs_safe")
            return {
                "equations": {
                    self.c_s: fx.dt(self.c_s) == 0.0,
                    self.v: self.v == c_s_safe.right,
                },
                "boundaries": {},
                "initial_conditions": {
                    self.c_s: 1.0 + 9.0 * self.r.coords,  # r=0 -> 1.0, r=1 -> 10.0
                    self.v: 10.0,
                },
            }

    eng = fx.Engine(model=ClampedCompositeBoundaryRefModel(), target="cpu:serial")
    y0, ydot0 = eng.manifest.evaluate_ic({})
    res_t0 = eng.evaluate_residual(y0, ydot0)
    off_v, size_v = eng.layout.state_offsets["v"]
    np.testing.assert_allclose(
        res_t0[off_v : off_v + size_v],
        0.0,
        atol=1e-12,
        err_msg="Composite c_s_safe.right evaluated wrong radial index at t=0.",
    )

    sim = eng.solve(t_span=(0.0, 0.1), show_progress=False)
    np.testing.assert_allclose(sim["v"].data[-1], 10.0, rtol=1e-6)


# ==============================================================================
# Edge Case 4: `Engine.load()`, `.so` metadata, and warm JIT cache audit parity
# ==============================================================================

class _SimpleAuditedODE(fx.PDE):
    y = fx.State(domain=None, name="y")
    k = fx.Parameter(default=1.0, name="k")

    def math(self):
        return {
            "equations": {self.y: fx.dt(self.y) == -self.k * self.y},
            "boundaries": {},
            "initial_conditions": {self.y: 1.0},
        }


@REQUIRES_RUNTIME
def test_edge_case_4a_engine_load_registers_static_audit_in_session(tmp_path):
    """
    Loading a compiled `.so` via `Engine.load()` inside `with fx.audit() as sess:`
    must record the static audit into `AuditSession.static_reports` consistently
    with `Engine(model=...)`.
    """
    so_path = tmp_path / "simple_audited.so"
    eng_build = fx.Engine(model=_SimpleAuditedODE(), target="cpu:serial", audit_compiler=True)
    eng_build.export_binary(str(so_path))

    with fx.audit() as sess:
        loaded = fx.Engine.load(str(so_path), target="cpu:serial")
        assert len(sess.static_reports) == 1, (
            "Engine.load() inside `with fx.audit():` did not record a static audit report."
        )
        assert sess.static_reports[0]["cpr_jvp_probe_passed"] is True
        assert sess.static_reports[0]["ic_health"]["ic_converged"] is True
        loaded.solve(t_span=(0.0, 0.5), show_progress=False)


@REQUIRES_RUNTIME
def test_edge_case_4b_exported_so_persists_native_static_audit_metadata(tmp_path):
    """
    Exported `.so` embedded metadata must already persist `cpr_jvp_probe_passed` and
    `ic_health` from `audit_compiler=True` without requiring a post-hoc
    `audit_compiler_report()` call.
    """
    so_path = tmp_path / "simple_meta.so"
    eng = fx.Engine(model=_SimpleAuditedODE(), target="cpu:serial", audit_compiler=True)
    eng.export_binary(str(so_path))

    manifest_loaded = ExecutableManifest.load(str(so_path))
    assert manifest_loaded.static_audit is not None
    assert "cpr_jvp_probe_passed" in manifest_loaded.static_audit, (
        "Exported .so embedded metadata dropped `cpr_jvp_probe_passed`."
    )
    assert "ic_health" in manifest_loaded.static_audit, (
        "Exported .so embedded metadata dropped `ic_health`."
    )


@REQUIRES_RUNTIME
def test_edge_case_4c_warm_jit_cache_hit_preserves_static_audit_and_session():
    """
    Cold and warm (`cache=True`) `Engine(model=...)` instantiations inside
    `with fx.audit():` must record identical static audit fields.
    """
    with fx.audit() as sess_cold:
        eng_cold = fx.Engine(model=_SimpleAuditedODE(), target="cpu:serial", cache=True)
        res_cold = eng_cold.solve(t_span=(0.0, 0.2), show_progress=False)

    with fx.audit() as sess_warm:
        eng_warm = fx.Engine(model=_SimpleAuditedODE(), target="cpu:serial", cache=True)
        res_warm = eng_warm.solve(t_span=(0.0, 0.2), show_progress=False)

    assert eng_cold.manifest.lib_path == eng_warm.manifest.lib_path
    assert len(sess_warm.static_reports) == 1
    assert sess_warm.static_reports[0]["cpr_jvp_probe_passed"] == sess_cold.static_reports[0]["cpr_jvp_probe_passed"]
    assert sess_warm.static_reports[0]["ic_health"] == sess_cold.static_reports[0]["ic_health"]
    assert res_warm.audit.status == res_cold.audit.status


@REQUIRES_RUNTIME
def test_edge_case_4d_engine_load_non_audited_binary_zero_overhead_and_opt_in(tmp_path):
    """
    A production binary compiled with `audit_compiler=False` stays zero-overhead on
    `Engine.load()` outside `fx.audit()`, yet runs native CPR/IC probes when loaded
    inside `with fx.audit():`.
    """
    so_path = tmp_path / "prod_only.so"
    eng_prod = fx.Engine(model=_SimpleAuditedODE(), target="cpu:serial", audit_compiler=False)
    eng_prod.export_binary(str(so_path))

    loaded_prod = fx.Engine.load(str(so_path), target="cpu:serial")
    assert loaded_prod.audit_compiler is False
    assert loaded_prod.manifest.static_audit is None

    with fx.audit() as sess:
        loaded_audited = fx.Engine.load(str(so_path), target="cpu:serial")
        assert loaded_audited.audit_compiler is True
        assert len(sess.static_reports) == 1
        assert sess.static_reports[0]["cpr_jvp_probe_passed"] is True
        assert sess.static_reports[0]["ic_health"]["ic_converged"] is True
        res = loaded_audited.solve(t_span=(0.0, 0.5), show_progress=False)

    assert res.audit is not None
    assert res.audit.passed is True


# ==============================================================================
# Edge Case 5: Unstructured `boundary_ledger` omits `Dirichlet` faces
# ==============================================================================

@REQUIRES_COMPILER
def test_edge_case_5_unstructured_boundary_ledger_records_dirichlet_faces():
    """
    In `Compiler._build_boundary_ledger`, the `coord_sys == 'unstructured'` branch
    checks `not dirichlet` for `missing_gauge_anchor`, yet only populates `faces`
    from `neumann_bcs` and silently omits `dirichlet_bcs`.
    """
    tet = {
        "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "elements": [[0, 1, 2, 3]],
    }

    class UnstructuredMixedBC(fx.PDE):
        mesh = fx.Domain.from_mesh(tet, name="mesh", surfaces={"left": [0], "top": [2, 3]})
        c = fx.State(domain=mesh, name="c")

        def math(self):
            flux = -fx.grad(self.c)
            return {
                "equations": {self.c: fx.dt(self.c) == -fx.div(flux)},
                "boundaries": {
                    self.c: {"left": fx.Dirichlet(0.0)},
                    flux: {"top": -1.0},
                },
                "initial_conditions": {self.c: 1.0},
            }

    eng = fx.Engine(model=UnstructuredMixedBC(), target="cpu:serial", audit_compiler=True)
    static = eng.audit_compiler_report()
    c_entry = next(e for e in static["boundary_ledger"] if e["state"] == "c")

    assert c_entry["missing_gauge_anchor"] is False
    assert c_entry["faces"].get("top") == "Neumann"
    assert c_entry["faces"].get("left") == "Dirichlet", (
        f"Unstructured boundary_ledger omitted Dirichlet surface 'left': {c_entry['faces']}"
    )
    


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])