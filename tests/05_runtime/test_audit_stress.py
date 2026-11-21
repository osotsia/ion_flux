import json
import os
import platform
import shutil
import sys
import numpy as np
import pytest

import ion_flux as fx
from ion_flux.compiler._2_middle_end.verification import TopologicalError
from ion_flux.protocols import Sequence, CC, CV, Rest
from ion_flux.runtime.audit import get_active_audit, run_solver_audit
from ion_flux.runtime.manifest import run_native_static_audit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models")))


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

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE,
    reason="Requires native C++ toolchain and compiled Rust backend.",
)


# ==============================================================================
# Shared Well-Posed Fixtures / Models
# ==============================================================================

class CleanAuditedBatteryCell(fx.PDE):
    """Well-posed 1D PDE + 0D DAE + Observable + active clamp (never saturates)."""
    x = fx.Domain(bounds=(0.0, 1.0), resolution=10, name="x")
    c = fx.State(domain=x, name="c")
    V_cell = fx.State(domain=None, name="V_cell")
    i_app = fx.State(domain=None, name="i_app")
    c_surf_obs = fx.Observable(domain=None, name="c_surf_obs")

    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    D = fx.Parameter(default=0.5, name="D")
    R_int = fx.Parameter(default=0.1, name="R_int")

    def math(self):
        c_safe = fx.clamp(self.c, lower=0.05, upper=5.0, label="c_bounds")
        flux = -self.D * fx.grad(self.c, axis=self.x)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.x) - 0.05 * c_safe,
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
            "observables": {
                self.c_surf_obs: self.c.right,
            },
        }


# ==============================================================================
# SECTION 1: Happy-Path Audit Tests
# ==============================================================================

@REQUIRES_RUNTIME
def test_happy_path_scoped_audit_all_pass(tmp_path, capsys):
    """
    Happy Path 1: Full compile + solve audit on a well-posed multi-stage CC protocol.
    Note: Ends on CC (nonzero i_app) so both D and R_int remain sensitive at T_final.
    """
    model = CleanAuditedBatteryCell()
    protocol = Sequence([
        CC(rate=1.0, until=model.V_cell <= 4.35, time=2.0),
        CC(rate=0.5, time=0.5),
    ])

    with fx.audit() as session:
        engine = fx.Engine(model=model, target="cpu:serial")
        res = engine.solve(protocol=protocol, show_progress=False)

    assert session.passed is True
    assert session.status == "PASS"
    assert len(session) == 1
    assert res.audit is session.report

    # Static integrity checks
    static = res.audit.static_audit
    assert static["cpr_jvp_probe_passed"] is True
    assert static["cpr_jvp_probe_max_diff"] < 1e-8
    assert static["ic_health"]["ic_converged"] is True
    assert static["ic_health"]["ic_final_max_res"] < 1e-8
    assert static["cpp_sha256"]
    assert static["math_pretty"]
    assert static["compute_pretty"]

    c_ledger = next(e for e in static["boundary_ledger"] if e["state"] == "c")
    assert c_ledger["faces"] == {"left": "Neumann", "right": "Neumann"}
    assert c_ledger["implicit_zero_flux_faces"] == []

    # Solver checks
    checks = res.audit.checks
    assert checks["dae_residual"]["status"] == "PASS"
    assert checks["dae_residual"]["max_wrms_residual"] <= 2.0
    assert checks["sundials_cross_check"]["status"] == "PASS"
    assert checks["sundials_cross_check"]["rmse_mv"] < 1.0
    assert checks["numerical_guardrails"]["status"] == "PASS"
    assert checks["numerical_guardrails"]["active_clamps"] == []
    assert checks["parameter_sensitivity"]["status"] == "PASS"
    assert checks["parameter_sensitivity"]["dead_params"] == []
    assert checks["step_health"]["status"] == "PASS"
    assert checks["step_health"]["realtime_frozen_steps"] == 0
    assert len(checks["sequence_transitions"]["transitions"]) == 2

    # Drill-downs & JSON serialization
    session.print_summary()
    session.show_math(state="c")
    session.show_compute(state="c")
    session.show_boundaries()
    session.show_residuals(n=3)
    session.show_sensitivities()
    session.show_transitions()
    out = capsys.readouterr().out
    assert "OVERALL STATUS: PASS" in out

    json_path = tmp_path / "clean_audit.json"
    parsed = json.loads(session.to_json(str(json_path)))
    assert json_path.exists()
    assert parsed["status"] == "PASS"
    assert parsed["passed"] is True


@REQUIRES_RUNTIME
def test_happy_path_standalone_engine_audit_and_context_isolation():
    """
    Happy Path 2: Validates `engine.audit()` standalone path, multi-run `AuditSession`,
    and strict restoration of zero-overhead default upon context exit (even on exception).
    """
    class SubParticle(fx.PDE):
        r = fx.Domain(bounds=(0.0, 1.0), resolution=6, coord_sys="spherical", name="r")
        c = fx.State(domain=r, name="c")
        D = fx.Parameter(default=0.2, name="D")

        def math(self, flux_bc):
            flux = -self.D * fx.grad(self.c, axis=self.r)
            return {
                "equations": {self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r)},
                "boundaries": {flux: {"left": 0.0, "right": flux_bc}},
                "initial_conditions": {self.c: 1.0},
            }

    class ModularCell(fx.PDE):
        anode = SubParticle()
        V_cell = fx.State(domain=None, name="V_cell")
        i_app = fx.State(domain=None, name="i_app")
        terminal = fx.Terminal(current=i_app, voltage=V_cell)

        def math(self):
            macro = {
                "equations": {self.V_cell: self.V_cell == 3.0 + self.anode.c.right - 0.05 * self.i_app},
                "boundaries": {},
                "initial_conditions": {self.V_cell: 4.0, self.i_app: 0.0},
            }
            return fx.merge(macro, self.anode.math(0.05 * self.i_app))

    # 1. Explicit standalone audit without context manager
    eng = fx.Engine(model=ModularCell(), target="cpu:serial", audit_compiler=True)
    rep = eng.audit(protocol=Sequence([CC(rate=1.0, time=1.0)]))
    assert rep.passed is True
    assert rep.status == "PASS"
    assert "anode_D" in {name for name, _, _ in rep.checks["parameter_sensitivity"]["sens_ranking"]}

    # 2. Multi-run AuditSession aggregation
    with fx.audit() as sess:
        eng_m = fx.Engine(model=ModularCell(), target="cpu:serial")
        eng_m.solve(protocol=Sequence([CC(rate=0.5, time=0.5)]), show_progress=False)
        eng_m.solve(protocol=Sequence([CC(rate=1.0, time=0.5)]), show_progress=False)

    assert len(sess) == 2
    assert sess.passed is True
    assert sess.to_dict()["n_runs"] == 2

    # 3. Exception safety: ContextVar must reset on unhandled exception
    with pytest.raises(RuntimeError, match="forced"):
        with fx.audit():
            assert get_active_audit() is not None
            raise RuntimeError("forced")
    assert get_active_audit() is None

    # 4. Zero-overhead restoration outside `with fx.audit():`
    eng_clean = fx.Engine(model=ModularCell(), target="cpu:serial")
    assert eng_clean.audit_compiler is False
    res_clean = eng_clean.solve(protocol=Sequence([CC(rate=1.0, time=0.5)]), show_progress=False)
    assert res_clean.audit is None


@REQUIRES_RUNTIME
def test_happy_path_unstructured_gmres_skips_sundials_cleanly():
    """
    Happy Path 3: Unstructured 3D tetrahedral mesh (bandwidth == -1) must pass
    residuals and sensitivities while explicitly skipping SUNDIALS differential cross-check.
    """
    tet = {
        "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "elements": [[0, 1, 2, 3]],
    }

    class UnstructuredAudited(fx.PDE):
        mesh = fx.Domain.from_mesh(tet, name="mesh", surfaces={"top": [2, 3]})
        c = fx.State(domain=mesh, name="c")
        D = fx.Parameter(default=1.0, name="D")

        def math(self):
            flux = -self.D * fx.grad(self.c)
            return {
                "equations": {self.c: fx.dt(self.c) == -fx.div(flux)},
                "boundaries": {flux: {"top": -2.0}},
                "initial_conditions": {self.c: 1.0},
            }

    with fx.audit() as sess:
        eng = fx.Engine(model=UnstructuredAudited(), target="cpu:serial")
        eng.solve(t_span=(0.0, 0.5), show_progress=False)

    assert eng.jacobian_bandwidth == -1
    assert sess.passed is True
    assert sess.checks["sundials_cross_check"]["status"] == "SKIP"
    assert sess.checks["dae_residual"]["status"] == "PASS"
    assert sess.checks["parameter_sensitivity"]["status"] == "PASS"


# ==============================================================================
# SECTION 2: Failure Category 1 — Static Topology, Rank & Broadcast Errors
# ==============================================================================

def test_failure_cat1_unreduced_1d_to_0d_broadcast_rejected():
    """Cat 1a: Referencing a spatial field inside a 0D equation without reduction must fail statically."""
    class BadBroadcast(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=5, name="x")
        c = fx.State(domain=x, name="c")
        V = fx.State(domain=None, name="V")

        def math(self):
            return {
                "equations": {
                    self.c: fx.dt(self.c) == 0.0,
                    self.V: self.V == self.c,  # Unreduced 1D -> 0D
                },
                "boundaries": {},
                "initial_conditions": {self.c: 1.0, self.V: 1.0},
            }

    with pytest.raises(ValueError, match="Dimensional Broadcast Error"):
        with fx.audit():
            fx.Engine(model=BadBroadcast(), target="cpu:serial", mock_execution=True)


def test_failure_cat1_math_leak_shared_flux_rejected():
    """Cat 1b: Sharing one Neumann-bound flux AST node across two equations must fail statically."""
    class SharedFluxLeak(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=5, name="x")
        c1 = fx.State(domain=x, name="c1")
        c2 = fx.State(domain=x, name="c2")

        def math(self):
            shared = -fx.grad(self.c1)
            return {
                "equations": {
                    self.c1: fx.dt(self.c1) == -fx.div(shared),
                    self.c2: fx.dt(self.c2) == -fx.div(shared),
                },
                "boundaries": {shared: {"left": 0.0, "right": 1.0}},
                "initial_conditions": {self.c1: 0.0, self.c2: 0.0},
            }

    with pytest.raises(ValueError, match="Math Leak Detected"):
        with fx.audit():
            fx.Engine(model=SharedFluxLeak(), target="cpu:serial", mock_execution=True)


def test_failure_cat1_open_manifold_and_unconstrained_state_rejected():
    """Cat 1c: Spatial PDE missing all BCs and declared states missing equations must fail statically."""
    class OpenManifold(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=5, name="x")
        c = fx.State(domain=x, name="c")

        def math(self):
            return {
                "equations": {self.c: fx.dt(self.c) == -fx.div(-fx.grad(self.c))},
                "boundaries": {},
                "initial_conditions": {self.c: 1.0},
            }

    with pytest.raises(TopologicalError, match="Missing Boundary Conditions"):
        fx.Engine(model=OpenManifold(), target="cpu:serial", mock_execution=True)

    class RankDeficient(fx.PDE):
        a = fx.State(domain=None, name="a")
        b = fx.State(domain=None, name="b")

        def math(self):
            return {"equations": {self.a: fx.dt(self.a) == -self.a}}

    with pytest.raises(ValueError, match="Unconstrained state detected"):
        fx.Engine(model=RankDeficient(), target="cpu:serial", mock_execution=True)


# ==============================================================================
# SECTION 3: Failure Category 2 — Boundary Completeness & Gauge Singularities
# ==============================================================================

@REQUIRES_RUNTIME
def test_failure_cat2_implicit_zero_flux_fallback_flagged():
    """
    Cat 2a: Specifying only one face of a 1D PDE triggers ImplicitZeroFlux on the
    unbound face, forcing boundary ledger detection and audit status == 'WARN'.
    """
    class OneSidedBCModel(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=8, name="x")
        c = fx.State(domain=x, name="c")
        D = fx.Parameter(default=1.0, name="D")

        def math(self):
            flux = -self.D * fx.grad(self.c)
            return {
                "equations": {self.c: fx.dt(self.c) == -fx.div(flux)},
                "boundaries": {flux: {"left": 1.0}},  # right face omitted
                "initial_conditions": {self.c: 1.0},
            }

    with fx.audit() as sess:
        eng = fx.Engine(model=OneSidedBCModel(), target="cpu:serial")
        eng.solve(t_span=(0.0, 0.5), show_progress=False)

    ledger = sess.report.static_audit["boundary_ledger"]
    c_entry = next(e for e in ledger if e["state"] == "c")
    assert c_entry["faces"]["left"] == "Neumann"
    assert c_entry["faces"]["right"] == "ImplicitZeroFlux"
    assert "right" in c_entry["implicit_zero_flux_faces"]
    assert sess.status == "WARN"


@REQUIRES_RUNTIME
def test_failure_cat2_floating_spatial_dae_gauge_singularity():
    """
    Cat 2b: A pure spatial DAE (`div(-grad(phi)) == 0`) with only Neumann BCs lacks
    a Dirichlet reference ground, making the Jacobian singular (missing_gauge_anchor).
    """
    class FloatingPotentialDAE(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=8, name="x")
        phi = fx.State(domain=x, name="phi")

        def math(self):
            i_e = -fx.grad(self.phi)
            return {
                "equations": {self.phi: fx.div(i_e) == 0.0},
                "boundaries": {i_e: {"left": 0.0, "right": 1.0}},  # Incompatible Neumann BCs, no Dirichlet
                "initial_conditions": {self.phi: 0.0},
            }

    with fx.audit() as sess:
        eng = fx.Engine(model=FloatingPotentialDAE(), target="cpu:serial")
        static = eng.audit_compiler_report()

    phi_entry = next(e for e in static["boundary_ledger"] if e["state"] == "phi")
    assert phi_entry["is_dae"] is True
    assert phi_entry["missing_gauge_anchor"] is True
    # Incompatible Neumann BCs with no ground must fail IC convergence
    assert static["ic_health"]["ic_converged"] is False
    assert sess.passed is False
    assert sess.status == "FAIL"


# ==============================================================================
# SECTION 4: Failure Category 3 — t=0 Initialization & Matrix Conditioning
# ==============================================================================

@REQUIRES_RUNTIME
def test_failure_cat3_unsolvable_ic_paradox_fails_audit_and_crashes_cleanly():
    """
    Cat 3a: An unsolvable algebraic constraint (`x == x + 1`) must mark `ic_converged=False`,
    set `audit.passed == False` / `audit.status == 'FAIL'`, and raise NativeSolverCrash on solve.
    """
    class UnsolvableIC(fx.PDE):
        x = fx.State(domain=None, name="x")

        def math(self):
            return {
                "equations": {self.x: self.x == self.x + 1.0},
                "boundaries": {},
                "initial_conditions": {self.x: 0.0},
            }

    with fx.audit() as sess:
        eng = fx.Engine(model=UnsolvableIC(), target="cpu:serial")
        static = eng.audit_compiler_report()

        assert static["ic_health"]["ic_converged"] is False
        assert static["ic_health"]["ic_final_max_res"] >= 0.99
        assert sess.passed is False
        assert sess.status == "FAIL"

        with pytest.raises(RuntimeError, match="NATIVE SOLVER CRASH"):
            eng.solve(t_span=(0.0, 1.0), show_progress=False)

    assert get_active_audit() is None


@REQUIRES_RUNTIME
def test_failure_cat3_large_t0_algebraic_snap_tracked_and_conditioning_warned():
    """
    Cat 3b & 3c:
      - Intentional IC mismatch (`V = 0.0` when true algebraic root is `4.0`) must surface
        pre_snap_alg_max_res ~= 4.0 and max_snap_state == 'V'.
      - Jacobian coefficients spanning > 1e20 (1e12 vs 1e-10) must trigger conditioning WARN.
    """
    class SnapAndIllConditioned(fx.PDE):
        a = fx.State(domain=None, name="a")
        b = fx.State(domain=None, name="b")
        V = fx.State(domain=None, name="V")
        k = fx.Parameter(default=1.0, name="k")

        def math(self):
            return {
                "equations": {
                    self.a: 1e12 * self.a == 1e12 * self.k,
                    self.b: 1e-10 * self.b == 1e-10 * self.k,
                    self.V: self.V == 4.0 * self.k,
                },
                "boundaries": {},
                "initial_conditions": {self.a: 1.0, self.b: 1.0, self.V: 0.0},
            }

    with fx.audit() as sess:
        eng = fx.Engine(model=SnapAndIllConditioned(), target="cpu:serial")
        eng.solve(t_span=(0.0, 1.0), show_progress=False)

    ic = sess.report.static_audit["ic_health"]
    assert ic["ic_converged"] is True
    assert ic["pre_snap_alg_max_res"] == pytest.approx(4.0, rel=1e-6)
    assert ic["ic_final_max_res"] < 1e-8
    assert ic["max_snap_state"] == "V"
    assert ic["max_snap_delta"] == pytest.approx(4.0, rel=1e-4)
    assert ic["init_jac_max"] / ic["init_jac_min"] > 1e20
    assert sess.status == "WARN"


# ==============================================================================
# SECTION 5: Failure Category 4 — Numerical Guardrail (Clamp) Saturation
# ==============================================================================

@REQUIRES_RUNTIME
def test_failure_cat4_clamp_saturation_detected_without_corrupting_user_obs():
    """
    Cat 4: Explicit `fx.clamp(..., label=...)` saturation must trigger `WARN` in
    `numerical_guardrails`, report node percentages & time ranges, and keep `obs_offsets` isolated.
    """
    class ClampSaturatingPDE(fx.PDE):
        x = fx.Domain(bounds=(0.0, 1.0), resolution=6, name="x")
        c = fx.State(domain=x, name="c")
        V = fx.State(domain=None, name="V")
        i_app = fx.State(domain=None, name="i_app")
        c_obs = fx.Observable(domain=None, name="c_obs")

        terminal = fx.Terminal(current=i_app, voltage=V)
        D = fx.Parameter(default=0.2, name="D")

        def math(self):
            c_safe = fx.clamp(self.c, lower=0.5, upper=1.5, label="ce_floor")
            flux = -self.D * fx.grad(self.c, axis=self.x)
            return {
                "equations": {
                    self.c: fx.dt(self.c) == -fx.div(flux, axis=self.x) - 0.8 * c_safe,
                    self.V: self.V == 2.0 + self.c.right - 0.05 * self.i_app,
                },
                "boundaries": {flux: {"left": 0.0, "right": 0.1 * self.i_app}},
                "initial_conditions": {self.c: 1.0, self.V: 3.0, self.i_app: 0.0},
                "observables": {self.c_obs: self.c.right},
            }

    protocol = Sequence([
        CC(rate=2.0, until=ClampSaturatingPDE.V <= 2.35, time=3.0),
        CC(rate=0.5, time=0.5),
    ])

    with fx.audit() as sess:
        model = ClampSaturatingPDE()
        eng = fx.Engine(model=model, target="cpu:serial")
        res = eng.solve(
            protocol=Sequence([CC(rate=2.0, until=model.V <= 2.35, time=3.0), CC(rate=0.5, time=0.5)]),
            show_progress=False,
        )

    assert eng.layout.n_obs == 1  # Audit clamp watchpoints must not pollute user observables
    assert res["c_surf_obs" if "c_surf_obs" in res._data else "c_obs"].data.shape == (len(res["Time [s]"].data),)

    guard = sess.checks["numerical_guardrails"]
    assert guard["status"] == "WARN"
    assert len(guard["active_clamps"]) == 1
    c_info = guard["active_clamps"][0]
    assert c_info["label"] == "ce_floor"
    assert c_info["lo_hits_max_pct"] > 0.0
    assert c_info["lo_t_range"] is not None
    assert sess.status == "WARN"
    assert sess.passed is True


# ==============================================================================
# SECTION 6: Failure Category 5 — Parameter Sensitivity Pathologies
# ==============================================================================

@REQUIRES_RUNTIME
def test_failure_cat5_dead_and_extreme_sensitivity_parameters():
    """
    Cat 5: All-parameter VJP scan must detect both disconnected (`0.0`-sensitivity)
    parameters and extreme-gain (`norm_sens > 1e4`) parameters.
    """
    class SensitivityPathology(fx.PDE):
        y = fx.State(domain=None, name="y")
        V_cell = fx.State(domain=None, name="V_cell")
        k_normal = fx.Parameter(default=1.0, name="k_normal")
        k_dead = fx.Parameter(default=5.0, name="k_dead")
        k_extreme = fx.Parameter(default=1.0, name="k_extreme")

        def math(self):
            return {
                "equations": {
                    self.y: fx.dt(self.y) == -self.k_normal * self.y,
                    self.V_cell: self.V_cell == self.y + 1e5 * (self.k_extreme - 1.0),
                },
                "boundaries": {},
                "initial_conditions": {self.y: 1.0, self.V_cell: 1.0},
            }

    with fx.audit() as sess:
        eng = fx.Engine(model=SensitivityPathology(), target="cpu:serial")
        eng.solve(t_span=(0.0, 1.0), show_progress=False)

    sens = sess.checks["parameter_sensitivity"]
    assert sens["status"] == "WARN"
    assert "k_dead" in sens["dead_params"]
    assert "k_extreme" in sens["extreme_params"]
    assert "k_normal" not in sens["dead_params"]
    assert "k_normal" not in sens["extreme_params"]
    assert sess.status == "WARN"


# ==============================================================================
# SECTION 7: Failure Category 6 — Solver Divergence & Realtime Step Freezing
# ==============================================================================

@REQUIRES_RUNTIME
def test_failure_cat6_hard_solver_divergence_reports_offenders():
    """
    Cat 6a: Stiff nonlinear model starved of Newton iterations (`max_newton_iters=1`)
    must crash deterministically with offender diagnostics rather than returning garbage.
    """
    class StiffCubicODE(fx.PDE):
        y = fx.State(domain=None, name="y")

        def math(self):
            return {
                "equations": {self.y: fx.dt(self.y) == -100.0 * (self.y ** 3)},
                "boundaries": {},
                "initial_conditions": {self.y: 5.0},
            }

    with pytest.raises(RuntimeError, match="NATIVE SOLVER CRASH") as excinfo:
        with fx.audit():
            eng = fx.Engine(model=StiffCubicODE(), target="cpu:serial")
            eng.solve(t_span=(0.0, 1.0), max_newton_iters=1, show_progress=False)

    assert "Top Offenders" in str(excinfo.value)
    assert get_active_audit() is None


@REQUIRES_RUNTIME
def test_failure_cat6_realtime_frozen_step_and_rejection_health():
    """
    Cat 6b:
      - Realtime session forced to diverge (`max_newton_iters=1`, large `dt`) must increment
        `realtime_frozen_steps` instead of crashing silently or pretending to converge.
      - Feeding frozen-step or high-rejection diagnostics into `run_solver_audit` must emit
        `FAIL` (frozen > 0) and `WARN` (rej_ratio > 0.2) respectively.
    """
    class StiffNonlinear(fx.PDE):
        y = fx.State(domain=None, name="y")

        def math(self):
            return {
                "equations": {self.y: fx.dt(self.y) == -100.0 * (self.y ** 3)},
                "boundaries": {},
                "initial_conditions": {self.y: 5.0},
            }

    eng = fx.Engine(model=StiffNonlinear(), target="cpu:serial", audit_compiler=True)
    rt_sess = eng.start_session(mode="realtime", max_newton_iters=1)
    rt_sess.step(dt=10.0)
    assert rt_sess.diagnostics["realtime_frozen_steps"] == 1

    # Verify audit classification on a result carrying frozen steps
    res = eng.solve(t_span=(0.0, 0.1), requires_grad=[], show_progress=False)
    res.diagnostics["realtime_frozen_steps"] = 2
    rep_frozen = run_solver_audit(eng, res, {}, None, (0.0, 0.1), None, None, None)
    assert rep_frozen.checks["step_health"]["status"] == "FAIL"
    assert rep_frozen.passed is False

    # Verify high rejection ratio triggers WARN
    res.diagnostics["realtime_frozen_steps"] = 0
    res.diagnostics["total_steps"] = 10
    res.diagnostics["rejected_steps"] = 4
    res.diagnostics["step_rejection_ratio"] = 0.4
    rep_rej = run_solver_audit(eng, res, {}, None, (0.0, 0.1), None, None, None)
    assert rep_rej.checks["step_health"]["status"] == "WARN"


# ==============================================================================
# SECTION 8: Failure Category 7 — Auditor Fault-Injection Self-Checks
# ==============================================================================

@REQUIRES_RUNTIME
def test_failure_cat7_corrupted_cpr_schedule_detected_by_static_audit():
    """
    Cat 7a: Drops an off-diagonal entry from `cpr_cache` and asserts `run_native_static_audit`
    catches the CPR-vs-Enzyme JVP mismatch instead of silently passing.
    """
    eng = fx.Engine(model=CleanAuditedBatteryCell(), target="cpu:serial", audit_compiler=True)
    m = eng.manifest

    # Sanity check: uncorrupted CPR passes
    clean_audit = run_native_static_audit(m, m.default_parameters, m.cpr_cache, m.spatial_diag, m.max_steps)
    assert clean_audit["cpr_jvp_probe_passed"] is True

    # Corrupt CPR by dropping all non-diagonal sparse triplets
    c_seeds, _, c_rows, c_cols, c_dense = m.cpr_cache
    diag_only = [(r, c) for r, c in zip(c_rows, c_cols) if r == c]
    assert len(diag_only) < len(c_rows), "Expected off-diagonal sparse entries in test model"

    corrupted_ptrs = [0]
    corrupted_rows, corrupted_cols = [], []
    color_of_col = {}
    for c_idx, seed in enumerate(c_seeds):
        for col, val in enumerate(seed):
            if val != 0.0:
                color_of_col[col] = c_idx
    for c_idx in range(len(c_seeds)):
        cnt = 0
        for r, c in diag_only:
            if color_of_col.get(c) == c_idx:
                corrupted_rows.append(r)
                corrupted_cols.append(c)
                cnt += 1
        corrupted_ptrs.append(corrupted_ptrs[-1] + cnt)

    bad_cpr = (c_seeds, corrupted_ptrs, corrupted_rows, corrupted_cols, c_dense)
    bad_audit = run_native_static_audit(m, m.default_parameters, bad_cpr, m.spatial_diag, m.max_steps)
    assert bad_audit["cpr_jvp_probe_passed"] is False
    assert bad_audit["cpr_jvp_probe_max_diff"] > 1e-4


@REQUIRES_RUNTIME
def test_failure_cat7_corrupted_trajectory_detected_by_residual_sweep():
    """
    Cat 7b: Perturbs a microstep state on a recorded trajectory and asserts
    `run_solver_audit` flags `dae_residual['status'] == 'FAIL'` and `passed is False`.
    """
    eng = fx.Engine(model=CleanAuditedBatteryCell(), target="cpu:serial", audit_compiler=True)
    res = eng.solve(t_span=(0.0, 1.0), requires_grad=["D"], show_progress=False)

    mid = len(res.trajectory["_micro_t"]) // 2
    res.trajectory["_micro_y"][mid, 0] += 5.0

    rep = run_solver_audit(eng, res, eng.manifest.default_parameters, None, (0.0, 1.0), None, None, None)
    assert rep.checks["dae_residual"]["status"] == "FAIL"
    assert rep.checks["dae_residual"]["max_wrms_residual"] > 20.0
    assert rep.passed is False


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])