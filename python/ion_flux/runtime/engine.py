import asyncio
import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional

from ion_flux.compiler._1_frontend.core import PDE
from ion_flux.runtime.manifest import ExecutableManifest, build_manifest
from ion_flux.runtime.session import Session
from ion_flux.runtime.results import SimulationResult, TelemetryReport

class _ParamHandle:
    __slots__ = ["name", "value"]
    def __init__(self, name: str, default: float):
        self.name = name
        self.value = default
    def __repr__(self) -> str: return f"Parameter({self.name}={self.value})"

class Engine:
    """
    The Facade unifying the Compiler and Runtime architecture.
    Orchestrates the lifecycle of lowering Python ASTs into ExecutableManifests, 
    and delegating the FFI workflows for Stateful or Stateless solving.
    """
    def __init__(self, model: Optional[PDE] = None, target: str = "cpu:serial", solver_backend: str = "native", 
                 cache: bool = True, mock_execution: bool = False, jacobian_bandwidth: Optional[int] = None, 
                 audit_compiler: Optional[bool] = None, **kwargs):
        from ion_flux.runtime.audit import get_active_audit
        active_audit = get_active_audit()
        resolved_audit_compiler = (
            audit_compiler if audit_compiler is not None
            else (active_audit.compiler if active_audit is not None else False)
        )

        self.model = model
        self.target = target
        self.solver_backend = solver_backend.lower()
        self.mock_execution = mock_execution
        self.audit_compiler = bool(resolved_audit_compiler)
        
        if model:
            self.manifest = build_manifest(
                model, target, cache, jacobian_bandwidth, mock_execution, audit_compiler=self.audit_compiler
            )
        else:
            self.manifest = None

        if active_audit is not None and self.audit_compiler and self.manifest and self.manifest.static_audit:
            active_audit.record_static(self.audit_compiler_report())
            
        self.parameters = {k: _ParamHandle(k, v) for k, v in self.manifest.default_parameters.items()} if self.manifest else {}
        for k, v in kwargs.items(): 
            setattr(self, k, v)

    @property
    def layout(self): 
        return self.manifest.layout if self.manifest else None

    @property
    def jacobian_bandwidth(self): 
        return self.manifest.jacobian_bandwidth if self.manifest else 0
    
    @property
    def cpp_source(self) -> str:
        return self.manifest.cpp_source if self.manifest else ""

    @property
    def cpp(self) -> str:
        return self.cpp_source

    def _ensure_ir_strings(self) -> None:
        if not self.manifest:
            return
        if self.manifest.math_ir and self.manifest.compute_ir:
            return
        if self.manifest.static_audit:
            self.manifest.math_ir = self.manifest.math_ir or self.manifest.static_audit.get("math_pretty", "")
            self.manifest.compute_ir = self.manifest.compute_ir or self.manifest.static_audit.get("compute_pretty", "")
            if self.manifest.math_ir and self.manifest.compute_ir:
                return
        if not self.manifest.ast_payload or self.model is None:
            return

        from ion_flux.compiler._1_frontend.core import State, Observable
        from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
        from ion_flux.compiler._2_middle_end.semantics import SemanticContext
        from ion_flux.compiler._3_backend.normalization import NormalizationPass
        from ion_flux.compiler._3_backend.discretizer import FVMDiscretizer
        from ion_flux.compiler._3_backend.compute_ir import format_compute_ir

        states = self.model.components(State) if hasattr(self.model, "components") else [a for a in self.model.__dict__.values() if isinstance(a, State)]
        observables = self.model.components(Observable) if hasattr(self.model, "components") else [a for a in self.model.__dict__.values() if isinstance(a, Observable)]
        state_map = {s.name: s for s in states}
        state_map.update({o.name: o for o in observables})

        topo = TopologyAnalyzer(self.manifest.ast_payload.get("domains", {}))
        semantic_ctx = SemanticContext(self.manifest.ast_payload)
        norm_pass = NormalizationPass(self.manifest.ast_payload, topo, semantic_ctx, state_map, self.manifest.layout)
        math_sys = norm_pass.lower_to_math_ir()
        self.manifest.math_ir = math_sys.pretty()
        discretizer = FVMDiscretizer(self.manifest.layout, topo, semantic_ctx, state_map, self.target, math_sys=math_sys)
        _, eq_stmts, _ = discretizer.discretize_system(math_sys, audit_compiler=self.audit_compiler)
        self.manifest.compute_ir = format_compute_ir(eq_stmts)

    @property
    def math_ir(self) -> str:
        self._ensure_ir_strings()
        return self.manifest.math_ir if self.manifest else ""

    @property
    def compute_ir(self) -> str:
        self._ensure_ir_strings()
        return self.manifest.compute_ir if self.manifest else ""

    def _pack_parameters(self, overrides): 
        return self.manifest.pack_parameters(overrides)

    # --- Architectural Serialization Boundaries ---
    @classmethod
    def load(cls, binary_path: str, target: str = "cpu:serial", solver_backend: str = "native") -> "Engine":
        engine = cls.__new__(cls)
        engine.target = target
        engine.solver_backend = solver_backend
        engine.mock_execution = False
        
        engine.manifest = ExecutableManifest.load(binary_path)
        engine.audit_compiler = bool((engine.manifest.static_audit or {}).get("audit_compiler", False))
        engine.parameters = {name: _ParamHandle(name, val) for name, val in engine.manifest.default_parameters.items()}
        return engine

    def export_binary(self, export_path: str) -> None:
        if not self.manifest or not self.manifest.lib_path:
            raise RuntimeError("Engine has not compiled a native binary. Cannot export.")
        self.manifest.save(export_path)

    @property
    def telemetry(self) -> Optional[TelemetryReport]: 
        return TelemetryReport(self.manifest) if self.manifest else None

    def audit_compiler_report(self) -> Dict[str, Any]:
        """Returns compile-time static audit diagnostics when Engine(..., audit_compiler=True)."""
        if not self.manifest or not self.manifest.static_audit:
            raise RuntimeError("Static compiler audit unavailable. Instantiate Engine with `audit_compiler=True`.")
        audit = self.manifest.static_audit
        if "cpr_jvp_probe_passed" not in audit and self.manifest.native_lib is not None:
            from ion_flux.runtime.manifest import run_native_static_audit
            audit.update(run_native_static_audit(
                self.manifest,
                self.manifest.default_parameters,
                self.manifest.cpr_cache,
                self.manifest.spatial_diag,
                self.manifest.max_steps,
            ))
        return dict(audit)

    def audit(self, protocol: Any = None, parameters: Optional[Dict[str, float]] = None,
              t_span: tuple = (0, 1), t_eval: Optional[np.ndarray] = None,
              rel_tol: Optional[float] = None, abs_tol: Optional[float] = None) -> Any:
        """Executes a full solver verification pass and returns the consolidated AuditReport."""
        res = self.solve(
            t_span=t_span,
            protocol=protocol,
            parameters=parameters,
            t_eval=t_eval,
            show_progress=False,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            audit_solver=True,
        )
        return res.audit

    def start_session(self, parameters: Optional[Dict[str, float]] = None, soc: Optional[float] = None, mode: str = "scientific",
                      rel_tol: Optional[float] = None, abs_tol: Optional[float] = None,
                      max_newton_iters: Optional[int] = None, record_trace: bool = False) -> Session:
        """
        Instantiates a persistent native solver session.

        Enables Hardware-in-the-Loop (HIL) and Real-Time control logic. BDF history 
        vectors, Nordsieck arrays, and sparse LU factorizations remain "hot" in hardware memory, 
        avoiding allocation overhead during continuous micro-stepping.

        Args:
            parameters (Dict[str, float], optional): Base parameter initialization for the session.
            soc (float, optional): Initial State of Charge.
            mode (str): Execution profile ("scientific" for adaptive VSVO BDF, "realtime" for deterministic bounds).
            rel_tol (float, optional): Override relative integration tolerance.
            abs_tol (float, optional): Override absolute integration tolerance.
            max_newton_iters (int, optional): Override maximum Newton iterations per step.
            record_trace (bool): Retain per-step solver trace vectors in scientific mode.

        Returns:
            Session: A stateful object holding the active FFI integration pointer.
        """
        return Session(
            engine=self, parameters=parameters or {}, soc=soc, mode=mode,
            rel_tol=rel_tol, abs_tol=abs_tol, max_newton_iters=max_newton_iters, record_trace=record_trace
        )

    # --- Primary Dispatch Hooks ---
    def evaluate_residual(self, y: List[float], ydot: List[float], parameters: Optional[Dict[str, float]] = None) -> List[float]:
        if self.mock_execution or not self.manifest:
            raise RuntimeError("Requires native execution.")
        return self.manifest.evaluate_residual(y, ydot, parameters)

    def evaluate_jacobian(self, y: List[float], ydot: List[float], c_j: float, parameters: Optional[Dict[str, float]] = None) -> List[List[float]]:
        if self.mock_execution or not self.manifest:
            raise RuntimeError("Requires native execution.")
        return self.manifest.evaluate_jacobian(y, ydot, c_j, parameters).tolist()

    def solve(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, 
              t_eval: Optional[np.ndarray] = None, requires_grad: Optional[List[str]] = None, threads: int = 1, show_progress: bool = True,
              rel_tol: Optional[float] = None, abs_tol: Optional[float] = None,
              max_newton_iters: Optional[int] = None, record_trace: bool = False,
              audit_solver: Optional[bool] = None) -> SimulationResult:
        """
        Executes a single continuous simulation trajectory.
        Canonicalizes inputs and delegates directly to solve_batch.
        """
        from ion_flux.runtime.audit import get_active_audit
        active_audit = get_active_audit()
        resolved_audit_solver = (
            audit_solver if audit_solver is not None
            else (active_audit.solver if active_audit is not None else False)
        )
        effective_rel_tol = rel_tol if rel_tol is not None else (active_audit.rel_tol if active_audit is not None else None)
        effective_abs_tol = abs_tol if abs_tol is not None else (active_audit.abs_tol if active_audit is not None else None)

        current_params = {**{k: v.value for k, v in self.parameters.items()}, **(parameters or {})}
        effective_record_trace = record_trace or bool(resolved_audit_solver)
        effective_requires_grad = requires_grad if requires_grad is not None else ([] if resolved_audit_solver else None)

        results = self.solve_batch(
            parameters=[current_params],
            t_span=t_span,
            protocols=[protocol] if protocol is not None else None,
            t_eval=t_eval,
            requires_grad=effective_requires_grad,
            max_workers=1,
            threads=threads,
            show_progress=show_progress,
            rel_tol=effective_rel_tol,
            abs_tol=effective_abs_tol,
            max_newton_iters=max_newton_iters,
            record_trace=effective_record_trace,
        )
        res = results[0]
        if resolved_audit_solver:
            from ion_flux.runtime.audit import run_solver_audit
            res.audit = run_solver_audit(
                engine=self,
                result=res,
                parameters=current_params,
                protocol=protocol,
                t_span=t_span,
                t_eval=t_eval,
                rel_tol=effective_rel_tol,
                abs_tol=effective_abs_tol,
            )
            if active_audit is not None:
                active_audit.record_solver(res.audit)
        return res

    def solve_batch(self, parameters: List[Dict[str, float]], t_span: tuple = (0, 1), protocols: Any = None, 
                    t_eval: Optional[np.ndarray] = None, requires_grad: Optional[List[str]] = None, 
                    max_workers: int = 1, threads: int = 1, show_progress: bool = False,
                    rel_tol: Optional[float] = None, abs_tol: Optional[float] = None,
                    max_newton_iters: Optional[int] = None, record_trace: bool = False) -> List[SimulationResult]:
        """
        Executes an array of independent models across multiple vCPUs concurrently.
        """
        from ion_flux.protocols import Sequence
        if protocols:
            if isinstance(protocols, Sequence): protocols = [protocols] * len(parameters)
            elif len(protocols) != len(parameters): raise ValueError("Batch length mismatch.")
        else:
            protocols = None

        total = len(parameters)
        spatial_threads = threads if total == 1 else 1

        if spatial_threads > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = str(spatial_threads)
        elif total > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = "1"

        if getattr(self.manifest, "native_lib", None):
            self.manifest.native_lib.set_spatial_threads(spatial_threads)

        base_params = {k: v.value for k, v in self.parameters.items()}
        merged_parameters = [{**base_params, **p} for p in parameters]

        if self.mock_execution or not self.manifest:
            from ion_flux.runtime.backends.mock import execute_mock
            mock_protos = protocols or [None] * len(merged_parameters)
            return [execute_mock(self.manifest, p, prot) for p, prot in zip(merged_parameters, mock_protos)]

        from ion_flux.runtime.backends import dispatch_native

        n_states = self.manifest.layout.n_states
        y0, ydot0 = [0.0] * n_states, [0.0] * n_states

        t_eval_arr = t_eval if t_eval is not None else np.linspace(t_span[0], t_span[1], 100)

        def _get_v_idx():
            for k in ["V_cell", "V", "v", "Voltage"]:
                if k in self.manifest.layout.state_offsets:
                    return self.manifest.layout.state_offsets[k][0]
            return -1

        v_idx = _get_v_idx()

        dispatched = dispatch_native(
            self.manifest, y0, ydot0, merged_parameters, t_eval_arr, protocols,
            self.solver_backend, requires_grad, max_workers, spatial_threads, show_progress, v_idx,
            record_trace=record_trace, rel_tol=rel_tol, abs_tol=abs_tol, max_newton_iters=max_newton_iters,
        )

        results = []
        for p, (t_res, y_res, obs_res, trajectory, diag_dict) in zip(merged_parameters, dispatched):
            data = {"Time [s]": t_res}
            for state_name, (offset, size) in self.manifest.layout.state_offsets.items():
                if size == 1: data[state_name] = y_res[:, offset]
                else: data[state_name] = y_res[:, offset:offset+size]
            for obs_name, (offset, size) in self.manifest.layout.obs_offsets.items():
                if size == 1: data[obs_name] = obs_res[:, offset]
                else: data[obs_name] = obs_res[:, offset:offset+size]
            results.append(SimulationResult(data, p, status="completed", engine=self, trajectory=trajectory, diagnostics=diag_dict))

        return results

    async def solve_async(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, t_eval: Optional[np.ndarray] = None, scheduler: Any = None) -> SimulationResult:
        """
        Asynchronous wrapper for `solve`.
        """
        if scheduler:
            async with scheduler: 
                return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)
        return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)