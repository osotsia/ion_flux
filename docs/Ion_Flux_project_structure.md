### **Core Principles Embodied in This Structure**

*   **Architecture Pattern:** A strict "Compiler-and-Runtime" architecture leveraging **Staged Intermediate Representations** and **Compiler-Level Automatic Differentiation (Enzyme)**.
    *   **Frontend (The DSL - `compiler/_1_frontend`):** Pure Python captures mathematical intent via operator overloading into an immutable Abstract Syntax Tree (AST). It performs no calculus, discretization, or execution.
    *   **Middle-end (Topology & Semantic Pass - `compiler/_2_middle_end`):** Validates manifold closure and topological continuity. Resolves boundary conditions and regional sub-meshes, lowering the AST into an N-Dimensional **Math IR**.
    *   **Backend (Discretization, Optimization, Codegen - `compiler/_3_backend` & `_4_codegen`):** Discretizes Math IR into a 1D loop-level **Compute IR**. Flattens multi-scale domains into contiguous C-array strides, applies Finite Volume Method (FVM) stencils, auto-stitches piecewise interfaces via harmonic mean flux balancing, and applies Arbitrary Lagrangian-Eulerian (ALE) grid kinematics. Analyzes Jacobian column intersections via Curtis-Powell-Reid (CPR) graph coloring, stringifies Compute IR to C++, and invokes Clang with the Enzyme LLVM plugin to emit a native `.so` binary with compiled JVP and VJP routines.
    *   **Runtime (Execution & Native Solver - `runtime/` & `rust/`):** Python handles declarative input normalization, telemetry, and FFI packing. Rust takes over numerical execution and protocol state-machine orchestration (`control/sequence.rs`), architected as a **Data-Oriented Modular Design** (`api/`, `ffi/`, `stepper/`, `nonlinear/`, `linear/`, `control/`, `diagnostics/`, `vendor/`), strictly segregating persistent physical states (`SystemState`), polynomial history (`BdfHistory`), and transient iteration scratchpads (`StepScratchpad`). This eliminates state cloning, ensures static sparsity retention, and avoids god-objects during stiff implicit integration.

---

### **Execution Pipeline (Data Flow)**

```text
[Python DSL] ──────────(Operator Overloading)──> [AST Payload]      # Mathematical intent captured as an immutable 
                                                                    AST. No execution or discretization occurs here.
                                                      │
[Middle-End] ──────────(Topology & Lowering)───> [Math IR (N-D)]    # Validates manifold closure. Resolves domains,
                                                                    boundary bindings, and unrolls piecewise syntax.
                                                      │
[Backend Discretizer] ─(FVM Lowering)──────────> [Compute IR (1D)]  # Linearizes N-D coordinates into 1D memory strides.
                                                                    Applies FVM stencils, harmonic stitching & ALE.
                                                      │
[Sparsity Optimizer] ──(CPR Graph Coloring)────> [AD Schedule]      # Traces Compute IR to find Jacobian triplets and
                                                                    derives minimal perturbation color seeds.
                                                      │
[Codegen + Clang/LLVM]─(Compile-Time AD)───────> [.so Binary]       # Stringifies Compute IR to C++, invokes Clang with
                                                                    Enzyme to emit exact analytical Jacobians & VJPs.
                                                      │
[Rust FFI Boundary] ───(Struct Unpacking)──────> [Problem + State]  # Maps multidimensional arrays to flat C-ABI pointers.
                                                                    Pre-allocates SystemState, History, & Scratchpad.
                                                      │
[Rust Native Solver] ──(Control -> Stepper)────> [Time Stepping]    # Integrates stiff DAEs by passing disjoint state and
                                                                    scratchpad references without state cloning.
                                                      │
                                                      ▼
                                       [Hardware: CPU Execution]    # Orchestrates Rayon CPU batching or OpenMP threads.
```

---

### **Project Structure: `ion_flux/`**

The compiler directory structure reflects a strict chronological lowering pipeline (`_1_` to `_4_`). The native Rust solver is organized as a **Data-Oriented Modular Architecture** partitioned by functional domain and memory lifecycle (`api/`, `ffi/`, `stepper/`, `nonlinear/`, `linear/`, `control/`, `diagnostics/`, `vendor/`), eliminating circular dependencies while ensuring deterministic memory access patterns.

```text
ion_flux/
├── docs/                           
│   ├── API.md                      
│   ├── Ion_Flux_project_structure.md 
│   └── refactor/                   # Architectural migration records and IR design specifications.
│
├── examples/                       
│   └── ...                         # Minimal runnable scripts isolating architectural features.
├── models/                         
│   └── ...                         # Full-scale reference battery models (Chen2020 DFN, TSPMe, etc.).
│
├── python/                         
│   └── ion_flux/
│       ├── cli.py                  # CLI utility for hermetic fetching/building of LLVM + Enzyme.
│       ├── metrics.py              # Bridges Python loss functions to Rust's VJP adjoint solvers.
│       ├── protocols.py            # Public facade exporting protocol definitions.
│       │
│       ├── compiler/               # --- COMPILER PIPELINE ---
│       │   ├── pipeline.py         # Central orchestrator driving stages _1 through _4.
│       │   │
│       │   ├── _1_frontend/        # --- INTENT CAPTURE ---
│       │   │   ├── core.py         # Public exports (PDE, State, Parameter, Observable, Domain, etc.).
│       │   │   ├── nodes.py        # Operator-overloaded AST nodes (BinaryOp, UnaryOp, Boundary, Scalar).
│       │   │   ├── operators.py    # Topology-agnostic math operators (grad, div, dt, integral, clamp).
│       │   │   ├── pde.py          # Submodel merging, AST namespace prefixing, and Terminal constraints.
│       │   │   └── spatial.py      # Domain topologies, sub-mesh slicing, and unstructured mesh loaders.
│       │   │
│       │   ├── _2_middle_end/      # --- TOPOLOGICAL ANALYSIS & MATH IR ---
│       │   │   ├── ast_utils.py    # AST node inspection and state extraction helpers.
│       │   │   ├── memory_layout.py# Computes contiguous 1D memory offsets and FVM metric caches.
│       │   │   ├── semantics.py    # Pre-processes boundary buckets into O(1) lookup tables.
│       │   │   ├── topology.py     # Analyzes composite domains, base axes, and memory strides.
│       │   │   └── verification.py # Proves manifold closure and catches topological gaps/overlaps.
│       │   │
│       │   ├── _3_backend/         # --- FVM DISCRETIZATION & SPARSITY ---
│       │   │   ├── math_ir.py      # Typed representation of continuum mathematics and boundary conditions.
│       │   │   ├── normalization.py# Lowers untyped AST dictionaries into typed MathSystem IR.
│       │   │   ├── discretizer.py  # FVMDiscretizer: lowers Math IR to 1D Compute IR loops and stencils.
│       │   │   ├── cpr_coloring.py # Welsh-Powell column-intersection graph coloring for compressed AD.
│       │   │   ├── cpr_orchestrator.py # Bridges Compute IR sparsity tracing to CPR color schedule generation.
│       │   │   └── sparsity_tracer.py # Evaluates Compute IR loops in Python to trace Jacobian (row, col) triplets.
│       │   │
│       │   └── _4_codegen/         # --- MECHANICAL EMISSION & AOT TOOLCHAIN ---
│       │       ├── builder.py      # Coordinates Math IR discretization and C++ string assembly.
│       │       ├── compute_ir.py   # Strongly typed loop-level IR (Loop, Assign, ArrayAccess, BinaryOp).
│       │       ├── cpp_emitter.py  # Mechanically stringifies Compute IR nodes into valid C++ code.
│       │       ├── templates.py    # Jinja2 C++ skeleton wrapping residuals, observables, JVPs, and VJPs.
│       │       └── clang_invoker.py# Subprocesses Clang + Enzyme to compile C++ into a native .so binary.
│       │
│       └── runtime/                # --- PYTHON EXECUTION ORCHESTRATION ---
│           ├── engine.py           # User-facing API facade (solve, solve_batch, solve_async, export_binary).
│           ├── manifest.py         # ExecutableManifest: serializable artifact holding layouts and binary paths.
│           ├── ffi_runtime.py      # ctypes C-ABI boundary loading and executing compiled .so binaries.
│           ├── session.py          # Stateful handle preserving hot solver memory for real-time HIL loops.
│           ├── _1_builder.py       # Bridges user model AST to the Compiler and builds ExecutableManifest.
│           ├── _2_initializers.py  # Evaluates initial conditions (y0, ydot0) from parameter inputs at runtime.
│           ├── _3_dispatch.py      # Packs flat C-arrays and delegates execution to Rust FFI entry points.
│           ├── _4_diagnostics.py   # Intercepts native solver crash payloads and maps flat indices to state names.
│           ├── eis.py              # Analytical Electrochemical Impedance Spectroscopy via steady-state Jacobians.
│           ├── results.py          # Unpacks flat C-arrays into indexed multidimensional trajectory structures.
│           ├── scheduler.py        # Asynchronous multi-tenant task scheduler with concurrency limits.
│           ├── telemetry.py        # Diagnoses cache hit rates, average stride jumps, and matrix sparsity.
│           └── protocols/
│               └── profiles.py     # Declarative cycler protocols (CC, CV, Rest, Sequence) and triggers.
│
├── rust/                           # --- NATIVE BACKEND (Data-Oriented Modular Solver) ---
│   ├── Cargo.toml                  
│   ├── build.rs                    # Links SUNDIALS archives and configures Mach-O dynamic lookup.
│   └── src/
│       ├── lib.rs                  # PyO3 module definition and Python C-extension exports.
│       ├── api/                    # --- PYTHON/RUST FFI BOUNDARY ---
│       │   ├── mod.rs              # PyO3 exception mapping and error conversion utilities.
│       │   ├── batch_dispatcher.rs # Unified single and batch solver entrypoint (`solve_native`).
│       │   ├── session_handle.rs   # SolverHandle exposing stateful step(), get_state(), and checkpoint().
│       │   └── adjoint_dispatcher.rs # Reverse-mode continuous discrete adjoint entrypoint.
│       │
│       ├── ffi/                    # --- C-ABI ENCAPSULATION LAYER ---
│       │   ├── mod.rs
│       │   ├── callbacks.rs        # Native function pointer signatures (JVP, VJP, Residual).
│       │   └── safe_evaluator.rs   # Bounds-checked slice wrapper over compiled C++ shared libraries.
│       │
│       ├── stepper/                # --- TIME-DOMAIN INTEGRATION & MEMORY ---
│       │   ├── mod.rs
│       │   ├── problem.rs          # Immutable system problem, CPR data, and solver tolerances.
│       │   ├── state.rs            # Segregated memory layouts: SystemState and StepScratchpad.
│       │   ├── history.rs          # Nordsieck history arrays and polynomial predictors (BdfHistory).
│       │   └── bdf.rs              # Variable-Step Variable-Order (VSVO) BDF integration loop.
│       │
│       ├── nonlinear/              # --- NON-LINEAR ROOT-FINDING ---
│       │   ├── mod.rs
│       │   ├── newton.rs           # Modified Newton-Raphson solver with contraction-rate monitoring.
│       │   ├── constraints.rs      # WRMS norms and physical state clamping (non-negative concentrations).
│       │   └── algebraic_roots.rs  # Pre-step DAE algebraic manifold root-finding.
│       │
│       ├── linear/                 # --- LINEAR ALGEBRA & JACOBIAN ASSEMBLY ---
│       │   ├── mod.rs
│       │   ├── solver.rs           # Closed LinearSolver enum dispatch (SparseLu vs. Gmres).
│       │   ├── sparse_lu.rs        # Faer Sparse LU solver with cached SymbolicLu reuse.
│       │   ├── gmres.rs            # Preconditioned matrix-free restarted GMRES for 3D CSR meshes.
│       │   └── jacobian.rs         # CPR color-sweep Jacobian assembly into pre-allocated scratch.
│       │
│       ├── control/                # --- HIGH-LEVEL CONTROL FLOW & SENSITIVITIES ---
│       │   ├── mod.rs
│       │   ├── sequence.rs         # Declarative multi-stage protocol runner (CC, CV, Rest).
│       │   ├── bisection.rs        # Dense bisection root-finding for discontinuous event triggers.
│       │   └── backward_pass.rs    # Reverse-mode discrete adjoint loop with factorization caching.
│       │
│       ├── diagnostics/            # --- TELEMETRY & CRASH REPORTING ---
│       │   ├── mod.rs
│       │   ├── crash_report.rs     # Structured CrashReport data structures (Offender, condition numbers).
│       │   └── telemetry.rs        # Non-allocating diagnostics tracking and performance timers.
│       │
│       └── vendor/                 # --- EXTERNAL ORACLES ---
│           └── sundials/
│               ├── mod.rs
│               ├── sys.rs          # Dynamic linking isolation and MPI workarounds.
│               └── wrapper.rs      # Embedded LLNL SUNDIALS (IDA) verification oracle.
│
├── tests/                          # --- COMPREHENSIVE ORACLE TEST SUITE ---
│   ├── conftest.py                 
│   ├── 01_frontend_dsl/            # Verifies immutable AST capture and semantic boundary routing.
│   ├── 02_middle_end_codegen/      # Verifies FVM geometry scaling, harmonic auto-stitching, and codegen.
│   ├── 03_backend_compilation/     # Verifies Clang invocation, Enzyme AD exactness, and CPR graph coloring.
│   ├── 04_runtime_execution/       # Verifies Rust solver, BDF convergence, ALE kinematics, and MMS oracles.
│   ├── 05_e2e_integration/         # Verifies industry benchmark models (DFN/TSPMe) and adjoint inversion.
│   ├── 06_benchmarks/              # Property-based hypothesis fuzzing and solver robustness torture suites.
│   └── bugfixes/                   # Explicit regression probes covering historical compiler/solver defects.
│
├── pyproject.toml                  
└── README.md
```