### **Core Principles Embodied in This Structure**

*   **Architecture Pattern:** A strict "Compiler-and-Runtime" architecture leveraging **Staged Intermediate Representations** and **Compiler-Level Automatic Differentiation (Enzyme)**.
    *   **Frontend (The DSL - `compiler/_1_frontend`):** Pure Python captures mathematical intent via operator overloading into an immutable Abstract Syntax Tree (AST). It performs no calculus, discretization, or execution.
    *   **Middle-end (Topology & Semantic Pass - `compiler/_2_middle_end`):** Validates manifold closure and topological continuity. Resolves boundary conditions and regional sub-meshes, lowering the AST into an N-Dimensional **Math IR**.
    *   **Backend (Discretization, Optimization, Codegen - `compiler/_3_backend` & `_4_codegen`):** Discretizes Math IR into a 1D loop-level **Compute IR**. Flattens multi-scale domains into contiguous C-array strides, applies Finite Volume Method (FVM) stencils, auto-stitches piecewise interfaces via harmonic mean flux balancing, and applies Arbitrary Lagrangian-Eulerian (ALE) grid kinematics. Analyzes Jacobian column intersections via Curtis-Powell-Reid (CPR) graph coloring, stringifies Compute IR to C++, and invokes Clang with the Enzyme LLVM plugin to emit a native `.so` binary with compiled JVP and VJP routines.
    *   **Runtime (Execution & Native Solver - `runtime/` & `rust/`):** Python handles high-level protocol state machines, telemetry, and FFI packing. Rust takes over numerical execution, architected as a chronological "Nested Doll" (`_0_ffi` $\rightarrow$ `_4_linear`), strictly separating immutable topological definitions (`Problem`) from pre-allocated mutable execution memory (`Workspace`). This eliminates memory fragmentation and avoids God objects during stiff implicit integration.

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
[Rust FFI Boundary] ───(Struct Unpacking)──────> [Problem + Wkspc]  # Maps multidimensional arrays to flat C-ABI pointers.
                                                                    Pre-allocates mutable Workspace arena.
                                                      │
[Rust Native Solver] ──(Orchestrator -> Linear)> [Time Stepping]    # Integrates stiff DAEs by passing Workspace down call
                                                                    stack without hot-loop allocations.
                                                      │
                                                      ▼
                                       [Hardware: CPU Execution]    # Orchestrates Rayon CPU batching or OpenMP threads.
```

---

### **Project Structure: `ion_flux/`**

The directory structure reflects a strict chronological execution flow. The numbered prefixes (`_1_` to `_4_` in the compiler, `_0_` to `_4_` in the solver) explicitly dictate the data dependency and call stack depth. A module is strictly prohibited from importing logic from a deeper chronological module.

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
├── rust/                           # --- NATIVE BACKEND (Nested Doll Solver) ---
│   ├── Cargo.toml                  
│   ├── build.rs                    # Dynamically links SUNDIALS for the C-ABI verification oracle.
│   └── src/
│       ├── lib.rs                  # PyO3 module bindings and NativeSolverCrash exception definition.
│       └── solver/                 
│           ├── _0_ffi/             # Python C-ABI boundary, Workspace allocation, and thread spawning.
│           │   ├── api_session.rs  # SolverHandle exposing stateful step(), get_state(), and checkpoint().
│           │   ├── api_batch.rs    # Rayon-distributed task-parallel batching bypassing the GIL.
│           │   ├── api_adjoint.rs  # Reverse-mode continuous discrete adjoint integration.
│           │   └── mod.rs          # Translates Rust CrashReports into structured Python dictionaries.
│           │
│           ├── _1_orchestrator/    # Control flow, absolute time advancement, and protocol state machines.
│           │   ├── protocol.rs     # Hot-swaps CC/CV/Rest terminal constraints without rebuilding factorizations.
│           │   └── bisection.rs    # Dense bisection root-finding for discontinuous event triggers.
│           │
│           ├── _2_stepper/         # Variable-Step Variable-Order (VSVO) time integration and error control.
│           │   ├── bdf.rs          # BDF integration loop (orders 1-5), step rejection, and step-size adaptation.
│           │   └── history.rs      # Nordsieck history arrays, polynomial predictors, and state checkpoints.
│           │
│           ├── _3_nonlinear/       # Non-linear algebraic root-finding and state-clamping constraints.
│           │   ├── newton.rs       # Modified Newton-Raphson solver with contraction-rate thrashing detection.
│           │   └── constraints.rs  # WRMS norms and physical state clamping (e.g., non-negative concentration).
│           │
│           ├── _4_linear/          # Linear system factorizations and Enzyme Jacobian evaluations.
│           │   ├── jacobian.rs     # Executes JVP/VJP function pointers to assemble the Sparse Jacobian.
│           │   ├── sparse_lu.rs    # Faer Sparse LU symbolic factorization, numeric factorization, and substitution.
│           │   └── gmres.rs        # Matrix-free restarted GMRES solver for 3D unstructured CSR meshes.
│           │
│           ├── shared/             # Unidirectional state separation eliminating God Objects.
│           │   ├── problem.rs      # IMMUTABLE: Topology definitions, CPR data, and solver tolerances.
│           │   ├── workspace.rs    # MUTABLE: Pre-allocated hot-loop execution arena (y, ydot, res, dy, ee).
│           │   ├── callbacks.rs    # Raw C-ABI function pointer signatures.
│           │   └── diagnostics.rs  # Performance timers, NaN tracking, and structured CrashReport builders.
│           │
│           └── sundials/           # External C-ABI reference oracle.
│               └── wrapper.rs      # Embedded LLNL SUNDIALS (IDA) solver wrapper for numerical cross-checks.
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