### **Core Principles Embodied in This Structure**

*   **Architecture Pattern:** A "Compiler-and-Runtime" architecture leveraging **Compiler-Level Automatic Differentiation (Enzyme)**.
    *   **Frontend (The DSL):** Pure Python captures mathematical intent via operator overloading into an Abstract Syntax Tree (AST). It performs no calculus or execution.
    *   **Middle-end (The Compiler):** Python dynamically dispatches the AST through a strict, numbered sequence of lowering passes (`_1_analysis` -> `_4_codegen`). It enforces unidirectional data flow, separating semantic topology resolution from string emission. It emits native C++ source code exclusively for the *residual* equation $F(t, y, \dot{y}) = 0$.
    *   **Backend (The AD Compiler):** Python subprocesses `clang++` equipped with the Enzyme LLVM plugin. Enzyme differentiates the compiled Intermediate Representation (IR) of the residual to automatically generate a highly optimized Forward Jacobian and Reverse Vector-Jacobian Product (VJP) at compile time.
    *   **Runtime (The Native Solver):** Rust takes over. The solver is architected as a chronological "Nested Doll" (`_0_ffi` -> `_4_linear`), strictly separating immutable topological definitions (`Problem`) from pre-allocated mutable execution memory (`Workspace`). This eliminates memory fragmentation and avoids "God objects" during the stiff implicit integration loop.

---

### **Execution Pipeline (Data Flow)**

```text
[Python DSL] ──────────(Operator Overloading)──> [AST Payload]      # Mathematical intent captured as a pure JSON/Dict 
                                                                    semantic graph. No execution occurs here.
                                                      │
[Python Compiler] ─────(Staged MIR Lowering)───> [C++ Source]       # Translates AST to Math IR, mapping dynamic domains
                                                                    and geometries (Cartesian/Spherical FVM) into explicit 
                                                                    flat C-arrays. Generates the residual skeleton.
                                                      │
[Clang/LLVM + Enzyme] ─(Compile-Time AD)───────> [.so Binary]       # JIT-compiles the residual. Enzyme differentiates 
                                                                    LLVM IR to emit exact analytical Jacobians and 
                                                                    Reverse-mode VJPs (Vector-Jacobian Products).
                                                      │
[Rust FFI Boundary] ───(Struct Unpacking)──────> [Problem + Wkspc]  # Maps multi-dimensional Python arrays to flat C-ABI 
                                                                    pointers. Isolates immutable config from the mutable arena.
                                                      │
[Rust Native Solver] ─-(Orchestrator -> Linear)> [Time Stepping]    # Integrates stiff non-linear DAEs by passing the workspace
                                                                    down the call stack, mutating state without reallocation.
                                                      │
                                                      ▼
                                       [Hardware: CPU Execution]    # Orchestrates Task-Parallel batching via Rayon
                                                                    or Data-Parallel OpenMP loops across available vCPUs.
```

---

### **Project Structure: `ion_flux/`**

The directory structure reflects a strict chronological execution flow. The numbered prefixes (`_1_` to `_5_` in the compiler, `_0_` to `_4_` in the solver) explicitly dictate the data dependency and call stack depth. A module is strictly prohibited from importing logic from a "deeper" chronological module.

```text
ion_flux/
├── docs/                           
│   ├── API.md                      
│   └── Ion_Flux_project_structure.md 
├── examples/                       
│   └── ...                         # Minimal runnable scripts isolating architectural features.
├── models/                         
│   └── ...                         # Full-scale implementations serving as regression baselines.
│
├── python/                         
│   └── ion_flux/
│       ├── cli.py                  # Automates hermetic fetching/building of LLVM 19 + Enzyme.
│       ├── metrics.py              # Bridges Python loss functions to Rust's VJP adjoint solvers.
│       ├── dsl/                    # --- FRONTEND ---
│       │   ├── core.py             
│       │   ├── nodes.py            # Operator-overloaded AST nodes (e.g., BinaryOp, UnaryOp).
│       │   ├── operators.py        # Topology-agnostic math operators (grad, div, dt).
│       │   ├── pde.py              # Handles hierarchical submodel merging and AST namespace isolation.
│       │   └── spatial.py          # Domain topologies and moving-mesh bindings.
│       ├── protocols/              # --- STATE MACHINES ---
│       │   └── profiles.py         # Declarative sequence protocols (CC, CV, Rest) mapped to the Native Orchestrator.
│       ├── runtime/                # --- PYTHON EXECUTION ORCHESTRATION ---
│       │   ├── engine.py           # 1. Facade. Unifies User API (solve, solve_batch, load, export).
│       │   ├── manifest.py         # 2. Immutable Data Target. Holds MemoryLayout & Topological Constants.
│       │   ├── _1_builder.py       # 3. Compiler Boundary. Bridges model AST to LLVM & returns Manifest.
│       │   ├── _2_initializers.py  # 4. AST Evaluator. Dynamically calcs y0 & ydot0 from parameters.
│       │   ├── _3_dispatch.py      # 5. FFI Execution. Packs C-arrays & handles Rayon/Rust invocation.
│       │   ├── _4_diagnostics.py   # 6. Observability. Formats native Rust panics to clean Python errors.
│       │   ├── eis.py              # Solves Frequency-Domain impedance analytically via Enzyme Mass Matrices.
│       │   ├── results.py          # Wraps flat FFI C-arrays back into multidimensional Python structures.
│       │   ├── scheduler.py        # Async task batching limits.
│       │   ├── session.py          # Persistent handles preserving native memory for micro-stepping HIL.
│       │   └── telemetry.py        # Observability metrics for cache hits/sparsity.
│       └── compiler/               # --- MIDDLE-END (STAGED LOWERING) ---
│           ├── _1_analysis/        # Intent: Topological resolution and validation.
│           │   ├── ast_utils.py    
│           │   ├── memory_layout.py# Resolves dynamic domains into flat FVM indexing strides and areas/volumes.
│           │   ├── semantics.py    # Pre-processes implicit boundaries into O(1) lookup tables.
│           │   ├── topology.py     
│           │   └── verification.py # Detects topological overlaps/gaps before lowering to prevent silent overwrites.
│           ├── _2_lowering/        # Intent: Math transformation.
│           │   ├── context.py      # Immutable context passed down the AST visitor to prevent recursive state bugs.
│           │   ├── dialects.py     # Dispatches abstract grad/div to specific Cartesian/Spherical/Unstructured Math IR.
│           │   ├── ir.py           # Strictly typed Intermediate Representation (MIR) for loops and assignments.
│           │   ├── normalization.py# Unrolls syntactic sugar (e.g., Piecewise domains) into explicit regional equations.
│           │   └── spatial_visitor.py # Translates topology-agnostic math into explicit MIR arrays.
│           ├── _3_optimization/    # Intent: Differentiability analysis.
│           │   ├── cpr_coloring.py # Welsh-Powell column-intersection graph coloring. Minimizes JVP sweeps.
│           │   └── sparsity_tracer.py # Evaluates the MIR natively in Python to trace Jacobian sparsity triplets.
│           ├── _4_codegen/         # Intent: Mechanical C++ emission.
│           │   ├── builder.py      
│           │   ├── emitter.py      # Stringifies the MIR into C++. Contains zero mathematical logic.
│           │   └── templates.py    
│           └── _5_toolchain/       # Intent: Systems invocation.
│               ├── clang_invoker.py# Subprocesses Clang+Enzyme to emit `.so` binaries.
│               └── ffi_runtime.py  # `ctypes` wrapper defining the C-ABI boundary for the Rust backend.
│
├── rust/                           # --- NATIVE BACKEND ---
│   ├── Cargo.toml                  
│   ├── build.rs                    # Dynamically links SUNDIALS for the C-ABI oracle wrapper.
│   └── src/
│       ├── lib.rs                  
│       └── solver/                 # --- THE NESTED DOLL SOLVER ARCHITECTURE ---
│           ├── _0_ffi/             # Intent: Python boundary. Allocates the Workspace and spawns threads.
│           │   ├── api_adjoint.rs  # Reverse-mode Vector-Jacobian Product execution.
│           │   ├── api_batch.rs    # Rayon-distributed task parallelism. Bypasses the GIL.
│           │   └── api_session.rs  # Exposes stateful integration steps to Python.
│           ├── _1_orchestrator/    # Intent: Control flow. Drives absolute time and BMS state machines.
│           │   ├── bisection.rs    # Exact bisection root-finding for discontinuous protocol triggers.
│           │   └── protocol.rs     # Hot-swaps constraints (CC/CV) without rebuilding matrices.
│           ├── _2_stepper/         # Intent: Time integration and error control.
│           │   ├── bdf.rs          # Predicts y(t+dt). Adjusts dt/order upon step rejection or truncation error.
│           │   └── history.rs      # Maintains Nordsieck arrays. Handles checkpointing and restorations.
│           ├── _3_nonlinear/       # Intent: Root-finding and bounding.
│           │   ├── constraints.rs  # Clamps proposed Newton steps to prevent physical violations.
│           │   └── newton.rs       # The Newton-Raphson loop. Detects divergence and thrashing.
│           ├── _4_linear/          # Intent: Algebraic math and C-ABI Enzyme evaluation.
│           │   ├── gmres.rs        # Matrix-free Krylov subspace methods.
│           │   ├── jacobian.rs     # Evaluates JVP/VJP function pointers to assemble the Sparse Jacobian.
│           │   └── sparse_lu.rs    # Faer LU factorization and substitution.
│           ├── shared/             # Intent: Eliminate God Objects. Forces unidirectional data flow.
│           │   ├── callbacks.rs    
│           │   ├── diagnostics.rs  # Localizes NaNs, dumps Matrix Market files, and emits JSON crash reports.
│           │   ├── problem.rs      # IMMUTABLE: Topology definitions, CPR data, and solver tolerances.
│           │   └── workspace.rs    # MUTABLE: Pre-allocated arrays (y, ydot, res, dy). Eliminates hot-loop allocation.
│           └── sundials/           # Intent: The external reference oracle.
│               └── wrapper.rs      # C-ABI callbacks mapping `sundials` structures to the `shared::problem`.
│
├── tests/                          # --- ORACLE-DRIVEN TEST SUITE ---
│   ├── conftest.py                 
│   ├── 01_frontend_dsl/            
│   ├── 02_middle_end_codegen/      
│   ├── 03_backend_compilation/     
│   ├── 04_runtime_execution/       
│   ├── 05_e2e_integration/         
│   ├── 06_benchmarks/              
│   └── bugfixes/                   # Explicit Method of Manufactured Solutions (MMS) probes designed to 
│                                   # isolate and prove the absence of specific historical compiler/solver failures.
├── pyproject.toml                  
└── README.md
```