### **Core Principles Embodied in This Structure**

*   **Architecture Pattern:** A "Compiler-and-Runtime" architecture leveraging **Compiler-Level Automatic Differentiation (Enzyme)**.
    *   **Frontend (The DSL):** Pure Python captures mathematical intent via operator overloading into an Abstract Syntax Tree (AST). It performs no calculus or execution.
    *   **Middle-end (The Translator):** Python (`codegen/`) dynamically dispatches the AST, navigating hierarchical topologies and moving boundaries (ALE). It emits native C++ source code exclusively for the *residual* equation $F(t, y, \dot{y}) = 0$. It does not perform symbolic differentiation.
    *   **Backend (The AD Compiler):** Python subprocesses `clang++` equipped with the Enzyme LLVM plugin. Enzyme differentiates the compiled Intermediate Representation (IR) of the residual to automatically generate a highly optimized Forward Jacobian and Reverse Vector-Jacobian Product (VJP) at compile time.
    *   **Runtime (The Native Solver):** Rust (`_core`) takes over. It loads the compiled `.so` binary via FFI, strictly scopes memory hardware lifecycles, and orchestrates the native implicit integration loop, passing the Enzyme-generated derivatives as C-callbacks to $\mathcal{O}(N)$ Banded solvers.

---

### **Execution Pipeline (Data Flow)**

```text
### **Execution Pipeline (Data Flow)**

```text
[Python DSL] ──────────(Operator Overloading)──> [AST Payload]      # Mathematical intent captured as a pure JSON/Dict 
                                                                    semantic graph. No execution occurs here.
                                                      │
[Middle-end Codegen] ──(Staggered FVM Lowering)─> [C++ Source]      # Translates AST to C++. Auto-stitches piecewise 
                                                                    regions, injects ALE moving mesh kinematics, and
                                                                    enforces mass-conserving Face/Volume mappings.
                                                      │
[Clang/LLVM + Enzyme] ─(Compile-Time AD)────────> [.so Binary]      # JIT-compiles the residual. Enzyme differentiates 
                                                                    LLVM IR to emit exact analytical Jacobians and 
                                                                    Reverse-mode VJPs (Vector-Jacobian Products).
                                                      │
[Rust Runtime] ────────(FFI Native Loading)─────> [Solver Handle]   # Maps multi-dimensional Python arrays to flat C-ABI 
                                                                    pointers. Enforces strict memory lifecycles and 
                                                                    bypasses the Python GIL.
                                                      │
[Native Math Solvers] ─(BDF VSVO & GMRES/LU)────> [Time Stepping]   # Integrates stiff non-linear DAEs natively using 
                                                                    adaptive time steps and Newton-Raphson iterations.
                                                      │
                                                      ▼
                                       [Hardware: CPU Execution]    # Orchestrates Task-Parallel batching via Rayon
                                                                    or Data-Parallel OpenMP loops across available vCPUs.
```

---

### **Project Structure: `ion_flux/`**

```text
ion_flux/
├── .github/                        # CI/CD Workflows
│   └── workflows/
│       └── build_and_test.yml      # CI pipeline for testing and Maturin wheel building
├── docs/                           # Documentation
│   ├── API.md                      # V2 Public API Reference Architecture
│   ├── Ion_Flux_project_structure.md # Project architecture and data flow
│   └── Pybamm_project_structure.md # Comparison architecture documentation
├── examples/                       # API usage and execution showcase scripts
│   ├── 0_simple_circuit.py         # 0D ODE equivalent circuit
│   ├── 1_equivalent_circuit.py     # 0D DAE equivalent circuit with thermal coupling
│   ├── 2_macro_micro_spm.py        # 1D-1D Macro-Micro cross-product mesh validation
│   ├── 3_single_particle.py        # 1D Lumped spherical particle validation
│   ├── 4_swelling_spm.py           # ALE moving mesh kinematics (Stefan problem)
│   ├── 5_model_composition.py      # Submodel instantiation and AST merging
│   └── 6_demo.py                   # Rayon SaaS batching, BMS HIL, and Adjoint optimization
├── models/                         # Reference industry implementations
│   └── 1_TSPMe.py                  # Exact asymptotic Thermal Single Particle Model with electrolyte
├── python/                         # Python Source Code (Frontend, Middle-end, & Runtime)
│   └── ion_flux/
│       ├── __init__.py             # Public API exposition
│       ├── metrics.py              # Differentiable loss tracking (RMSE, etc.)
│       ├── battery/                # Pre-built domain models and parameter sets
│       │   ├── __init__.py
│       │   ├── models.py           # SPM, DFN PDE definitions
│       │   └── parameters.py       # Chen2020, Marquis2019 flat parameter dicts
│       ├── compiler/               # Middle-end: C++ Code Generation & FFI Mapping
│       │   ├── __init__.py
│       │   ├── invocation.py       # Subprocesses Clang + Enzyme & loads binaries
│       │   ├── memory.py           # Maps multi-dim spatial states to flat 1D C-arrays
│       │   ├── codegen/            # C++ Emission
│       │   │   ├── __init__.py
│       │   │   ├── ast_analysis.py # AST traversal and State extraction utilities
│       │   │   ├── builder.py      # Orchestrator for IR string emission
│       │   │   ├── templates.py    # C++ execution skeleton and Enzyme boilerplate
│       │   │   └── topology.py     # Hierarchical grid math and spatial strides
│       │   └── passes/             # Compilation transformations
│       │       ├── ir.py           # Intermediate Representation node definitions
│       │       ├── semantic.py     # Boundary condition routing and ALE triggers
│       │       ├── spatial.py      # Discretization, FVM lowering, and dimension unrolling
│       │       └── verification.py # Topological manifold and memory overlap verification
│       ├── dsl/                    # The Researcher API: Mathematical AST Construction
│       │   ├── __init__.py
│       │   ├── core.py             # Public exports for the AST components
│       │   ├── nodes.py            # Base AST node wrappers (Scalar, State, BinaryOp, Piecewise)
│       │   ├── operators.py        # Topology-agnostic math operators (grad, div, dt)
│       │   ├── pde.py              # Base PDE class, Terminal abstraction, and AST compiler
│       │   └── spatial.py          # Domain, CompositeDomain, and Mesh generation
│       ├── protocols/              # Experimental execution instructions
│       │   ├── __init__.py
│       │   └── profiles.py         # CC, CV, Rest, and continuous CurrentProfile logic
│       └── runtime/                # The Engineer API: Execution and Scheduling
│           ├── __init__.py
│           ├── eis.py              # Native Frequency-Domain solver via Enzyme Jacobians
│           ├── engine.py           # Main orchestrator (compiles, routes, and solves)
│           ├── results.py          # Multidimensional array mapping and plotting wrappers
│           ├── scheduler.py        # Async task queueing for multi-tenant constraints
│           ├── session.py          # Stateful HIL/SIL memory session manager
│           └── telemetry.py        # Hardware cache-hit heuristics and sparsity metrics
├── rust/                           # Rust Source Code (Backend: Execution & Linear Algebra)
│   ├── Cargo.toml                  # Rust dependencies (PyO3, rayon, libloading, faer)
│   ├── build.rs                    # Linker script for dynamic SUNDIALS libraries
│   └── src/
│       ├── lib.rs                  # PyO3 extension module entry point
│       └── solver/                 # Highly modular numerical math engine
│           ├── adjoint.rs          # Reverse-time continuous sensitivity tracking via VJPs
│           ├── bindings.rs         # PyO3 FFI definitions and Rayon parallel batching
│           ├── integrator.rs       # Implicit VSVO BDF stepper and truncation error logic
│           ├── linalg.rs           # Sparse LU factorization (faer) and Matrix-Free GMRES
│           ├── mod.rs              # C-ABI signatures, Config, and Diagnostics
│           ├── newton.rs           # Inexact Newton-Raphson root finding and constraint checks
│           ├── session.rs          # FFI pointer management & state lifecycle
│           └── sundials.rs         # Direct C-ABI wrapper for the SUNDIALS IDAS library
├── tests/                          # Automated Test Suite
│   ├── conftest.py                 # Shared mock PDEs
│   ├── 01_frontend_dsl/            # Tests semantic AST capture and component isolation
│   ├── 02_middle_end_codegen/      # Tests C++ spatial lowering, Piecewise logic, and topology
│   ├── 03_backend_compilation/     # Tests Clang invocation, CPR coloring, and Enzyme gradients
│   ├── 04_runtime_execution/       # Tests integrators, ALE, EIS, and stateful sessions
│   ├── 05_e2e_integration/         # Tests full battery models against the Sundials oracle
│   ├── 06_benchmarks/              # Performance tracking
│   └── bugfixes/                   # Strict mathematical oracles preventing regression
├── pyproject.toml                  # Maturin build configuration
└── README.md                       # Core concepts and Quick Start guide
```