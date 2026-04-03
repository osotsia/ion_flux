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
[Python DSL] ──────────(Operator Overloading)──> [AST Payload]      # Mathematical intent captured as a pure JSON/Dict semantic 
                                                                    graph. No execution occurs here.
                                                      │
[Middle-end Codegen] ──(Staggered FVM Lowering)─> [C++ Source]      # Translates AST to C++. Auto-stitches piecewise regions,
                                                                    injects ALE moving mesh kinematics, and enforces 
                                                                    mass-conserving Face/Volume mappings.
                                                      │
[Clang/LLVM + Enzyme] ─(Compile-Time AD)────────> [.so Binary]      # JIT-compiles the residual. Enzyme differentiates LLVM IR
                                                                    to emit exact analytical Jacobians and Reverse-mode
                                                                    VJPs (Vector-Jacobian Products).
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
│   └── project_structure.md        # This file
├── python/                         # Python Source Code (Frontend & Middle-end)
│   └── ion_flux/
│       ├── __init__.py             # Public API exposition
│       ├── dsl/                    # The Researcher API: Mathematical AST Construction
│       │   ├── core.py             # PDE, State, Parameter, Domain, Condition base classes
│       │   ├── operators.py        # Topology-agnostic math operators (grad, div, dt, etc.)
│       │   └── ast.py              # AST payload validation
│       ├── compiler/               # Middle-end: C++ Code Generation & Memory Mapping
│       │   ├── codegen/            # Modular C++ Emission Engine
│       │   │   ├── __init__.py
│       │   │   ├── builder.py      # Orchestrator sorting boundaries, PDEs, and DAEs
│       │   │   ├── translator.py   # Recursive dynamic dispatcher for AST -> C++
│       │   │   ├── topology.py     # Hierarchical grid math, dimensions, and strides
│       │   │   ├── ale.py          # Moving mesh (ALE) advection velocity injection
│       │   │   ├── ast_analysis.py # AST traversal and State extraction utilities
│       │   │   └── templates.py    # C++ execution skeleton and Enzyme boilerplate
│       │   ├── invocation.py       # Subprocesses Clang + Enzyme & loads binaries
│       │   └── memory.py           # Maps multi-dim spatial states to flat 1D C-arrays
│       ├── runtime/                # The Engineer API: Execution and Scheduling
│       │   ├── engine.py           # Main orchestrator (compiles, routes, and solves)
│       │   ├── session.py          # Stateful HIL/SIL memory session manager
│       │   └── scheduler.py        # Async task queueing for multi-tenant constraints
│       ├── battery/                # Domain Library: Pre-built models
│       │   ├── models.py           # DFN, SPM pre-packaged PDE definitions
│       │   └── parameters.py       # Chen2020, Marquis2019 flat parameter dicts
│       ├── protocols/              # Experimental execution instructions
│       │   └── profiles.py         # CCCV logic, Sequences, Custom step profiles
│       └── metrics.py              # Differentiable Loss tracking (Triggers Adjoint Reverse-AD)
├── rust/                           # Rust Source Code (Backend: Execution & Linear Algebra)
│   ├── Cargo.toml                  # Rust dependencies (PyO3, rayon, libloading)
│   ├── build.rs                    # Linker script for dynamic libraries
│   └── src/
│       ├── lib.rs                  # PyO3 extension module entry point
│       └── solver/                 # Highly modular numerical math engine
│           ├── mod.rs              # C-ABI signatures and module exports
│           ├── bindings.rs         # PyO3 Python-to-Rust type conversion & Rayon batching
│           ├── integrator.rs       # Implicit BDF1 stepper with adaptive tolerances
│           ├── linalg.rs           # O(N^3) Dense & O(N*bw^2) Banded Gaussian solvers
│           ├── adjoint.rs          # Reverse-time continuous sensitivity tracking via VJPs
│           └── session.rs          # FFI pointer management & state lifecycle (SolverHandle)
├── tests/                          # Automated Test Suite (100% Passing)
│   ├── conftest.py                 # Shared mock models (TransientHeatPDE, CoupledDAE)
│   ├── benchmarks/                 # Pytest-benchmark suites for JIT vs Execution speed
│   ├── compiler/                   # Tests AST translation, Adjoint derivatives, and Topology
│   ├── dsl/                        # Operator overloading contract tests
│   ├── execution/                  # E2E solver and complex industry model integration tests
│   └── library/                    # Parameter injection parsing tests
├── pyproject.toml                  # Maturin build configuration
└── README.md                       # Core concepts and Quick Start guide
```