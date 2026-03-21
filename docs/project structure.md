### **Core Principles Embodied in This Structure**

*   **Architecture Pattern:** A "Compiler-and-Runtime" architecture leveraging **Compiler-Level Automatic Differentiation (Enzyme/Clad)**. 
    *   **Frontend (The DSL):** Python captures the mathematical intent via operator overloading into an Abstract Syntax Tree (AST). It performs no calculus or execution.
    *   **Middle-end (The Translator):** Rust deserializes the AST and lowers it directly to C++ or CUDA C++. Crucially, it only generates code for the *residual* equation $F(t, y, \dot{y}) = 0$. It does not perform symbolic differentiation or construct a Jacobian DAG.
    *   **Backend (The AD Compiler):** Rust invokes `clang++` (or `nvcc`) equipped with the Enzyme LLVM plugin. Enzyme differentiates the compiled Intermediate Representation (IR) of the residual to automatically generate a highly optimized Jacobian function at compile time. Common Subexpression Elimination (CSE) is handled natively by LLVM's `-O3` passes.
    *   **Runtime (The Engine):** Rust loads the compiled `.so` binary, manages strictly scoped hardware memory (RAII), and orchestrates the native implicit solver integration loop, passing the Enzyme-generated Jacobian as a C-callback.

---

### **Execution Pipeline (Data Flow)**

```text
[Python DSL] 
      │ (JSON AST)
      ▼
[Rust Middle-end] ──(Lowers to)──> [C++ Source] (Residual only + Enzyme hooks)
                                          │
                                          ▼
[Clang/LLVM + Enzyme Plugin] ──(Auto-Diff & Optimize)──> [.so Binary] (Residual + Jacobian)
                                          │
                                          ▼
[Rust Runtime] ──(Loads via FFI)──> [Native Solver] ──> [Hardware: CPU/GPU]
```

---

### **Project Structure: `ion_flux/`**

```text
ion_flux/
├── .github/                        # CI/CD Workflows
│   └── workflows/
│       └── build_and_test.yml      # Maturin wheel building and testing matrix
├── docs/                           # Documentation
│   ├── architecture.md
│   ├── researcher_guide.md         # Writing custom PDEs via the DSL
│   └── engineer_guide.md           # Multi-tenant execution and engine routing
├── python/                         # Python source code (Frontend)
│   └── ion_flux/
│       ├── __init__.py             # Public API exposition
│       ├── dsl/                    # The Researcher API: Mathematical AST Construction
│       │   ├── __init__.py
│       │   ├── core.py             # PDE, State, Parameter, Domain base classes
│       │   └── ast.py              # Serialization of expressions to dictionaries
│       ├── battery/                # The Domain Library: Pre-built models
│       │   ├── __init__.py
│       │   ├── models.py           # SPM, DFN classes inheriting from fx.PDE
│       │   └── parameters.py       # Chen2020, Marquis2019 flat dictionaries
│       ├── protocols/              # Experimental conditions
│       │   ├── __init__.py
│       │   └── profiles.py         # ConstantCurrent, CurrentProfile (arrays)
│       └── runtime/                # The Engineer API: Execution and Scheduling
│           ├── __init__.py
│           ├── engine.py           # The Engine class (FFI bridge to Rust)
│           └── scheduler.py        # MultiTenantScheduler (asyncio queueing)
├── rust/                           # Rust source code (Middle-end / Backend)
│   ├── Cargo.toml                  # Rust dependencies (PyO3, rayon, libloading)
│   ├── build.rs                    # Linker script for LLVM/Clang presence
│   └── src/
│       ├── lib.rs                  # PyO3 module registration
│       ├── bindings/               # Python-to-Rust deserialization
│       │   └── ast_bridge.rs       # Rebuilds the AST in Rust from Python payload
│       ├── compiler/               # JIT Compilation Pipeline
│       │   ├── mod.rs
│       │   ├── codegen_cpu.rs      # Emits residual C++ & Enzyme __enzyme_autodiff pragmas
│       │   ├── codegen_gpu.rs      # Emits residual CUDA C++ & Enzyme pragmas
│       │   └── invoke.rs           # Subprocesses Clang + Enzyme plugin & caches binaries
│       ├── runtime/                # Memory Management
│       │   ├── mod.rs
│       │   ├── mem_cpu.rs          # Host memory arrays
│       │   └── mem_gpu.rs          # Device memory wrappers (RAII cudaMalloc/Free)
│       └── solver/                 # Native Solver Orchestration
│           ├── mod.rs
│           └── ida.rs              # Custom native implicit solver logic
├── tests/                          # Automated Test Suite
│   ├── conftest.py                 # Shared fixtures (TransientHeatPDE, CoupledDAE)
│   ├── benchmarks/
│   │   └── test_performance.py    # Pytest-benchmark suite
│   ├── compiler/                   # JIT Compiler and Codegen Tests
│   │   ├── test_codegen_cpu.py
│   │   └── test_invocation.py
│   ├── dsl/                        # DSL Contract Tests
│   │   └── test_declarative_math.py
│   ├── execution/                  # Engine Contract Tests
│   │   ├── test_engine_execution.py
│   │   └── test_multi_tenant.py    # Async scheduler validation
│   └── library/                    # Domain Contract Tests
│       └── test_battery_library.py
├── .gitignore
├── pyproject.toml                  # Maturin build configuration
└── README.md
