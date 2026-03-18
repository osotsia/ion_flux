### **Core Principles Embodied in This Structure**

*   **Architecture Pattern:** The system follows a strict "Compiler-and-Runtime" architecture tailored for Just-In-Time (JIT) execution of PDEs on heterogeneous hardware (CPU/OpenMP/CUDA).
    *   **Frontend (The DSL):** A declarative Python interface mapping mathematical operations to an Abstract Syntax Tree (AST) without executing them.
    *   **Middle-end (The Compiler):** Rust traverses the serialized AST, performs symbolic differentiation to derive the analytical Jacobian, and lowers it to an Intermediate Representation (IR).
    *   **Backend (The Emitter):** Rust emits native C++ (for CPU/OpenMP) or PTX/CUDA C++ (for GPUs) and invokes the local toolchain (`clang++` or `nvcc`) at runtime.
    *   **Runtime (The Engine):** Rust loads the compiled binary via FFI, allocates hardware-specific memory (enforcing RAII), and orchestrates the SUNDIALS IDA integration loop.
*   **Clear Boundaries:** The `Engine` serves as the explicit boundary between mathematical declaration (cheap) and JIT compilation/execution (expensive).
*   **Persona-Driven:** The `ion_flux/dsl/` is optimized for Battery Researchers authoring custom physics. The `ion_flux/runtime/` and `ion_flux/battery/` are optimized for SaaS Engineers orchestrating high-throughput, multi-tenant workloads.

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
│       │   ├── operators.py        # Overloaded math (grad, div, dt, min, max)
│       │   └── ast.py              # Serialization of expressions to Protobuf/Dict
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
│   ├── build.rs                    # Linker script for SUNDIALS and CUDA
│   └── src/
│       ├── lib.rs                  # PyO3 module registration
│       ├── bindings/               # Python-to-Rust deserialization
│       │   └── ast_bridge.rs       # Rebuilds the AST in Rust from Python payload
│       ├── compiler/               # JIT Compilation Pipeline
│       │   ├── mod.rs
│       │   ├── symbolic.rs         # CAS: Generates analytical Jacobian DAG
│       │   ├── codegen_cpu.rs      # Emits C++ / OpenMP pragmas
│       │   ├── codegen_gpu.rs      # Emits CUDA C++ / PTX
│       │   └── invoke.rs           # Subprocesses gcc/clang/nvcc & caches binaries
│       ├── runtime/                # Memory Management
│       │   ├── mod.rs
│       │   ├── mem_cpu.rs          # Host memory arrays
│       │   └── mem_gpu.rs          # Device memory wrappers (RAII cudaMalloc/Free)
│       └── solver/                 # SUNDIALS IDA Orchestration
│           ├── mod.rs
│           ├── ida_serial.rs       # KLU sparse direct solver (Task Parallelism)
│           ├── ida_openmp.rs       # Threaded iterative solvers (Data Parallelism)
│           └── ida_cuda.rs         # cuSOLVER sparse direct (GPU execution)
├── static_kernels/                 # Pre-written C++/CUDA headers injected during JIT
│   ├── stencils.hpp                # Finite-volume spatial discretization stencils
│   └── sundials_c_abi.hpp          # Boilerplate C-callbacks required by IDA
├── tests/                          # Automated Test Suite
│   ├── conftest.py                 # Shared fixtures (TransientHeatPDE, CoupledDAE)
│   ├── benchmarks/
│   │   └── test_performance.py    # Pytest-benchmark suite
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
```
