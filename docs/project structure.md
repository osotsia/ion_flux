### **Core Principles Embodied in This Structure**

*   **Architecture Pattern:** A "Compiler-and-Runtime" architecture leveraging **Compiler-Level Automatic Differentiation (Enzyme/Clad)**. 
    *   **Frontend (The DSL):** Python captures the mathematical intent via operator overloading into an Abstract Syntax Tree (AST). It performs no calculus or execution.
    *   **Middle-end (The Translator):** Rust deserializes the AST and lowers it directly to C++ or CUDA C++. Crucially, it only generates code for the *residual* equation $F(t, y, \dot{y}) = 0$. It does not perform symbolic differentiation or construct a Jacobian DAG.
    *   **Backend (The AD Compiler):** Rust invokes `clang++` (or `nvcc`) equipped with the Enzyme LLVM plugin. Enzyme differentiates the compiled Intermediate Representation (IR) of the residual to automatically generate a highly optimized Jacobian function at compile time. Common Subexpression Elimination (CSE) is handled natively by LLVM's `-O3` passes.
    *   **Runtime (The Engine):** Rust loads the compiled `.so` binary, manages strictly scoped hardware memory (RAII), and orchestrates the SUNDIALS integration loop, passing the Enzyme-generated Jacobian as a C-callback.

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
[Rust Runtime] ──(Loads via FFI)──> [SUNDIALS Solver] ──> [Hardware: CPU/GPU]
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
│   ├── build.rs                    # Linker script for SUNDIALS and LLVM/Clang presence
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




### Environment

For strict isolation or exact version parity with Linux cloud environments, use Conda.

**1. Create the Environment**
```bash
conda create -n ion_flux -c conda-forge python=3.11 clang llvmdev enzyme sundials cmake pkg-config -y
conda activate ion_flux
pip install maturin pytest pytest-asyncio numpy scipy pandas matplotlib pytest-benchmark
```

**2. Verify Plugin Path**
In Conda, the plugin is installed into the environment's `lib` directory.
```bash
# The plugin path your Rust engine must target:
echo $CONDA_PREFIX/lib/ClangEnzyme-*.dylib
```

### Critical Risks and Uncertainties on macOS

1.  **C++ Standard Library Clashes (libc++ vs. libstdc++):**
    macOS uses `libc++` by default. If SUNDIALS was compiled with a different standard library implementation or ABI flag than the one used by your upstream LLVM installation, linking your JIT-compiled `residual.so` against `libsundials_ida.dylib` will result in segmentation faults at runtime.
2.  **Dynamic Library Sandboxing (SIP/Gatekeeper):**
    macOS System Integrity Protection (SIP) aggressively restricts passing environment variables like `DYLD_LIBRARY_PATH` to subprocesses. If your Rust `invoke.rs` relies on passing these variables to `clang++` to locate SUNDIALS during the JIT phase, the command will silently drop the variables and fail to link. Absolute paths must be hardcoded or passed via `-rpath` during emission.
3.  **LLVM Version Pinning:**
    Enzyme is tightly coupled to the major version of LLVM. If Homebrew updates LLVM from 17 to 18, but Enzyme is still compiled against 17, the `-fplugin` flag will yield a symbol resolution error. The CI/CD pipeline must enforce strict version pinning.
