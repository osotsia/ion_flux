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

The directory structure reflects a strict "Compiler-and-Runtime" separation. Each module is designed to isolate the *intent* of the battery physics from the *execution* of the numerical linear algebra, pushing heavy computation down to LLVM and Rust.

```text
ion_flux/
├── .github/                        
│   └── workflows/
│       └── build_and_test.yml      # Automates C++ toolchain caching and Rust FFI wheel building (Maturin). 
│                                   # Prevents manual setup errors and Mach-O linking crashes on release.
├── docs/                           
│   ├── API.md                      
│   └── Ion_Flux_project_structure.md 
├── examples/                       
│   ├── 1_equivalent_circuit.py     
│   ├── 2_macro_micro_spm.py        
│   ├── 3_single_particle.py        
│   ├── 4_swelling_spm.py           
│   ├── 5_model_composition.py      
│   └── 6_demo.py                   # Minimal runnable scripts isolating specific architectural features 
│                                   # (ALE, Macro-Micro, Composition) without the noise of full industry parameter sets.
│
├── models/                         # Full-scale industry implementations. Provides rigorous, peer-reviewed                          
│   │                               # baselines proving the engine scales to highly-coupled real-world battery physics.
│   ├── Brosa2021_TSPMe.py          
│   ├── Chen2020_DFN.py             
│   ├── Marquis2019_1Plus1D_SPMe.py 
│   └── ORegan2022_ThermalDFN.py    
│                                   
├── python/                         
│   └── ion_flux/
│       ├── __init__.py             
│       ├── cli.py                  # Automates hermetic fetching/building of LLVM 19 + Enzyme to ensure 
│       │                           # strict ABI compatibility for Compile-Time Automatic Differentiation (AD).
│       ├── metrics.py              # Bridges Python loss functions to Rust's VJP adjoint solvers for optimization.
│       ├── compiler/               # --- MIDDLE-END ---
│       │   ├── invocation.py       # Subprocesses Clang+Enzyme, statically generating exact analytical Jacobians 
│       │   │                       # into portable `.so` binaries, bypassing Python JIT overhead.
│       │   ├── memory.py           # Flattens hierarchical topologies (e.g., Macro x Micro) into highly 
│       │   │                       # optimized 1D C-arrays for the native solver.
│       │   ├── codegen/            
│       │   │   ├── ast_analysis.py 
│       │   │   ├── builder.py      # Orchestrates the emission of the raw C++ residual/observable skeletons.
│       │   │   ├── templates.py    
│       │   │   └── topology.py     # Resolves dimension strides for C-array multi-dimensional indexing.
│       │   └── passes/             
│       │       ├── discretization.py # Isolates coordinate geometry (Spherical/Cartesian FVM math) from AST traversal.
│       │       ├── ir.py           
│       │       ├── semantic.py     # Pre-processes domain boundary logic into O(1) lookup tables.
│       │       ├── spatial.py      # Transforms abstract mathematical operators into explicit, mass-conserving 
│       │       │                   # Finite Volume Method (FVM) stencils.
│       │       └── verification.py # Defends against topological overlaps, memory overwrites, and missing boundaries.
│       ├── dsl/                    # --- FRONTEND ---
│       │   ├── core.py             
│       │   ├── nodes.py            # Allows researchers to declare physics using operator-overloaded Python 
│       │   │                       # AST nodes, rather than tracking flat array indices manually.
│       │   ├── operators.py        # Provides topology-agnostic math operators (grad, div, dt).
│       │   ├── pde.py              # Manages submodel deepcopying and namespace prefixing during composition.
│       │   └── spatial.py          
│       ├── protocols/              
│       │   ├── profiles.py         # Abstracts CCCV cycling into triggers that the solver can bisect natively.
│       │   └── __init__.py
│       └── runtime/                # --- PYTHON EXECUTION ORCHESTRATION ---
│           ├── eis.py              # Solves Frequency-Domain impedance analytically via Enzyme Mass Matrices.
│           ├── engine.py           # Main interface. Compiles ASTs, routes FFI payloads, and manages multi-threading.
│           ├── results.py          # Wraps flat C-arrays back into multidimensional Python structures for plotting.
│           ├── scheduler.py        
│           ├── session.py          # Keeps native memory and integration history "hot" for micro-stepping HIL.
│           └── telemetry.py        
├── rust/                           # --- NATIVE BACKEND ---
│   ├── Cargo.toml                  
│   ├── build.rs                    # Dynamically links SUNDIALS for the C-ABI oracle wrapper.
│   └── src/
│       ├── lib.rs                  
│       └── solver/                 
│           ├── adjoint.rs          # Tracks continuous sensitivities backwards through time, providing O(1) 
│           │                       # memory exact gradients for stiff trajectories.
│           ├── bindings.rs         # Safely maps Python Numpy arrays to Rust/C pointers. Uses Rayon to 
│           │                       # distribute concurrent battery models across all vCPUs, bypassing the GIL.
│           ├── integrator.rs       # VSVO BDF Stepper. Handles extreme stiffness by dynamically adapting step sizes.
│           ├── linalg.rs           # Dispatches Faer (Sparse LU) or GMRES based on the battery mesh complexity.
│           ├── newton.rs           # Resolves non-linear algebraic roots and enforces state constraints.
│           ├── session.rs          
│           └── sundials.rs         # Battle-tested industry oracle (IDAS) to cross-validate the custom solver.
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
├── pyproject.toml                  # Configures Maturin to build the mixed Python/Rust/C++ ecosystem securely.
└── README.md
```