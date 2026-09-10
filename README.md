# ion_flux: The End-to-End Differentiable Engine for Battery Physics

![CI Status](https://github.com/osotsia/ion_flux/actions/workflows/build_and_test_and_publish.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ion-flux.svg)](https://badge.fury.io/py/ion-flux)

`ion_flux` is a next-generation execution engine for battery simulations. 

It is built on a radical philosophy: **Ruthlessly separate *Physical Intent* from *Computational Execution*.** 

You write physics in pure, declarative Python almost exactly as they appear in a textbook. Behind the scenes, the engine automatically figures out how to discretize, differentiate, compile, and solve them using a native Rust orchestrator, Ahead-of-Time (AOT) LLVM compilation, and Enzyme Automatic Differentiation (AD).

No flattened arrays. No massive Python memory graphs. No "math gymnastics." Just zero boilerplate and infinite scalability.

---

## 🚀 Quick Start

Because `ion_flux` leverages Ahead-of-Time (AOT) compilation and LLVM-level Automatic Differentiation, you must fetch and build its hermetic C++ toolchain locally after installing the Python package.

**1. Install and Configure `ion_flux`**
```bash
# Create a virtual environment (recommended)
# Requires python 3.10–3.14
mkdir my_battery_project && cd my_battery_project
python -m venv .venv && source .venv/bin/activate

# Install the package
pip install ion_flux

# Fetch LLVM and compile the Enzyme AD plugin
ion-flux install-toolchain
```
**2. Get Started**

```bash
# Clone the repository to access the examples and reference models
git clone https://github.com/osotsia/ion_flux.git
cd ion_flux

# Run the performance showcase
python examples/6_demo.py

# Run a full DFN model
python models/Chen2020_DFN.py

# Optimize the same model to recover parameters
python examples/7_Gitt_inversion_demo.py

# Run sensitivity analyses on the same model
python examples/9_sensitivity_analysis.py
```

---

## 🔬 For Researchers: Declarative Physics & Infinite Modularity

### 1. Topology-Agnostic Operators
Mathematical operators like `fx.grad` and `fx.div` dynamically adapt to their domain. The exact same Python syntax compiles to a tridiagonal banded matrix on a 1D line, a spherical finite-volume stencil, or a massive unstructured 3D CSR graph traversal.

### 2. The "Lego" Approach to Battery Modeling
Build highly reusable libraries of isolated physical mechanisms. When instantiated inside a parent cell, the compiler automatically deep-copies the AST, safely namespaces variables (e.g., `anode_c_s`), and merges the computational graphs.

```python
import ion_flux as fx

class FickianParticle(fx.PDE):
    """A strictly isolated, reusable submodel for solid transport."""
    r = fx.Domain(bounds=(0, 5e-6), resolution=15, coord_sys="spherical")
    c_s = fx.State(domain=r)
    D_s = fx.Parameter(default=1e-14)
    
    def math(self, external_flux: fx.Node):
        # fx.grad natively applies spherical coordinate scaling (r^2)
        flux = -self.D_s * fx.grad(self.c_s, axis=self.r)
        
        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(flux, axis=self.r)
            },
            "boundaries": {
                # Map Neumann flux perfectly to the particle surface
                flux: {"left": 0.0, "right": external_flux}
            },
            "initial_conditions": {self.c_s: 500.0}
        }

class ModularSPM(fx.PDE):
    """Composes a full cell using instantiated submodels."""
    anode = FickianParticle()
    cathode = FickianParticle()
    
    V_cell = fx.State(domain=None)
    i_app = fx.State(domain=None)
    
    # Binds these specific states to external cycler protocols (CC, CV, Rest). 
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        j_n = self.i_app / 96485.0
        j_p = -self.i_app / 96485.0
        
        macro_physics = {
            "equations": {
                # Omitting fx.dt() inherently flags this as a Spatial DAE!
                self.V_cell: self.V_cell == (4.2 - 0.001 * self.cathode.c_s.right) \
                                          - (0.1 - 0.001 * self.anode.c_s.right)
            },
            "boundaries": {},
            "initial_conditions": {
                self.V_cell: 4.1, 
                self.i_app: 0.0
            }
        }
        
        # Seamlessly merge all sub-graphs into one unified implicit solve
        return fx.merge(
            macro_physics, 
            self.anode.math(j_n), 
            self.cathode.math(j_p)
        )

# Compile the unified physics and execute a 1-hour simulation
engine = fx.Engine(model=ModularSPM())
result = engine.solve(t_span=(0, 3600))

# Export the compiled native C++ binary for zero-overhead deployments
engine.export_binary("models/spm_prod.so")
```

### 3. Native Multi-Scale Meshes & DAEs
Want to build a Doyle-Fuller-Newman (DFN) model? Just multiply domains together (`macro_micro = x * r`). `ion_flux` automatically logically places a full 1D spherical mesh at *every* node in the macroscopic `x` mesh, flattening them into highly efficient C-arrays automatically.

---

## ⚙️ For Engineers: Cloud Scale, Adjoints, & Co-Simulation

Because `ion_flux` relies on **Ahead-of-Time (AOT) C++ binaries** and strictly scopes execution in **Rust**, it brings systems-level performance to battery math.

### Scenario A: Zero Cold-Start Cloud Batching (Bypass the GIL)
Deploy to serverless environments (AWS Lambda, APIs) by exporting your compiled physics to a portable `.so` shared library. Instantiation takes `0ms`. 

```python
# 1. Load a compiled model instantly on a serverless worker (Bypasses Python AST/Clang entirely)
stateless_engine = fx.Engine.load("models/spm_prod.so", target="cpu:serial")

# 2. Distribute 10,000 implicit solves across all available vCPUs
# Drops into Rust's Rayon thread-pool, completely bypassing the Python GIL.
fleet_size = 10_000

param_payloads = [
    {"anode_D_s": p} 
    for p in np.linspace(1e-14, 5e-14, fleet_size)
]

results = stateless_engine.solve_batch(
    parameters=param_payloads, 
    t_span=(0, 3600), 
    max_workers=64
)
```

### Scenario B: Exact Analytical Adjoint Sensitivities
`ion_flux` does not build massive, memory-hogging computation graphs. It uses the **Enzyme LLVM plugin** to differentiate the generated C++ natively. This enables continuous, reverse-mode Vector-Jacobian Products (VJPs) that scale to thousands of stiff time-steps.

```python
# Forward pass: Record the highly non-linear integration trajectory in C++ RAM
res = engine.solve(protocol=fast_charge, requires_grad=["anode_D_s"])

# Compute a differentiable loss metric against experimental lab data
loss = fx.metrics.rmse(predicted=res["V_cell"], target=experimental_csv)

# Reverse pass: Exact, continuous analytical gradients
grads = loss.backward()

# Hand the exact gradients directly to L-BFGS-B or Adam optimizers
print(grads["anode_D_s"]) 
```

### Scenario C: Microsecond-Latency Hardware-in-the-Loop (HIL)
For Real-Time Software-in-the-Loop (SIL) or BMS testing, use a **Stateful Session**. The BDF history vectors, sparse LU matrices, and exact analytical Jacobians stay "hot" in hardware memory.

```python
session = engine.start_session(parameters={"anode_D_s": 1e-14})

while session.time < 3600.0:
    current_v = session.get("V_cell")
    
    # Evaluate external C++ / Python BMS logic
    i_req = custom_bms.compute_current(v_cell=current_v)
    
    # Advance the native solver by 10ms. 
    # Dynamically toggling `_term_mode` (CC vs CV) hot-swaps the underlying 
    # multiplexer Jacobian natively, without re-initializing the solver!
    session.step(dt=0.01, inputs={"_term_i_target": i_req, "_term_mode": 1.0})
    
    if session.triggered(fx.Condition(model.V_cell > 4.25)):
        print("BMS Safety Halt Triggered!")
        break
```

---
## 🧠 How it Works Under the Hood

`ion_flux` replaces the traditional symbolic-to-numeric computation graph paradigm with a strictly staged Compiler-and-Runtime architecture structured as a unidirectional lowering pipeline: $1 \rightarrow 2 \rightarrow 3 \rightarrow 4$.

**Stage 1: Declarative Intent Capture (Frontend)**
*   **Elements:** Pure Python DSL (`compiler/_1_frontend`: `PDE`, `State`, `Parameter`, `Domain`, `grad`, `div`, etc.).
*   **How:** Operator overloading intercepts mathematical syntax to construct an immutable Abstract Syntax Tree (AST). No geometry, discretization, or numerical execution occurs here.
*   **Why:** Ruthlessly separates physical intent from computational execution. Researchers specify continuum physics without hardcoding loop indices, memory strides, coordinate transformations, or boundary boilerplate.

**Stage 2: Continuum Topology & Semantic Normalization (Middle-end)**
*   **Elements:** Topological Analysis & Math IR (`compiler/_2_middle_end` and `compiler/_3_backend/math_ir.py`).
*   **How:** Analyzes domain manifolds, composite cross-products ($x \times r$), and boundary constraints. The `NormalizationPass` unrolls syntactic sugar (e.g., piecewise regional equations) and resolves continuum boundary constraints, lowering the untyped AST into a strongly typed, N-Dimensional **Math IR** (`MathSystem`).
*   **Why:** Isolates physical mathematics and boundary conditions in a structured, topology-aware representation before geometric discretization or memory indexing is introduced.

**Stage 3: FVM Discretization, Static AD Analysis & Codegen (Backend)**
*   **Elements:** FVM Discretizer, CPR Sparsity Optimizer, and Clang/Enzyme Invoker (`compiler/_3_backend` and `compiler/_4_codegen`).
*   **How:** `FVMDiscretizer` lowers Math IR into **Compute IR**—linearizing N-dimensional coordinates into 1D memory strides (`idx`), applying Finite Volume Method (FVM) stencils (Cartesian, Cylindrical, Spherical, and Unstructured CSR), auto-stitching piecewise interfaces via harmonic mean flux matching, and injecting Arbitrary Lagrangian-Eulerian (ALE) grid kinematics. Static graph coloring (Curtis-Powell-Reid) analyzes the Compute IR to schedule Jacobian evaluation sweeps. `CppEmitter` stringifies Compute IR to C++, and `NativeCompiler` subprocesses Clang with the Enzyme LLVM plugin to synthesize exact analytical Forward Jacobians (JVP) and Reverse Vector-Jacobian Products (VJP) Ahead-of-Time into a `.so` binary.
*   **Why:** Completely eliminates runtime Python graph traversal. Contiguous 1D C-arrays ensure cache locality and SIMD vectorization, while compile-time Enzyme AD eliminates out-of-memory crashes inherent to tape-based symbolic AD.

**Stage 4: FFI Boundary & Native Implicit Solver (Runtime)**
*   **Elements:** Rust Implicit Solver (`runtime/` & `rust/src/solver/`: `_0_ffi` $\rightarrow$ `_1_orchestrator` $\rightarrow$ `_2_stepper` $\rightarrow$ `_3_nonlinear` $\rightarrow$ `_4_linear`).
*   **How:** Python packs parameters and initial conditions into flat C-ABI pointers. The native Rust solver loads the compiled `.so` binary. It integrates stiff non-linear Differential-Algebraic Equations (DAEs) by passing a pre-allocated memory arena (`Workspace`) down a layered call stack (BDF stepper $\rightarrow$ Newton-Raphson $\rightarrow$ Faer Sparse LU / Matrix-Free GMRES), mutating state without hot-loop heap reallocations.
*   **Why:** Completely bypasses the Python Global Interpreter Lock (GIL). Zero-allocation hot loops sustain microsecond-latency control loops for Hardware-in-the-Loop (HIL) testing and unlock linear task-parallel CPU scaling across vCPUs via Rayon.

For more details, see the [Project Structure Documentation](/docs/Ion_Flux_project_structure.md).

## ✅ Testing & Verification

The framework is verified through an automated CI pipeline running across macOS and Linux (Python 3.10–3.14). The testing architecture relies on the following methodologies:

*   **Literature Validation:** We reproduce discharge curves, temperature profiles, etc from benchmark battery literature (e.g., the LG M50 parameterizations by Chen *et al.* 2020 and O'Regan *et al.* 2022, and asymptotic reductions by Brosa Planella *et al.* 2021).
*   **Oracle Validation:** The LLNL SUNDIALS suite (IDA) is compiled and embedded as a reference C-ABI oracle. The custom Rust implicit solver (BDF/Newton-Raphson/Sparse LU) is checked against SUNDIALS.
*   **Method of Manufactured Solutions (MMS):** Spatial FVM discretization, topological mapping, and dynamic Arbitrary Lagrangian-Eulerian (ALE) kinematics are verified on PDEs with known analytical solutions.
*   **Adjoint Exactness:** Compile-time Enzyme Automatic Differentiation sensitivities (VJPs) are tested against central finite-difference perturbations.
