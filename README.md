# ion_flux: The End-to-End Differentiable Engine for Battery Physics

![CI Status](https://github.com/osotsia/ion_flux/actions/workflows/build_and_test_and_publish.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ion-flux.svg)](https://badge.fury.io/py/ion-flux)

`ion_flux` is a next-generation execution engine for battery simulations. 

It is built on a radical philosophy: **Ruthlessly separate *Physical Intent* from *Computational Execution*.** 

You write physics in pure, declarative Python almost exactly as they appear in a textbook. Behind the scenes, the engine automatically deduces how to discretize, differentiate, compile, and solve them using a native Rust orchestrator, Ahead-of-Time (AOT) LLVM compilation, and Enzyme Automatic Differentiation (AD).

No flattened arrays. No massive Python memory graphs. No "math gymnastics." Just zero boilerplate and infinite scalability.

---

## 🚀 Quick Start

Because `ion_flux` leverages Ahead-of-Time (AOT) compilation and LLVM-level Automatic Differentiation, you must fetch and build its hermetic C++ toolchain locally after installing the Python package.

**1. Install System Build Tools**
```bash
# macOS
brew install cmake ninja sundials

# Ubuntu
sudo apt install cmake ninja-build
```

**2. Install and Configure `ion_flux`**
```bash
# Clone the repository to access the examples and reference models
git clone https://github.com/osotsia/ion_flux.git
cd ion_flux

# Create a virtual environment and install the package
python -m venv .venv && source .venv/bin/activate
pip install ion_flux

# Fetch LLVM and compile the Enzyme AD plugin
ion-flux install-toolchain
```
**3. Get Started**

```bash
# Run the performance showcase
python examples/6_demo.py

# Run a full DFN model
python models/Chen2020_DFN.py
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
            "initial_conditions": {self.V_cell: 4.1, self.i_app: 0.0}
        }
        
        # Seamlessly merge all sub-graphs into one unified implicit solve
        return fx.merge(
            macro_physics, 
            self.anode.math(j_n), 
            self.cathode.math(j_p)
        )
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

# Reverse pass: Exact, continuous analytical gradients in O(1) memory!
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

`ion_flux` discards the traditional "Symbolic-to-Numeric" approach (which builds massive, memory-heavy Python computation graphs) in favor of a **Compiler-and-Runtime** architecture. This pushes the heavy lifting down to the LLVM systems level.

Here is the exact lifecycle of your physics:

1.   **Frontend (The DSL):** Pure Python captures your mathematical intent via operator overloading into a lightweight Abstract Syntax Tree (AST). Because the AST is purely semantic, operators like `fx.grad` and `fx.div` are strictly topology-agnostic. The exact same Python node dynamically adapts to represent a 1D Cartesian finite-volume stencil, a volume-scaled spherical particle, or an unstructured 3D graph traversal depending on context.

2.   **Middle-end (The Translator):** The Python compiler traverses the AST and lowers your continuous PDEs into a discrete, mass-conservative C++ residual using the Finite Volume Method (FVM). It automatically handles the hardest parts of numerical modeling, such as flattening hierarchical macro/micro domains into efficient C-arrays, stitching piecewise interface fluxes, and injecting Arbitrary Lagrangian-Eulerian (ALE) kinematics for moving meshes.

3.   **Backend (The AD Compiler):** The generated C++ is passed to `clang++` equipped with the Enzyme LLVM plugin. Unlike traditional ML frameworks that differentiate slow computation graphs in memory, Enzyme performs Automatic Differentiation on the machine code (IR) *after* LLVM has aggressively optimized it. This Ahead-of-Time (AOT) process statically emits mathematically exact Forward Jacobians and Reverse-Mode Vector-Jacobian Products (VJPs) into a `.so` binary.

4.   **Runtime (The Native Solver):** Python loads the compiled binary via FFI into a memory-safe Rust orchestrator, entirely bypassing the Python Global Interpreter Lock (GIL). Rust drives the stiff implicit integration loop (using adaptive VSVO BDF solvers), feeding Enzyme's exact Jacobians into highly optimized O(N) Banded solvers or Sparse LU / Matrix-Free GMRES backends depending on your mesh complexity.