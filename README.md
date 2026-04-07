# ion_flux: The End-to-End Differentiable Engine for Battery Physics

![CI Status](https://github.com/organization/osotsia/ion_flux/actions/workflows/build_and_test.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ion-flux.svg)](https://badge.fury.io/py/ion-flux)

`ion_flux` is a next-generation execution engine for battery simulations. It is built on a pure Python Abstract Syntax Tree (AST) compiler, a custom native Rust implicit solver, and Enzyme Automatic Differentiation (AD). 

Designed with a philosophy of **zero boilerplate and infinite scalability**, `ion_flux` ruthlessly separates *Physical Intent* from *Computational Execution*. Researchers declare physics using topology-agnostic math; the engine automatically deduces how to difference, linearize, compile, and solve them across 1D grids, massive 3D unstructured meshes, or scalable cloud infrastructure.

## Quick Start (MacOS / Linux)
```bash
# ion_flux relies on LLVM and Enzyme for JIT compilation and automatic differentiation.
brew install llvm enzyme cmake

# Clone the repository
git clone https://github.com/organization/ion_flux.git
cd ion_flux

# Create a strict Python environment
python3.11 -m venv .venv
source .venv/bin/activate
# Install Maturin (required to build the Rust backend), plus numerical/data libraries
pip install maturin pytest pytest-asyncio numpy scipy pandas matplotlib

# Compile the Rust backend and Python bindings natively for your hardware
pip install .

# Run the showcase example
python examples/6_demo.py
```

## Key Paradigm Shifts

*   **Topology-Agnostic Math:** Spatial operators (`fx.grad`, `fx.div`) dynamically adapt. The exact same equation syntax compiles to a tridiagonal banded matrix on a 1D line, or a Matrix-Free CSR graph traversal on a 3D pouch cell mesh.
*   **The Hardware Abstraction (`fx.Terminal`):** Completely decouples battery physics from cycler testing logic. The native engine automatically handles the zero-sum structural DAEs required to toggle between Constant Current (CC) and Constant Voltage (CV) mid-solve without re-initializing the Jacobian.
*   **Compile-Time Automatic Differentiation:** `ion_flux` does not build massive Python memory graphs (like JAX/CasADi). The AST generates C++ which is differentiated *Ahead-of-Time* by Enzyme via LLVM. The Rust native solver pushes exact continuous gradients backward (`loss.backward()`) using compiled Vector-Jacobian Products (VJPs) in $O(1)$ memory.
*   **Stateful Co-Simulation:** Microsecond-latency `session.step(dt)` allows for real-time Hardware-in-the-Loop (HIL) and Battery Management System (BMS) testing. The Rust backend keeps the BDF history vectors and factored LU matrices "hot" in RAM.
*   **Cloud-Native SaaS Scale:** Export compiled physics to portable `.so` binaries for 0ms serverless cold-starts. Bypass the Python GIL entirely using Rust's Rayon thread-pool for massive `solve_batch` parameter sweeps.

---

## Core Concepts

### 1. The Researcher View: Declarative Physics & Infinite Modularity

In legacy frameworks, translating a mathematical theory into code requires severe "math gymnastics"—flattening arrays, tracking indices, manually deriving Jacobian sparsities, or fighting numerical instability. 

`ion_flux` eradicates numerical boilerplate. You write equations exactly as they appear in a textbook. The Python AST compiler dynamically maps your physical intent to the correct finite-volume stencils, coordinate geometries, and root-finding solvers. 

#### Highlight A: Hierarchical Model Composition (The "Lego" Approach)
Researchers can build highly reusable libraries of isolated physical mechanisms (e.g., solid diffusion, SEI growth, thermal lumped masses). When instantiated inside a parent cell, the compiler automatically deep-copies the AST, isolates namespaces (preventing variable collisions), and merges the computational graphs.

```python
import ion_flux as fx

class FickianParticle(fx.PDE):
    """A strictly isolated, highly reusable submodel for solid transport."""
    # Topology-agnostic. The math works identically for 1D slabs or 3D spheres.
    r = fx.Domain(bounds=(0, 5e-6), resolution=15, coord_sys="spherical")
    c_s = fx.State(domain=r)
    D_s = fx.Parameter(default=1e-14)
    
    def math(self, j_flux: fx.Node):
        # fx.grad dynamically adapts to the 'spherical' geometric scaling natively
        flux = -self.D_s * fx.grad(self.c_s, axis=self.r)
        
        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(flux, axis=self.r)
            },
            "boundaries": {
                # Neumann tensor boundary injection
                flux: {"left": 0.0, "right": j_flux}
            },
            "initial_conditions": {}
        }

class ModularCell(fx.PDE):
    """Composes the full cell using instantiated submodels."""
    
    # 1. Instantiate submodels. 
    # The framework automatically prefixes states (e.g., 'anode_c_s') in memory.
    anode = FickianParticle()
    cathode = FickianParticle()
    
    V_cell = fx.State(domain=None)
    i_app = fx.State(domain=None)
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        j_n = self.i_app / 96485.0
        j_p = -self.i_app / 96485.0
        
        # Extract specific boundaries cleanly from submodels using `.right`
        macro_physics = {
            "equations": {
                self.V_cell: self.V_cell == (4.2 - 0.001 * self.cathode.c_s.right) \
                                          - (0.1 - 0.001 * self.anode.c_s.right)
            },
            "boundaries": {},
            "initial_conditions": {
                self.V_cell: 4.1, 
                self.i_app: 0.0,
                self.anode.c_s: 800.0,
                self.cathode.c_s: 200.0
            }
        }
        
        # 2. Seamlessly merge all sub-graphs into one unified implicit solve
        return fx.merge(
            macro_physics,
            self.anode.math(j_flux=j_n),
            self.cathode.math(j_flux=j_p)
        )
```

#### Highlight B: Multi-Scale Topologies & Native Spatial DAEs
Modeling macro-micro scale coupling (like the Doyle-Fuller-Newman model) or solving instantaneous algebraic constraints across a spatial mesh usually breaks standard ODE solvers. `ion_flux` treats complex topologies and Differential-Algebraic Equations (DAEs) as native syntax.

*   **Pseudo-Dimensions:** Multiply domains together (`x * r`). The compiler logically places a full 1D spherical mesh at *every* node in the macroscopic `x` mesh, flattening them into highly efficient C-arrays automatically.
*   **Zero-Boilerplate DAEs:** If you omit a `fx.dt()` operator, the compiler inherently flags the equation as a pure Algebraic constraint. The Rust implicit solver seamlessly co-solves it at every spatial node alongside the PDEs.

```python
class MinimalDFN(fx.PDE):
    x_n = fx.Domain(bounds=(0, 40e-6), resolution=10, name="x_n")
    r_n = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_n") 
    
    # Multiply domains to create a hierarchical cross-product mesh
    macro_micro_n = x_n * r_n 
    
    c_s_n = fx.State(domain=macro_micro_n) # 2D Multi-Scale PDE State
    phi_s_n = fx.State(domain=x_n)         # 1D Spatial DAE State
    
    # ... additional state declarations omitted ...
    
    def math(self):
        # The `axis` argument forces operators to differentiate against a 
        # specific spatial dimension within the composite domain.
        i_s_n = -100.0 * fx.grad(self.phi_s_n, axis=self.x_n) 
        N_s_n = -1e-14 * fx.grad(self.c_s_n, axis=self.r_n) 
        
        j_n = 1e6 * (self.phi_s_n - (0.1 - 0.0001 * self.c_s_n.boundary("right", domain=self.r_n))) 
        
        return {
            "equations": {
                # --- Spatial Algebraic Constraints (DAEs) ---
                # Notice `phi_s_n` has no time derivative (fx.dt). The native 
                # compiler automatically routes this to the 0.0 root-finding Jacobian mask.
                self.phi_s_n: fx.div(i_s_n, axis=self.x_n) == -j_n,
                
                # --- Multi-Scale PDEs ---
                # Solves diffusion inside the particle at *every* macroscopic node in x_n simultaneously.
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n)
            },
            # ... boundaries and ICs omitted ...
        }
```

---

### 2. The Engineer View: Cloud Scale, Sensitivities, & Co-Simulation

Legacy symbolic graph frameworks (like CasADi) often break down in production software environments due to massive Python memory footprints, Just-In-Time (JIT) compilation cold-starts, and the Python Global Interpreter Lock (GIL). 

Because `ion_flux` generates Ahead-of-Time (AOT) C++ binaries and strictly scopes execution lifecycles in Rust, it treats cloud-scale SaaS architectures, continuous gradient optimization, and real-time execution as native capabilities.

#### Scenario A: Stateless Fleet-Scale SaaS (Zero Cold-Start)
To deploy models into serverless environments (e.g., AWS Lambda, fast web APIs), you can export the compiled model into a standalone shared object (`.so`) library. Python AST parsing and LLVM compilation are entirely bypassed at runtime.

```python
# --- 1. Compile exactly once on your CI/CD Pipeline ---
compiler_engine = fx.Engine(model=FastSPM(), target="cpu:serial")
compiler_engine.export_binary("models/spm_prod.so")

# --- 2. Instant Serverless Load (0ms Cold Start) ---
# Instantiates instantly without invoking Clang or building memory graphs.
stateless_engine = fx.Engine.load("models/spm_prod.so", target="cpu:serial")

# Create a massive payload of concurrent parameter permutations
fleet_size = 10_000
param_payloads = [
    {"D_s": np.random.uniform(1e-14, 5e-14), "R_internal": np.random.uniform(0.01, 0.05)} 
    for _ in range(fleet_size)
]

# --- 3. Rayon Task-Parallel Execution ---
# Drops into Rust, utilizing the Rayon thread-pool to distribute 10,000 independent 
# implicit integration handles across all vCPUs. 
# It completely bypasses the Python GIL with zero multiprocessing serialization bloat.
results = stateless_engine.solve_batch(
    parameters=param_payloads, 
    t_span=(0, 3600), 
    max_workers=64
)
```

#### Scenario B: Exact Adjoint Sensitivities (Parameter Optimization)
Because the Python compiler applies Enzyme Automatic Differentiation (AD) at the LLVM level, the entire execution loop is a mathematically exact differentiable graph. You can backpropagate gradients through thousands of stiff implicit time-steps in $\mathcal{O}(1)$ memory.

```python
# The `requires_grad` flag instructs the native solver to record the highly non-linear 
# integration trajectory in C++ RAM.
res = engine.solve(protocol=fast_charge_cycle, requires_grad=["D_s", "R_internal"])

# Compute a differentiable loss metric against experimental CSV lab data
loss = fx.metrics.rmse(predicted=res["V_cell"], target=experimental_voltage)

# --- The Magic Happens Here ---
# Triggers the Reverse-Mode Vector-Jacobian Product (VJP) loop through the native Rust solver.
# Exact continuous gradients route perfectly through discrete state-machine bounds 
# (e.g., CCCV phase changes) without diverging like numerical Finite Differences (FD).
grads = loss.backward()

# Hand the exact analytical gradients to L-BFGS-B or Adam optimizers
print(f"Exact dLoss/dDs: {grads['D_s']}") 
print(f"Exact dLoss/dR:  {grads['R_internal']}")
```

#### Scenario C: Microsecond-Latency BMS Hardware-in-the-Loop (HIL)
For Real-Time Software-in-the-Loop (SIL) or HIL, we use a **Stateful Session Generator**. The compiled sparse matrices, Exact Enzyme Jacobians, and BDF integration history vectors stay "hot" in Rust hardware memory. Only 64-bit float inputs cross the Python/Rust FFI boundary.

```python
# Pin the execution session natively.
session = engine.start_session(parameters={"D_s": 1e-14})

# E.g., Executing a 10ms BMS cycle over a 60-second simulated pulse
while session.time < 60.0:
    # 1. Read state directly from the underlying flat C-array with zero overhead
    current_v = session.get("V_cell")
    
    # 2. Evaluate external BMS logic
    # Charge aggressively, but taper current dynamically to prevent 
    # the voltage from exceeding a hard 4.2V safety ceiling.
    i_req = -5.0 if current_v < 4.15 else -max(0.0, 50.0 * (4.2 - current_v))
    
    # 3. Inject dynamic parameters and advance the native solver seamlessly.
    # The `_term_mode` flag dynamically hot-swaps the underlying multiplexer constraint 
    # in the Jacobian matrix natively without re-initializing the solver.
    session.step(dt=0.01, inputs={"_term_i_target": i_req, "_term_mode": 1.0})
    
    # 4. Built-in trigger checks for arbitrary safety aborts
    if session.triggered(model.T_cell > 323.15):
        print("BMS Safety Halt Triggered: Thermal Limit Reached.")
        break
```

---

## Frequently Asked Questions

**Q1: How does the "Topology-Agnostic Math" actually work under the hood?**

**A:** When you write `fx.grad(T)`, you are simply placing a node in the Python AST. During compilation, the Python codegen layer inspects the `Domain` assigned to `T`. If it is a 1D spherical domain, it lowers `fx.grad` into a finite-volume tridiagonal stencil (applying L'Hopital's limits at the origin natively to prevent 0/0 singularities). If the domain was loaded via `Domain.from_mesh("3d_cell.msh")`, it lowers the exact same AST node into a 3D unstructured Matrix-Free CSR graph traversal before passing the C++ code to LLVM.

**Q2: How does the compiler handle Moving Boundaries (ALE) without severe performance penalties?**

**A:** In a Stefan problem (like particle swelling or lithium plating), you bind a state to a domain boundary (`self.x.right == self.L`). The AST compiler detects this and dynamically flags the domain as deformable. It automatically injects an Arbitrary Lagrangian-Eulerian (ALE) grid velocity advection term into any transport equations evaluating over that mesh. It also implements directional upwinding to guarantee advective stability. Because the native solver and the AST are tightly coupled, the Jacobian sparsity pattern remains constant; only the spatial derivatives shift smoothly as the mesh stretches.

**Q3: How does Enzyme Automatic Differentiation (AD) interact with stiff PDEs?**

**A:** Traditional autodiff frameworks (like JAX or PyTorch) build massive computational graphs in memory, which scales poorly for stiff PDEs requiring thousands of implicit time steps. Enzyme operates at the LLVM IR level, performing AD *after* the code is optimized. `ion_flux` uses this Enzyme capability in two passes:
1.  **Forward-mode:** To automatically generate exact, analytical Jacobians for the implicit solver's Newton iterations (guaranteeing quadratic convergence).
2.  **Reverse-mode (Adjoint):** To trace sensitivities backward through the time-stepping trajectory, allowing you to compute the gradient of a loss function with respect to any parameter without storing the entire forward memory state.
