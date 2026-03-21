# ion_flux: The End-to-End Differentiable Engine for Battery Physics

![CI Status](https://github.com/organization/ion_flux/actions/workflows/build_and_test.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ion-flux.svg)](https://badge.fury.io/py/ion-flux)

`ion_flux` is a next-generation execution engine for battery simulations, built on a pure Python Abstract Syntax Tree (AST) compiler, a custom native implicit solver, and Enzyme Automatic Differentiation (AD). 

Designed with a philosophy of **zero boilerplate and infinite scalability**, `ion_flux` ruthlessly separates *Physical Intent* from *Computational Execution*. Researchers declare physics using topology-agnostic math; the engine automatically deduces how to difference, linearize, compile, and solve them across 1D grids, massive 3D unstructured meshes, or scalable cloud infrastructure.

## Quick Start (MacOS)
```bash
# ion_flux relies on LLVM and Enzyme for compilation and automatic differentiation.
brew install llvm enzyme sundials cmake

# Clone the repository
git clone https://github.com/organization/ion_flux.git
cd ion_flux

# Create a strict Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install maturin pytest pytest-asyncio numpy scipy pandas matplotlib

# Install the package
pip install ion-flux

# Run the examples
python examples/1_swelling_electrode.py
```

## Key Paradigm Shifts

*   **Topology-Agnostic Math:** Spatial operators (`fx.grad`, `fx.div`) dynamically adapt. The exact same equation syntax compiles to a tridiagonal matrix on a 1D line or a massive sparse stiffness matrix on a 3D pouch cell mesh.
*   **No "Math Gymnastics":** Complex phenomena are native syntax. Cross-product pseudo-dimensions are just `macro * micro`. Moving boundaries (Stefan problems) are simply `domain.right = state`. 
*   **Differentiable by Default:** The custom native solver doesn't just push state forward; it pushes exact gradients backward. Extract analytical Jacobians or compute full-trajectory adjoint sensitivities (`loss.backward()`) for free.
*   **Stateful Co-Simulation:** Microsecond-latency `session.step(dt)` allows for real-time Hardware-in-the-Loop (HIL) and Battery Management System (BMS) testing without recompiling or reallocating solver memory.
*   **Cloud-Native SaaS Scale:** Export compiled physics to portable `.so` binaries for 0ms serverless cold-starts.

---

## Core Concepts

### 1. The Researcher View: Declarative Physics & Moving Boundaries

Physics are defined by inheriting from `fx.PDE`. The `math()` method traces the AST of the system. 

Here is how easily `ion_flux` handles a historically complex problem: a macro-scale electrode coupled to a micro-scale particle, complete with physical swelling (a moving boundary).

```python
import ion_flux as fx

class SwellingElectrode(fx.PDE):
    # 1. Topology: Cross-Product Domains
    # 'macro_micro' places a spherical particle (r) at every node in the electrode (x)
    x = fx.Domain(bounds=(0, 100e-6), resolution=20)
    r = fx.Domain(bounds=(0, 5e-6), resolution=15, coord_sys="spherical")
    macro_micro = x * r 
    
    # 2. States
    c_s = fx.State(domain=macro_micro)  # Solid concentration (2D pseudo-domain)
    thickness = fx.State(domain=None)   # Scalar tracking the expanding electrode
    
    def math(self):
        # 3. Dimensional Reduction
        # Integrate the micro-scale (r) to feed information to the macro-scale (x)
        average_soc = fx.integral(self.c_s, over=self.r)
        
        return {
            # 4. Moving Boundaries
            # Define the physical expansion rate as an ODE
            fx.dt(self.thickness): 1e-9 * average_soc,
            
            # Dynamically bind the right edge of the 'x' domain to the expanding state.
            # The compiler automatically handles mesh deformation and ALE advection!
            self.x.right: self.thickness, 
            
            # ... standard transport PDEs (grad/div adapt automatically) ...
        }
```

### 2. The Engineer View: Co-Simulation & Multi-Mode Protocols

The compiled `Engine` handles execution. For Software-in-the-Loop (SIL), V2 uses a **Stateful Session Generator**. The compiled sparse matrices stay "hot" in the target hardware's memory. Only scalar inputs cross the Python boundary, enabling microsecond-latency control loops.

```python
engine = fx.Engine(model=FullCell(), target="cuda:0")

# Start a hot session on the GPU
session = engine.start_session(parameters={"D": 2e-14})

# Real-time control loop (e.g., 100ms BMS cycle)
while session.time < 3600:
    # 1. Read from external BMS logic
    i_req = my_bms.compute_current(v_cell=session.get("Voltage"))
    
    # 2. Step the native implicit solver forward. 
    # High-order integration history is seamlessly preserved.
    session.step(dt=0.1, inputs={"i_app": i_req})
    
    # 3. Arbitrary event catching
    if session.triggered("Lithium Plating"):
        break
```

For standard lab testing (e.g., CCCV), engineers supply a compiled state-machine (`fx.protocols.Sequence`). The native solver dynamically hot-swaps the differential and algebraic constraints (e.g., Current Control to Voltage Control) on the fly without halting.

### 3. Advanced Analytics: Differentiability & EIS

Because the Python compiler uses Enzyme AD natively at the LLVM level, the entire execution loop is a differentiable computational graph.

#### End-to-End Differentiable Simulation
Optimize charging protocols or empirical parameters using exact, continuous adjoint gradients extracted directly from the native solver.

```python
# Pass `requires_grad` to track sensitivities through the integration trajectory
result = engine.solve(protocol=fast_charge, requires_grad=["porosity", "D"])

# Compute a loss function against experimental lab data
loss = fx.metrics.rmse(result["Voltage"], lab_data)

# Backpropagate through the entire implicit solver!
loss.backward()

# Access exact gradients for gradient descent / Adam optimizers
print(engine.parameters["porosity"].grad) 
```

#### Native Frequency-Domain Solvers (EIS)
Simulating Electrochemical Impedance Spectroscopy (EIS) no longer requires complex AC perturbation math. The engine extracts the analytical Jacobian at steady-state and solves the transfer function algebraically in milliseconds.

```python
import numpy as np

session = engine.start_session(soc=0.5)
session.reach_steady_state()

frequencies = np.logspace(-3, 5, 100)
eis_spectrum = session.solve_eis(frequencies=frequencies, input="i_app", output="Voltage")
```

### 4. DevOps & Cloud Scale: JIT Caching & Hardware Targets

In a production environment (e.g., a Battery Digital Twin SaaS), compilation is an expensive bottleneck. `ion_flux` allows you to export compiled physics to portable binaries for stateless, zero-start deployment.

```python
# --- 1. Compile on your CI/CD Pipeline ---
engine = fx.Engine(model=FullCell())
engine.export_binary("models/dfn_prod_v1.so")

# --- 2. Load instantly on a Serverless Worker (e.g., AWS Lambda / FastAPI) ---
engine = fx.Engine.load("models/dfn_prod_v1.so", target="cpu:serial")

async def simulate_endpoint(payload: dict):
    # Stateless, thread-safe execution perfect for concurrent web requests
    return await engine.solve_async(parameters=payload["params"])
```

Hardware targeting is explicit and declarative:
*   `target="cpu:serial"`: Emits scalar C++. Ideal for task parallelism (thousands of independent models for MCMC sweeps or fleet monitoring) via `engine.solve_batch()`.
*   `target="cpu:omp"`: Identifies spatial domains in the AST and generates OpenMP loops. Ideal for data parallelism (massive 3D unstructured meshes) distributing sparse linear algebra across dozens of cores.
*   `target="cuda:0"`: Emits PTX kernels. The native implicit solver runs entirely on the GPU, keeping all memory in VRAM during the integration loop.

---

## Frequently Asked Questions

**Q1: Why did `ion_flux` drop SUNDIALS in favor of a custom native implicit solver?**

**A:** SUNDIALS is an incredible, battle-tested library, but its monolithic C architecture restricts deep control over the integration loop. By building a custom native solver tightly coupled to our AST, we unlocked three critical capabilities:
1.  **Seamless BMS Co-Simulation:** We can "pause" the solver, yield to Python (`session.step(dt)`), and resume without destroying the history vectors required for high-order BDF integration.
2.  **Hot-Swapping Equations:** We can dynamically swap algebraic constraints (e.g., switching from Constant Current to Constant Voltage) mid-solve without re-initializing the Jacobian sparsity pattern.
3.  **End-to-End AD:** Integrating Enzyme AD directly into our own solver loops allows us to natively generate analytical Jacobians for Newton-Raphson steps *and* continuous adjoints for `loss.backward()`.

**Q2: How does the "Topology-Agnostic Math" actually work under the hood?**

**A:** When you write `fx.grad(T)`, you are simply placing a node in the Python AST. The actual mathematics are not finalized until you instantiate the `Engine(model, target=...)`. During compilation, the Python compiler inspects the `Domain` assigned to `T`. If it is a 1D spherical domain, it lowers `fx.grad` into a finite-volume tridiagonal stencil. If the domain was loaded via `Domain.from_mesh("3d_cell.msh")`, it lowers the exact same AST node into a 3D unstructured finite-element stiffness matrix assembly routine before passing the Intermediate Representation (IR) to LLVM.

**Q3: How does Enzyme Automatic Differentiation (AD) interact with stiff PDEs?**

**A:** Traditional autodiff frameworks (like JAX or PyTorch) build massive computational graphs in memory, which scales poorly for stiff PDEs requiring thousands of implicit time steps. Enzyme operates at the LLVM IR level, performing AD *after* the code is optimized. `ion_flux` uses Enzyme in two passes:
1.  **Forward-mode:** To automatically generate exact, analytical Jacobians for the implicit solver's Newton iterations (guaranteeing quadratic convergence).
2.  **Reverse-mode (Adjoint):** To trace sensitivities backward through the time-stepping trajectory, allowing you to compute the gradient of a loss function with respect to any parameter without storing the entire forward memory state.

**Q4: How does the compiler handle Moving Boundaries (ALE) without severe performance penalties?**

**A:** When you bind a state to a domain boundary (`self.x.right: self.thickness`), the AST compiler automatically injects an Arbitrary Lagrangian-Eulerian (ALE) grid velocity term into your transport equations (e.g., an extra advection term proportional to the moving boundary velocity). Because the native solver and the AST are tightly coupled, the Jacobian sparsity pattern remains constant; only the matrix coefficients change as the mesh stretches, ensuring the linear solver remains highly performant.