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

# Run the showcase DFN example
python examples/6_DFN.py
```

## Key Paradigm Shifts

*   **Topology-Agnostic Math:** Spatial operators (`fx.grad`, `fx.div`) dynamically adapt. The exact same equation syntax compiles to a tridiagonal banded matrix on a 1D line, or a Matrix-Free CSR graph traversal on a 3D pouch cell mesh.
*   **The Hardware Abstraction (`fx.Terminal`):** Completely decouples battery physics from cycler testing logic. The native engine automatically handles the zero-sum structural DAEs required to toggle between Constant Current (CC) and Constant Voltage (CV) mid-solve without re-initializing the Jacobian.
*   **Compile-Time Automatic Differentiation:** `ion_flux` does not build massive Python memory graphs (like JAX/CasADi). The AST generates C++ which is differentiated *Ahead-of-Time* by Enzyme via LLVM. The Rust native solver pushes exact continuous gradients backward (`loss.backward()`) using compiled Vector-Jacobian Products (VJPs) in $O(1)$ memory.
*   **Stateful Co-Simulation:** Microsecond-latency `session.step(dt)` allows for real-time Hardware-in-the-Loop (HIL) and Battery Management System (BMS) testing. The Rust backend keeps the BDF history vectors and factored LU matrices "hot" in RAM.
*   **Cloud-Native SaaS Scale:** Export compiled physics to portable `.so` binaries for 0ms serverless cold-starts. Bypass the Python GIL entirely using Rust's Rayon thread-pool for massive `solve_batch` parameter sweeps.

---

## Core Concepts

### 1. The Researcher View: Declarative Physics

Physics are defined by inheriting from `fx.PDE`. The `math()` method acts as a tracer: it maps your equations to specific spatial regions, boundaries, or 0D global states to build a computational graph.

Here is how `ion_flux` handles the foundational Single Particle Model (SPM). Notice how the code reads exactly like a math textbook, devoid of numerical for-loops or array slicing.

```python
import ion_flux as fx

class SingleParticleModel(fx.PDE):
    # -------------------------------------------------------------------------
    # 1. Topology Declaration
    # Defines the spatial discretization. The Python AST operates independently 
    # of the underlying mesh, making the math highly reusable.
    # -------------------------------------------------------------------------
    r_n = fx.Domain(bounds=(0, 10e-6), resolution=15, coord_sys="spherical")
    r_p = fx.Domain(bounds=(0, 10e-6), resolution=15, coord_sys="spherical")

    # -------------------------------------------------------------------------
    # 2. State Variable Registration
    # These are the unknowns targeted by the implicit solver. 
    # States bound to a `Domain` compile into spatial C-arrays; 
    # unbound states (None) compile as 0D scalars.
    # -------------------------------------------------------------------------
    c_s_n = fx.State(domain=r_n)
    c_s_p = fx.State(domain=r_p)
    V_cell = fx.State(domain=None) 
    i_app = fx.State(domain=None)  

    # -------------------------------------------------------------------------
    # 3. Hardware Abstraction (The Cycler Terminal)
    # Explicitly decouples internal physics from the cycler testing logic. 
    # This prevents the user from having to manually cancel out algebraic terms 
    # to switch between Current Control and Voltage Control.
    # -------------------------------------------------------------------------
    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    # -------------------------------------------------------------------------
    # 4. Parameters
    # Constants dynamically mapped into the C-ABI memory buffer.
    # They can be overridden at runtime without triggering LLVM recompilation.
    # -------------------------------------------------------------------------
    Ds_n = fx.Parameter(default=1e-14)
    Ds_p = fx.Parameter(default=1e-14)

    def math(self):
        # Topology-agnostic operators. The AST compiler automatically applies
        # the correct spherical finite-volume Jacobian transformations here.
        flux_n = -self.Ds_n * fx.grad(self.c_s_n, axis=self.r_n)
        flux_p = -self.Ds_p * fx.grad(self.c_s_p, axis=self.r_p)

        # Extract specific boundaries cleanly using the `.boundary()` method
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n)
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p)

        # Faraday's flux conversion (assuming specific area was parameterized)
        j_flux = self.i_app / 96485.0

        return {
            # --- Regional Physics (PDEs) ---
            # Maps bulk governing equations to specific spatial meshes.
            "regions": {
                self.r_n: [ fx.dt(self.c_s_n) == -fx.div(flux_n, axis=self.r_n) ],
                self.r_p: [ fx.dt(self.c_s_p) == -fx.div(flux_p, axis=self.r_p) ]
            },
            
            # --- Spatial Boundaries ---
            # Dictates local Dirichlet overrides or Neumann flux injections.
            "boundaries": [
                flux_n.left == 0.0, flux_n.right == -j_flux,
                flux_p.left == 0.0, flux_p.right == j_flux
            ],
            
            # --- Global Mathematics (DAEs & ODEs) ---
            # Captures 0D physics, pure Algebraic Constraints, and Initial Conditions.
            "global": [
                # Algebraic Voltage constraint. Notice cycler logic is omitted!
                self.V_cell == (4.2 - 0.0001 * c_surf_p) - (0.1 - 0.0001 * c_surf_n) - 0.02 * self.i_app,
                
                # Initial Conditions (.t0)
                self.c_s_n.t0 == 800.0,
                self.c_s_p.t0 == 200.0,
                self.V_cell.t0 == 4.18,
                self.i_app.t0 == 0.0
            ]
        }
```

### 2. The Engineer View: Co-Simulation & Multi-Mode Protocols

The compiled `Engine` handles execution. For lab testing (e.g., CCCV cycling), engineers supply a compiled state-machine (`fx.protocols.Sequence`). The native solver dynamically hot-swaps the active constraints utilizing safe root-finding to land perfectly on voltage trigger asymptotes.

```python
from ion_flux.protocols import Sequence, CC, CV, Rest

model = SingleParticleModel()
engine = fx.Engine(model=model, target="cpu:serial")

# Declarative, compiled state machine.
# Notice how we use Python operator overloading (`model.V_cell <= 3.0`) 
# to build strictly-typed AST trigger conditions safely, rather than using strings.
protocol = Sequence([
    CC(rate=10.0, until=model.V_cell <= 3.0, time=3600),
    CV(voltage=3.0, until=model.i_app >= -0.05, time=1800),
    Rest(time=600)
])

result = engine.solve(protocol=protocol)
result.plot_dashboard() # Launches an interactive time-scrubbing UI
```

For **Real-Time HIL/SIL (Battery Management Systems)**, V2 uses a **Stateful Session Generator**. Only simple scalar inputs cross the Python/Rust FFI boundary, enabling microsecond-latency control loops.

```python
session = engine.start_session(parameters={"Ds_n": 2e-14})

# Real-time control loop (e.g., 10ms BMS cycle over 60 seconds)
while session.time < 60.0:
    # 1. Read state directly from the underlying C-array with zero overhead
    current_v = session.get("V_cell")
    
    # 2. Custom external control logic (Software-in-the-Loop)
    i_req = -5.0 if current_v < 4.15 else 0.0 
    
    # 3. Step the native implicit solver forward. 
    # High-order BDF integration history vectors are seamlessly preserved in Rust.
    # The `_term_mode` parameter safely toggles the `Terminal` abstraction
    # (1.0 = Current Control, 0.0 = Voltage Control).
    session.step(dt=0.01, inputs={"_term_i_target": i_req, "_term_mode": 1.0})
```

---

## Frequently Asked Questions

**Q1: How does the "Topology-Agnostic Math" actually work under the hood?**

**A:** When you write `fx.grad(T)`, you are simply placing a node in the Python AST. During compilation, the Python codegen layer inspects the `Domain` assigned to `T`. If it is a 1D spherical domain, it lowers `fx.grad` into a finite-volume tridiagonal stencil (applying L'Hopital's limits at the origin natively to prevent 0/0 singularities). If the domain was loaded via `Domain.from_mesh("3d_cell.msh")`, it lowers the exact same AST node into a 3D unstructured Matrix-Free CSR graph traversal before passing the C++ code to LLVM.

**Q2: How does the compiler handle Moving Boundaries (ALE) without severe performance penalties?**

**A:** In a Stefan problem (like particle swelling or lithium plating), you bind a state to a domain boundary (`self.x.right == self.L`). The AST compiler detects this and dynamically flags the domain as deformable. It automatically injects an Arbitrary Lagrangian-Eulerian (ALE) grid velocity advection term into any transport equations evaluating over that mesh. It also implements directional upwinding to guarantee advective stability. Because the native solver and the AST are tightly coupled, the Jacobian sparsity pattern remains constant; only the spatial derivatives shift smoothly as the mesh stretches.

**Q3: How does Enzyme Automatic Differentiation (AD) interact with stiff PDEs?**

**A:** Traditional autodiff frameworks (like JAX or PyTorch) build massive computational graphs in memory, which scales poorly for stiff PDEs requiring thousands of implicit time steps. Enzyme operates at the LLVM IR level, performing AD *after* the code is optimized. `ion_flux` uses Enzyme in two passes:
1.  **Forward-mode:** To automatically generate exact, analytical Jacobians for the implicit solver's Newton iterations (guaranteeing quadratic convergence).
2.  **Reverse-mode (Adjoint):** To trace sensitivities backward through the time-stepping trajectory, allowing you to compute the gradient of a loss function with respect to any parameter without storing the entire forward memory state.
