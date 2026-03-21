### `ion_flux` V2 API Reference Architecture

The `ion_flux` V2 API embraces a design philosophy of **Zero boilerplate, infinite scalability.** 

By leveraging a purely Python-based AST compiler, a custom native implicit solver, and Enzyme for Automatic Differentiation (AD), the API ruthlessly separates *Physical Intent* from *Computational Execution*. Researchers declare what the physics *are*; the engine automatically deduces how to difference, linearize, compile, and solve them across 1D grids, 3D meshes, or scalable cloud infrastructure.

---

### Level 1: The Basics (Declarative Physics)
Physics are declared as Python classes inheriting from `fx.PDE`. The compiler traces the operations inside the `math()` method to build a computational graph. 

To reduce cognitive load, mathematical operators (`fx.grad`, `fx.div`) are **topology-agnostic**. The exact same equation syntax will compile to a tridiagonal matrix on a 1D line, or a massive unstructured stiffness matrix on a 3D pouch cell mesh.

#### The Simple Case: A 1D Spherical Particle
```python
import ion_flux as fx

class SingleParticle(fx.PDE):
    # 1. Topology
    # Defines the spatial discretisation. Can be 1D, 2D, or a 3D unstructured mesh.
    r = fx.Domain(bounds=(0.0, 1.0), resolution=50, coord_sys="spherical")
    
    # 2. States
    # Unknowns to be solved. If assigned a domain, they become PDEs. 
    # If domain=None, they are scalar ODEs or Algebraic constraints.
    c = fx.State(domain=r)      # Lithium concentration (PDE)
    V = fx.State(domain=None)   # Particle voltage (Algebraic scalar)
    
    # 3. Parameters
    # Constants or inputs that can be modified at runtime without recompiling.
    D = fx.Parameter(default=1e-14)
    i_app = fx.Parameter(default=1.0)
    
    # 4. Physics (Traced by the Python compiler to build the AST)
    def math(self):
        # The compiler automatically handles the spherical coordinate Jacobian
        flux = -self.D * fx.grad(self.c)
        
        return {
            # --- PDE: Spherical Diffusion ---
            fx.dt(self.c): -fx.div(flux),
            self.c.t0:     1000.0,               # Initial Condition
            
            # --- Boundary Conditions ---
            flux.left:     0.0,                  # Symmetry at particle center
            flux.right:    self.i_app / 96485.0, # Faraday's flux at surface
            
            # --- Algebraic Constraint (DAE) ---
            # Dictates that 0 = V - f(c_surface). 
            # The solver natively handles the resulting DAE system.
            self.V:        self.V - (4.2 - 0.1 * self.c.right),
            self.V.t0:     4.1
        }
```

---

### Level 2: Scaling Complexity (Multi-Scale & Moving Boundaries)
In legacy frameworks, modeling Particle Size Distributions, macro-micro scale coupling, or physical swelling requires severe "math gymnastics." V2 treats these complex phenomena as native syntax.

*   **Pseudo-Dimensions:** Multiply domains together to create hierarchical cross-product meshes. Integrate over them natively.
*   **Moving Boundaries (Stefan Problems):** Assign an equation to a domain's boundary property. The compiler automatically injects the Arbitrary Lagrangian-Eulerian (ALE) convection terms into your PDEs to account for the stretching mesh.

#### The Complex Case: Macro-Micro Electrode with Swelling
```python
class SwellingElectrode(fx.PDE):
    """
    A comprehensive example of a macro-micro scale coupled electrode 
    featuring dynamic moving boundaries (Stefan problem) due to particle swelling.
    """
    
    # 1. Topology: Cross-Product Domains
    # 'x' is the macroscopic through-thickness of the electrode.
    # 'r' is the microscopic interior of the spherical active material particles.
    x = fx.Domain(bounds=(0, 100e-6), resolution=20, name="macro_x")
    r = fx.Domain(bounds=(0, 5e-6), resolution=15, coord_sys="spherical", name="micro_r")
    
    # Multiplying domains creates a hierarchical pseudo-dimension.
    # This places a full 1D spherical particle mesh at every node in the 1D 'x' mesh.
    macro_micro = x * r 
    
    # 2. States
    c_s = fx.State(domain=macro_micro)  # Solid lithium concentration (defined on x * r)
    c_e = fx.State(domain=x)            # Electrolyte concentration (defined on x)
    L   = fx.State(domain=None)         # Scalar tracking the macroscopic electrode thickness
    
    # 3. Parameters
    D_s   = fx.Parameter(default=1e-14) # Solid diffusivity
    D_e   = fx.Parameter(default=1e-10) # Electrolyte diffusivity
    i_app = fx.Parameter(default=30.0)  # Applied current density [A/m^2]
    omega = fx.Parameter(default=5e-7)  # Volumetric expansion coefficient
    
    # 4. Physics (Traced by the Python AST Compiler)
    def math(self):
        # The `axis` argument tells the topology-agnostic operators which coordinate
        # system to differentiate against. 
        # N_s computes spherical gradients inside the particles.
        N_s = -self.D_s * fx.grad(self.c_s, axis=self.r)
        
        # N_e computes Cartesian gradients across the macro electrode.
        N_e = -self.D_e * fx.grad(self.c_e, axis=self.x)
        
        # Assume a uniform reaction distribution for simplicity
        # (Faraday's law flux: [mol / m^2 s])
        j_flux = self.i_app / (96485.0 * self.L)
        
        # Calculate specific interfacial area (a = 3 / R_particle)
        a = 3.0 / 5e-6

        return {
            # -----------------------------------------------------------------
            # MOVING BOUNDARY (STEFAN PROBLEM)
            # -----------------------------------------------------------------
            # 1. Define the ODE for the physical expansion of the electrode.
            # Here, thickness grows proportionally to the total flux integrated across 'x'.
            fx.dt(self.L): self.omega * fx.integral(j_flux, over=self.x),
            self.L.t0:     100e-6,
            
            # 2. DYNAMIC MESH BINDING
            # By assigning the scalar state 'L' to the right edge of the 'x' domain, 
            # the compiler automatically turns this into a moving mesh. It natively 
            # injects Arbitrary Lagrangian-Eulerian (ALE) advection terms into the 
            # macro-scale PDEs (like c_e) to account for the stretching coordinate system.
            self.x.right:  self.L,

            # -----------------------------------------------------------------
            # MICRO-SCALE PDEs: Solid Particle Diffusion (x * r)
            # -----------------------------------------------------------------
            fx.dt(self.c_s):          -fx.div(N_s, axis=self.r),
            self.c_s.t0:              1000.0,
            
            # Boundaries evaluated at the 'r' edges for EVERY particle along 'x'
            N_s.left(domain=self.r):  0.0,     # Symmetry at particle center
            N_s.right(domain=self.r): j_flux,  # Intercalation flux at particle surface

            # -----------------------------------------------------------------
            # MACRO-SCALE PDEs: Electrolyte Transport (x)
            # -----------------------------------------------------------------
            # The electrolyte is depleted by the flux entering the solid phase
            fx.dt(self.c_e):          -fx.div(N_e, axis=self.x) - (a * j_flux),
            self.c_e.t0:              1000.0,
            
            # Boundaries evaluated at the 'x' edges
            N_e.left(domain=self.x):  0.0,     # Flux from separator (assumed 0 for isolation)
            N_e.right(domain=self.x): 0.0,     # Impermeable current collector
        }
```

---

### Level 3: The Engineer API (Execution & Protocols)
Once the AST is built, the `fx.Engine` takes over. It compiles the math via Enzyme AD into native machine code (C++/PTX). V2 provides ultimate flexibility for execution, whether simulating standard lab tests or running real-time Battery Management System (BMS) co-simulations.

#### Use Case A: Step-by-Step Co-Simulation (HIL/SIL)
For Software-in-the-Loop, V2 uses a **Stateful Session Generator**. The compiled sparse matrices, Jacobians, and history vectors stay "hot" in the target hardware's memory. Only scalar inputs cross the Python boundary, enabling microsecond-latency control loops.

```python
engine = fx.Engine(model=FullCell(), target="cuda:0")

# Start a hot session on the GPU
session = engine.start_session(parameters={"D": 2e-14})

# Real-time control loop (e.g., 100ms BMS cycle)
while session.time < 3600:
    # 1. Read from external BMS logic / sensors
    i_req = my_bms.compute_current(v_cell=session.get("Voltage"))
    
    # 2. Step the native implicit solver forward. 
    # High-order integration history is seamlessly preserved.
    session.step(dt=0.1, inputs={"i_app": i_req})
    
    # 3. Arbitrary event catching (e.g., safety limits)
    if session.triggered("Lithium Plating Threshold"):
        print("BMS Triggered Safety Halt")
        break
```

#### Use Case B: Multi-Mode Protocols (CCCV)
For standard lab testing, engineers supply a compiled state-machine (`Sequence`). The native solver dynamically hot-swaps the differential and algebraic constraints (e.g., from Current Control to Voltage Control) upon hitting trigger conditions—without halting the solver or recompiling.

```python
from ion_flux.protocols import Sequence, CC, CV, Rest

# A declarative, compiled state machine
protocol = Sequence([
    CC(rate=1.0, until=fx.Condition("Voltage >= 4.2")),
    CV(voltage=4.2, until=fx.Condition("i_app <= 0.05")),
    Rest(time=3600)
])

result = engine.solve(protocol=protocol)
```

---

### Level 4: Advanced Analytics (Differentiability & EIS)
Because the Python compiler uses Enzyme AD natively, the entire execution loop is a **differentiable computational graph**. Optimization and frequency-domain analysis are virtually free byproducts of the architecture.

#### End-to-End Differentiable Simulation
Optimize charging protocols or parameter estimation using exact, continuous adjoint gradients extracted directly from the native solver.

```python
# Pass `requires_grad` to track sensitivities through the entire time-stepping loop
result = engine.solve(protocol=fast_charge, requires_grad=["D", "thickness"])

# Compute a scalar loss function against experimental lab data
loss = fx.metrics.rmse(result["Voltage"], lab_data)

# Backpropagate through the entire implicit solver! (PyTorch style)
loss.backward()

# Access exact gradients for gradient descent / Adam optimizers
print(engine.parameters["D"].grad) 
```

#### Native Frequency-Domain Solvers (EIS)
Simulating Electrochemical Impedance Spectroscopy (EIS) no longer requires complex AC perturbation math. The engine extracts the analytical Jacobian at steady-state and solves $(j\omega I - J)^{-1} B$ algebraically.

```python
import numpy as np

# 1. Drive the cell to steady state at 50% SOC
session = engine.start_session(soc=0.5)
session.reach_steady_state()

# 2. Directly compute the EIS transfer function algebraically (takes milliseconds)
frequencies = np.logspace(-3, 5, 100)
eis_spectrum = session.solve_eis(frequencies=frequencies, input="i_app", output="Voltage")
```

---

### Level 5: Cloud Scale & Native Parallelism
In a production environment (e.g., a SaaS Battery Digital Twin), the API eliminates Python's multiprocessing boilerplate. The compiled Engine handles memory routing and thread locks natively via explicit API calls, categorized into **Task Parallelism** and **Data Parallelism**.

#### 1. Task Parallelism (CPU/GPU Batching)
Used when solving thousands of independent, small-to-medium models simultaneously (e.g., MCMC parameter estimation, fleet-wide EV monitoring, or cell manufacturing variance).

```python
# Create a massive list of parameter dictionaries
fleet_params = [{"porosity": p, "D": d} for p, d in zip(porosities, diffusivities)]

# Target 'cpu:serial' emits scalar C++. 
# `solve_batch` utilizes an internal C++ thread pool to distribute the solves
# across all available vCPUs perfectly, bypassing the Python GIL.
engine = fx.Engine(model=FullCell(), target="cpu:serial")
fleet_results = engine.solve_batch(protocol=wltp_cycle, parameters=fleet_params)

# Target 'cuda:0' maps the batched states into flat VRAM tensors. 
# Thousands of independent solves execute simultaneously on the GPU streaming multiprocessors.
engine_gpu = fx.Engine(model=FullCell(), target="cuda:0")
fleet_results_gpu = engine_gpu.solve_batch(protocol=wltp_cycle, parameters=fleet_params)
```

#### 2. Data Parallelism (Massive 3D Models)
Used when solving a *single*, massively complex model that requires distributed memory and compute (e.g., a 3D unstructured pouch cell with 1,000,000+ degrees of freedom tracking thermal gradients).

```python
# Target 'cpu:omp' identifies spatial domains in the AST and generates 
# OpenMP parallel loops for matrix assembly and residual evaluation.
engine = fx.Engine(model=PouchCell3D(), target="cpu:omp")

# `threads=64` instructs the native solver to distribute the single Jacobian 
# and sparse linear algebra (e.g., GMRES) across 64 CPU cores.
result = engine.solve(protocol=fast_charge, threads=64)
```

#### 3. JIT Caching & Zero-Start Deployments
AST parsing and Enzyme AD generation happen exactly once. The resulting binary is serialized and distributed to cloud worker nodes, ensuring 0ms startup times for incoming API requests.

```python
# --- On your CI/CD Pipeline ---
engine = fx.Engine(model=FullCell())
engine.export_binary("models/dfn_prod_v1.so")

# --- On your Serverless Worker (e.g., AWS Lambda / FastAPI) ---
# Loads instantly.
engine = fx.Engine.load("models/dfn_prod_v1.so", target="cpu:serial")

async def simulate_endpoint(payload: dict):
    # Stateless, thread-safe execution perfect for concurrent web requests
    return await engine.solve_async(parameters=payload["params"])
```