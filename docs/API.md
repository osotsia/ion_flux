### `ion_flux` V2 API Reference Architecture

The `ion_flux` V2 API enforces a strict architectural boundary between **Mathematical Declaration** (Python AST generation) and **Execution** (Rust JIT compilation and hardware orchestration). This separation minimizes cognitive load for researchers designing physics, while providing engineers with predictable compilation boundaries, flat data structures, and explicit hardware control.

---

### 1. The Researcher API: Declarative PDEs
Researchers use a Declarative Class Pattern to define physics. The class builds a computational graph (AST) by tracing the operations inside the `math()` method. 

*   **States (`fx.State`)**: Unknowns to be solved. If assigned a domain, they are discretized (PDEs). If domain is `None`, they are scalars (ODEs or Algebraic constraints).
*   **Dictionaries**: The `math()` method returns a dictionary. Keys dictate the mathematical intent:
    *   `fx.dt(State)` specifies an ordinary or partial differential equation.
    *   `State` (without `dt`) specifies an algebraic constraint, turning the system into a DAE ($0 = \text{residual}$).
    *   `State.left` / `fx.grad(State).right` specifies Dirichlet or Neumann boundary conditions.
    *   `State.t0` specifies initial conditions.

#### Example: Coupled Particle Diffusion (PDE + Algebraic DAE)
```python
import ion_flux as fx

class SingleParticle(fx.PDE):
    # 1. Topology
    r = fx.Domain(bounds=(0.0, 1.0), resolution=50, coord_sys="spherical")
    
    # 2. States
    c = fx.State(domain=r)      # Lithium concentration (PDE)
    V = fx.State(domain=None)   # Particle voltage (Algebraic scalar)
    
    # 3. Parameters
    D = fx.Parameter(default=1e-14)
    i_app = fx.Parameter(default=1.0)
    
    # 4. Physics (Traced to build the AST)
    def math(self):
        flux = -self.D * fx.grad(self.c)
        
        return {
            # --- PDE: Spherical Diffusion ---
            fx.dt(self.c): -fx.div(flux),
            self.c.t0:     1000.0,
            
            # Boundary Conditions
            flux.left:     0.0,                  # Symmetry at center
            flux.right:    self.i_app / 96485.0, # Flux at surface
            
            # --- Algebraic Constraint (DAE) ---
            # Dictates that 0 = V - f(c_surface)
            self.V:        self.V - (4.2 - 0.1 * self.c.right),
            self.V.t0:     4.1
        }

# Local prototyping compiles and runs implicitly on the default target (CPU serial)
model = SingleParticle()
result = model.solve(t_span=(0, 3600))
```

---

### 2. The Engineer API: SaaS Backend & Compilation
In a production environment, models are pre-built and compilation is highly expensive. The API introduces the `Engine` abstraction to completely isolate the JIT compilation phase from the execution phase.

Engineers interact with flat, dot-notated parameter dictionaries suitable for JSON payloads.

#### Example: FastAPI Async Endpoint
```python
from ion_flux.battery import DFN
from ion_flux import Engine
import ion_flux.protocols as protocols

# 1. JIT Compilation (Executed exactly once at worker boot)
# Parses the DFN AST, generates analytical Jacobian, emits C++/PTX, compiles.
engine = Engine(model=DFN(thermal="lumped"), target="cuda:0")

async def process_simulation_request(payload: dict):
    # payload = {
    #   "c_rate": 2.5, 
    #   "params": {"neg_elec.porosity": 0.3, "pos_elec.thickness": 1e-4}
    # }
    
    protocol = protocols.ConstantCurrent(
        rate=payload["c_rate"], 
        until_voltage=2.5
    )
    
    # 2. Execution (Non-blocking)
    # Rust simply updates the device memory pointers and triggers SUNDIALS via FFI.
    result = await engine.solve_async(
        protocol=protocol,
        parameters=payload["params"]
    )
    
    return result.to_dict(variables=["Voltage [V]", "Current [A]", "Time [s]"])
```

---

### 3. Hardware Targets & Parallelism Models
The computational bottleneck shifts depending on the problem size and the number of permutations. The `Engine` target dictates how the Rust middle-end emits code and configures the SUNDIALS linear solvers.

#### A. Task Parallelism (CPU Sweeps / MCMC)
Optimized for running thousands of small-to-medium models concurrently. The generated C++ is strictly serial. Concurrency is handled by Rust's `rayon` thread pool, spinning up multiple independent SUNDIALS instances in host memory.
```python
engine = Engine(model=DFN(), target="cpu")

# solve_batch utilizes all available vCPUs to solve parameter sets concurrently.
results = engine.solve_batch(
    protocol=drive_cycle,
    parameters=[param_set_1, param_set_2, ..., param_set_1000]
)
```

#### B. Data Parallelism (Massive 3D Models)
Optimized for single, massive models (e.g., $10^6$ degrees of freedom). The Rust JIT compiler emits `#pragma omp parallel for` across the finite-volume spatial loops. SUNDIALS is configured with `NVECTOR_OPENMP` and an iterative linear solver (e.g., GMRES).
```python
engine = Engine(model=DFN(dimensions=3), target="cpu:omp")

# A single solve() call utilizes all specified cores to compute 
# the residual and Jacobian matrices faster.
result = engine.solve(
    protocol=standard_charge,
    parameters=base_params,
    threads=16
)
```

#### C. GPU Throughput (CUDA/PTX)
Optimized for multi-tenant SaaS routing or massive batched tensors. The Rust JIT compiler emits PTX/CUDA. SUNDIALS uses `NVECTOR_CUDA` and `cuSOLVER`. Memory remains entirely in VRAM during the integration loop.
```python
engine = Engine(model=DFN(), target="cuda:0")

# Solves happen on the device. Data is only transferred back to 
# host memory upon completion.
result = engine.solve(protocol=drive_cycle)
```

---

### 4. Advanced Protocols (Drive Cycles)
For arbitrary experimental data (e.g., WLTP drive cycles), injecting Python interpolation functions into the execution loop destroys performance. The API requires wrapping numpy arrays in `fx.protocols` types. The Rust orchestrator converts these into tightly packed, contiguous C-arrays, and the JIT compiler embeds a fast, branchless 1D interpolation kernel directly into the emitted C++/CUDA residual function.

```python
import pandas as pd
import ion_flux.protocols as protocols

# Load high-frequency current data
raw_data = pd.read_csv("WLTP.csv").to_numpy()
time_array = raw_data[:, 0]
current_array = raw_data[:, 1]

# Wrap in typed protocol
drive_cycle = protocols.Profile(time=time_array, current=current_array)

engine = Engine(model=DFN(), target="cpu")
result = engine.solve(protocol=drive_cycle)
```