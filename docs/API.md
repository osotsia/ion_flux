### `ion_flux` V2 API Reference Architecture

The `ion_flux` V2 API embraces a design philosophy of **Zero boilerplate, infinite scalability.** 

By leveraging a purely Python-based AST compiler, a custom native Rust implicit solver, and Enzyme for Compile-Time Automatic Differentiation (AD), the API ruthlessly separates *Physical Intent* from *Computational Execution*. 

---

### Level 1: Declarative Physics & The AST

Physics are declared as Python classes inheriting from `fx.PDE`. The compiler traces the operations inside the `math()` method to build a semantic computational graph. 

To reduce cognitive load, mathematical operators (`fx.grad`, `fx.div`) are **topology-agnostic**. The exact same equation syntax will compile to a tridiagonal matrix on a 1D line, or a massive sparse stiffness matrix on a 3D pouch cell mesh. 

#### Submodels & Composition
`ion_flux` encourages hierarchical object composition. Submodels can be instantiated as attributes, and the framework will automatically deepcopy them, namespace their states and parameters (e.g., prefixing `c_s` to `anode_c_s`), and prevent topological collisions.

```python
import ion_flux as fx

class FickianParticle(fx.PDE):
    """A highly reusable submodel representing simple solid diffusion."""
    r = fx.Domain(bounds=(0, 5e-6), resolution=15, coord_sys="spherical")
    c_s = fx.State(domain=r)
    D_s = fx.Parameter(default=1e-14)
    
    # The Interface Contract: This submodel NEEDS an external flux mapped 
    # to its boundary to function in a broader cell.
    def math(self, j_flux: fx.Node):
        flux = -self.D_s * fx.grad(self.c_s, axis=self.r)
        
        return {
            # Explicitly bind the PDE to the State node it dictates.
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(flux, axis=self.r)
            },
            # Target the specific evaluated tensor for Neumann bounds.
            "boundaries": {
                flux: {"left": 0.0, "right": j_flux}
            },
            "initial_conditions": {}
        }

class ModularSPM(fx.PDE):
    """Composes the full cell using instantiated submodels."""
    
    # 1. Instantiate submodels as attributes
    neg_particle = FickianParticle()
    pos_particle = FickianParticle()
    
    # 2. Define Macro states
    V_cell = fx.State(domain=None)
    i_app = fx.State(domain=None)
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        # 3. Define the coupling physics (Faraday's Law)
        j_n = self.i_app / 96485.0
        j_p = -self.i_app / 96485.0
        
        # 4. Extract surface concentrations from the submodels for the OCV calculation
        c_surf_n = self.neg_particle.c_s.right
        c_surf_p = self.pos_particle.c_s.right
        
        macro_physics = {
            "equations": {
                # Global algebraic variables natively map to V2 standard equation structures
                self.V_cell: self.V_cell == (4.2 - 0.001 * c_surf_p) - (0.1 - 0.001 * c_surf_n)
            },
            "boundaries": {},
            "initial_conditions": {
                self.V_cell: 4.1, 
                self.i_app: 0.0,
                
                # Initial conditions for submodels can be targeted explicitly 
                # from the parent level.
                self.neg_particle.c_s: 800.0,
                self.pos_particle.c_s: 200.0
            }
        }
        
        # 5. Merge all ASTs into a single seamless dictionary payload.
        # The submodel `math()` methods are called, passing the macro fluxes in.
        return fx.merge(
            macro_physics,
            self.neg_particle.math(j_flux=j_n),
            self.pos_particle.math(j_flux=j_p)
        )
```

---

### Level 2: Scaling Complexity (Multi-Scale & DAEs)

In legacy frameworks, modeling macro-micro scale coupling or differential-algebraic equations (DAEs) requires severe "math gymnastics." V2 treats these complex phenomena as native syntax.

*   **Pseudo-Dimensions:** Multiply domains together to create hierarchical cross-product meshes (`macro_micro = x * r`). The compiler unrolls these into highly efficient flat C-array strides.
*   **Spatial DAEs:** Any equation omitting a `fx.dt()` operator is automatically flagged by the compiler as a pure Algebraic constraint. The Rust implicit solver will solve it at every spatial node concurrently with the ODEs.

#### The Complex Case: 1D-1D Doyle-Fuller-Newman (DFN)
```python
class MinimalDFN(fx.PDE):
    """
    Demonstrates explicit interface stitching, hierarchical domains, and spatial DAEs.
    """
    # 1. Topology
    x_n = fx.Domain(bounds=(0, 40e-6), resolution=10, name="x_n")
    x_p = fx.Domain(bounds=(60e-6, 100e-6), resolution=10, name="x_p")
    r_n = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_p") 
    
    # Multiplying domains creates a hierarchical pseudo-dimension.
    # This logically places a full 1D spherical particle mesh at *every* node 
    # in the macroscopic 'x' mesh.
    macro_n = x_n * r_n 
    macro_p = x_p * r_p 
    
    # 2. States
    c_s_n = fx.State(domain=macro_n, name="c_s_n") # 2D PDE State
    phi_s_n = fx.State(domain=x_n, name="phi_s_n") # 1D Spatial DAE State
    
    # ... additional state declarations omitted for brevity ...
    
    def math(self):
        # The `axis` argument forces topology-agnostic operators to differentiate 
        # against a specific spatial dimension within a composite domain.
        i_s_n = -100.0 * fx.grad(self.phi_s_n, axis=self.x_n) 
        N_s_n = -1e-14 * fx.grad(self.c_s_n, axis=self.r_n) 
        
        # Volumetric current evaluation
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        U_n = 0.1 - 0.0001 * c_surf_n 
        j_n = 1e6 * (self.phi_s_n - U_n) 
        
        return {
            "equations": {
                # Notice `phi_s_n` has no time derivative (fx.dt). 
                # It is automatically processed as a spatial Algebraic constraint (DAE)
                # and flagged inherently by the Native Compiler as a 0.0 root dependency.
                self.phi_s_n: fx.div(i_s_n, axis=self.x_n) == -j_n,
                
                # Hierarchical PDE. Solves diffusion inside the particle 
                # at *every* macroscopic node in x_n simultaneously.
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n)
            },
            "boundaries": {
                # Apply cycler current demand to the solid phase boundary via Neumann tensor.
                i_s_n: {"left": -self.i_app, "right": 0.0},
                
                # Map volumetric macro current (j_n) to area flux at the micro boundary.
                N_s_n: {"left": 0.0, "right": -j_n / 5.78e10}
            },
            "initial_conditions": {
                # Provided for completeness
                self.phi_s_n: 0.05,
                self.c_s_n: 500.0
            }
        }
```

---

### Level 3: The Engineer API (Execution & Protocols)

Once the AST is built, the `fx.Engine` uses LLVM to JIT-compile the math via Enzyme into native machine code (C++). `ion_flux` V2 provides ultimate flexibility for execution, seamlessly handling continuous protocol blocks or microsecond HIL stepping.

#### Use Case A: Multi-Mode Protocols (CCCV)
Engineers supply a compiled state-machine (`Sequence`). By targeting the `fx.Terminal` abstraction, the native solver dynamically hot-swaps the active constraint (e.g., forcing $i_{app} = i_{target}$ for CC, or $V_{cell} = v_{target}$ for CV). It utilizes dense bisection root-finding to land perfectly on voltage asymptotes without re-compiling the AST or rebuilding the Jacobian sparsity pattern.

```python
from ion_flux.protocols import Sequence, CC, CV, Rest

model = MinimalDFN()
engine = fx.Engine(model=model, target="cpu:serial")

# Protocol Triggers leverage Python Operator Overloading (model.V_cell >= 4.2)
# to construct safe, strictly-typed AST bounds (V1 string-parsing is deprecated).
protocol = Sequence([
    CC(rate=1.0, until=model.V_cell >= 4.2, time=7200),
    CV(voltage=4.2, until=model.i_app <= 0.05, time=3600),
    Rest(time=3600)
])

result = engine.solve(protocol=protocol)
result.plot_dashboard(["V_cell", "i_app"]) # Launch interactive visualization UI
```

#### Use Case B: Micro-Stepping Co-Simulation (BMS HIL)
For Software-in-the-Loop, the compiled sparse matrices, exact Enzyme Jacobians, and BDF history vectors stay "hot" in the Rust hardware memory via a `Session`. Only 64-bit float inputs cross the Python boundary, enabling real-time control loops.

```python
session = engine.start_session(parameters={"anode_D_s": 2e-14})

# E.g., Executing a 100ms BMS cycle over a 1 hour simulated test
while session.time < 3600:
    # 1. Read state directly from the underlying C-array with zero overhead
    current_v = session.get("V_cell")
    
    # 2. Evaluate external BMS logic
    i_req = my_bms.compute_current(v_cell=current_v)
    
    # 3. Safely toggle the multiplexer using the hidden terminal flags
    # (1.0 = Current Control, 0.0 = Voltage Control).
    # The native Rust solver integrates forward 100ms, preserving history.
    session.step(dt=0.1, inputs={"_term_i_target": i_req, "_term_mode": 1.0})
    
    # 4. Check for arbitrary safety aborts
    if session.triggered(fx.Condition(model.V_cell < 2.5)):
        print("BMS Safety Halt Triggered")
        break
```

---

### Level 4: Advanced Analytics (Differentiability & EIS)

Because the Python compiler uses Enzyme AD natively at the LLVM level, the entire execution loop is a **differentiable computational graph**. Optimization and frequency-domain analysis are mathematically exact byproducts of the architecture.

#### Native Frequency-Domain Solvers (EIS)
Simulating Electrochemical Impedance Spectroscopy (EIS) no longer requires slow, noise-prone finite-difference time-domain perturbation loops. 

The engine drives the cell to steady-state, extracts the Enzyme analytical Jacobian ($J$), and solves the transfer function $Z(\omega) = C (j\omega M - J)^{-1} B$ directly and algebraically using `scipy.linalg`.

```python
import numpy as np

session = engine.start_session(parameters={"anode_D_s": 1e-14})
session.reach_steady_state()

# Solves analytically in milliseconds
frequencies = np.logspace(-3, 5, 100)
eis_spectrum = session.solve_eis(frequencies=frequencies, input_var="i_app", output_var="V_cell")

# Access the complex components natively
print(eis_spectrum["Z_real"].data, eis_spectrum["Z_imag"].data)
```

#### Continuous Adjoint Sensitivities
To optimize parameters against lab data, trigger reverse-mode Automatic Differentiation via the `discrete_adjoint_native` backend loop. It calculates exact, continuous sensitivities backward through the time-stepping trajectory. This $\mathcal{O}(1)$ memory approach perfectly integrates over discrete trigger jumps (e.g., CC to CV sequence crossovers) without diverging.

```python
# The `requires_grad` flag instructs the solver to record the trajectory history
res = engine.solve(protocol=fast_charge, requires_grad=["anode_D_s"])

# Compute a differentiable loss metric against lab CSV data
loss = fx.metrics.rmse(predicted=res["V_cell"], target=experimental_voltage)

# Backpropagate through the entire implicit solver via Vector-Jacobian Products!
grads = loss.backward()

# The exact, continuous gradient is available for L-BFGS-B or Adam optimizers
print(grads["anode_D_s"]) 
```

---

### Level 5: Cloud Scale & Native Parallelism

In a production environment (e.g., a Battery Digital Twin SaaS), Python's Global Interpreter Lock (GIL) and multiprocessing pickling serializers become massive scaling bottlenecks. `ion_flux` solves this natively at the compiled systems level.

#### 1. Task Parallelism (Rayon CPU Batching)
Used when solving thousands of independent parameter permutations simultaneously (e.g., MCMC parameter estimation or fleet-wide monitoring).

```python
engine = fx.Engine(model=ModularSPM(), target="cpu:serial")

# Create a massive list of flat parameter dictionaries
param_payloads = [{"anode_D_s": p} for p in np.linspace(1e-14, 5e-14, 1000)]

# `solve_batch` drops into Rust, utilizing the Rayon thread-pool to distribute 
# 1000 independent BDF integration handles across all vCPUs. 
# It completely bypasses the Python GIL.
results = engine.solve_batch(parameters=param_payloads, t_span=(0, 3600), max_workers=64)
```

#### 2. Data Parallelism (Massive 3D Models)
Used when solving a *single*, massively complex unstructured finite-element mesh utilizing OpenMP.

```python
# `target="cpu:omp"` flags the AST translator to identify spatial arrays 
# and emit `#pragma omp parallel for` during the C++ translation phase.
engine = fx.Engine(model=PouchCell3D(), target="cpu:omp")

# The Rust orchestrator sets `OMP_NUM_THREADS=32` natively before execution,
# distributing the massive sparse matrix assembly across 32 cores.
res = engine.solve(t_span=(0, 3600), threads=32)
```

#### 3. JIT Caching & Zero-Start Deployments
To deploy models into serverless environments (e.g., AWS Lambda, FastAPI endpoints), you can export the compiled model into a standalone shared object (`.so`) library. Python AST parsing, translation, and LLVM/Clang compilation are entirely bypassed at runtime, enabling 0ms cold-starts.

```python
# --- 1. Compile on your CI/CD Pipeline ---
engine = fx.Engine(model=MinimalDFN(), target="cpu:serial")
engine.export_binary("models/dfn_prod.so")

# --- 2. Load instantly on a Serverless Worker ---
# Instantiates instantly without invoking Clang or LLVM.
stateless_engine = fx.Engine.load("models/dfn_prod.so", target="cpu:serial")

async def simulate_endpoint(payload: dict):
    # Stateless, thread-safe execution perfect for concurrent web requests
    return await stateless_engine.solve_async(parameters=payload["params"])
```