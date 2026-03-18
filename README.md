# ion_flux: A JIT-Compiled Execution Engine for Battery Simulations

![CI Status](https://github.com/organization/ion_flux/actions/workflows/build_and_test.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/ion-flux.svg)](https://badge.fury.io/py/ion-flux)

`ion_flux` is a Python library with a high-performance Rust, C++, and CUDA core for solving stiff differential-algebraic equations (DAEs) in battery modeling. It is designed for multi-tenant SaaS infrastructures and high-throughput research where compilation overhead, hardware allocation, and memory management must be explicitly controlled.

Models are constructed as mathematical declarations. The core engine dynamically parses the Abstract Syntax Tree (AST), computes analytical Jacobians, and Just-In-Time (JIT) compiles native binaries for execution on targeted hardware.

## Quick Start
```bash
# ion_flux requires the SUNDIALS integration library and a C++ compiler.
# For GPU targets, the NVIDIA CUDA Toolkit is also required.
conda create -n ion_flux python=3.11 c-compiler cxx-compiler sundials cmake -c conda-forge -y
conda activate ion_flux

# Install the package
pip install ion-flux

# Clone the repository
git clone https://github.com/organization/ion_flux.git
cd ion_flux

# Run the examples
python examples/1_simple_diffusion.py
```

## Key Features

*   **Declarative DSL:** Define arbitrary physics (PDEs, ODEs, algebraic constraints) using class-based mathematical declarations without procedural boilerplate.
*   **JIT PDE Compiler:** The Rust middle-end captures the Python AST, performs symbolic differentiation, and emits optimized C++, OpenMP pragmas, or CUDA PTX strings dynamically.
*   **Hardware-Agnostic Execution:** Target specific hardware contexts (`cpu`, `cpu:omp`, `cuda:0`) via the `Engine` interface.
*   **Asynchronous Orchestration:** Designed for multi-tenancy. Execute thousands of parameterized variations concurrently on a single compiled model without blocking the Python GIL.

## Core Concepts

All simulation logic is split into Mathematical Declaration (the model) and Orchestration (the engine).

### 1. The Researcher View: Declaring Custom PDEs

Physics are defined by inheriting from `fx.PDE`. The `math()` method traces the AST of the system.

```python
import ion_flux as fx

class TransientHeat(fx.PDE):
    # 1. Topology & States
    rod = fx.Domain(bounds=(0.0, 2.0), resolution=30)
    T = fx.State(domain=rod)
    k = fx.Parameter(default=0.75)

    # 2. Physics Declaration
    def math(self):
        """
        Returns a dictionary mapping states (and their boundaries) to equations.
        The compiler uses the keys to infer the mathematical intent.
        """
        x = self.rod.coords
        heat_flux = -self.k * fx.grad(self.T)
        source = 1 - fx.abs(x - 1)
        
        return {
            # Governing PDE
            fx.dt(self.T): -fx.div(heat_flux) + source,
            
            # Homogeneous Dirichlet boundaries
            self.T.left:   0.0,
            self.T.right:  0.0,
            
            # Initial temperature profile
            self.T.t0:     2*x - x**2 
        }

# Implicit compilation and execution on default CPU target
model = TransientHeat()
result = model.solve(t_span=(0, 1))
print(f"Final T (center): {result['T'].data[-1, 15]:.3f}")
```

### 2. The Engineer View: Multi-Tenant SaaS Execution

For deployment, the expensive JIT compilation phase is isolated in the `Engine`. Execution is handled asynchronously, allowing multiple user requests to reuse the same cached binary with different parameters.

```python
import asyncio
from ion_flux.battery import DFN, parameters
from ion_flux import Engine
import ion_flux.protocols as protocols

# 1. Ahead-of-Time Compilation (Executes once at worker boot)
engine = Engine(model=DFN(thermal="lumped"), target="cuda:0")
base_params = parameters.Chen2020()

async def process_user_request(c_rate: float, porosity_override: float):
    # 2. Define experimental protocol
    protocol = protocols.ConstantCurrent(rate=c_rate, until_voltage=2.5)
    
    # 3. Asynchronous execution (Data pushed to GPU VRAM, SUNDIALS invoked)
    result = await engine.solve_async(
        protocol=protocol,
        parameters={
            **base_params, 
            "neg_elec.porosity": porosity_override
        }
    )
    return result["Voltage [V]"].data

# Simulate concurrent webhook requests
tasks = [
    process_user_request(c_rate=1.0, porosity_override=0.33),
    process_user_request(c_rate=2.0, porosity_override=0.25)
]
asyncio.run(asyncio.gather(*tasks))
```

## Frequently Asked Questions

**Q1: Why not just use PyBaMM or standard JAX?**

**A:** Traditional scientific Python libraries incur significant overhead from string-based dictionaries, Python-side execution loops, and implicit context management. `ion_flux` strictly separates AST scraping from numerical execution. By using a specialized Rust compiler to emit native SUNDIALS/CUDA C++ code, it avoids the memory fragmentation and scaling bottlenecks inherent to general-purpose autodiff frameworks when dealing with highly sparse, stiff DAE systems.

**Q2: What is the "Compiler-and-Runtime" architecture?**

**A:** The system operates like a virtual machine. Python acts solely as a frontend to serialize equations. Rust processes this Intermediate Representation, manages explicitly scoped hardware memory (RAII), and interacts directly with the `nvcc` compiler and the NVIDIA Driver API. The mathematical computation happens strictly inside the emitted C++/CUDA binary.

**Q3: Is NVIDIA hardware required?**

**A:** No. The target architecture is configurable. `target="cpu"` utilizes standard C++ and the KLU sparse direct solver (ideal for high-throughput 1D parameter sweeps). `target="cpu:omp"` emits OpenMP pragmas for multi-core data parallelism (ideal for massive 3D meshes). `target="cuda:0"` requires an NVIDIA GPU and uses `cuSOLVER`.
