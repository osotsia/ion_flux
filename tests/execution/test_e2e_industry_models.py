import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE
from ion_flux.protocols import Sequence, CC, CV, Rest

# ==============================================================================
# 1. End-to-End Industry Models written in V2 DSL
# ==============================================================================

class SPM_LumpedThermal(fx.PDE):
    """
    Single Particle Model coupled with a lumped thermal ODE.
    Replaces V1's `pybamm.lithium_ion.SPM({"thermal": "lumped"})`
    """
    r = fx.Domain(bounds=(0, 5e-6), resolution=20, coord_sys="spherical")
    
    c_s = fx.State(domain=r, name="c_s")
    T_cell = fx.State(domain=None, name="T_cell") # 0D Lumped State
    V_cell = fx.State(domain=None, name="V_cell") # Algebraic DAE State
    i_app = fx.State(domain=None, name="i_app")   # Cycler controlled state
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    h_conv = fx.Parameter(default=10.0)
    T_amb = fx.Parameter(default=298.15)
    
    def math(self):
        D_s = 3.9e-14 * fx.exp(-2000.0 / self.T_cell)
        flux = -D_s * fx.grad(self.c_s, axis=self.r)
        
        # Faraday's law at particle surface
        j_surf = self.i_app / 96485.0
        
        # Classic lumped thermal generation (Joule heating + Reaction heat)
        Q_gen = self.i_app * (4.2 - self.V_cell) 
        Q_cool = self.h_conv * (self.T_cell - self.T_amb)
        
        return {
            # PDE: Solid diffusion
            fx.dt(self.c_s): -fx.div(flux, axis=self.r),
            flux.left: 0.0,
            flux.right: j_surf,
            self.c_s.t0: 25000.0,
            
            # ODE: Lumped Thermal
            fx.dt(self.T_cell): (Q_gen - Q_cool) / 1000.0, # Divided by heat capacity
            self.T_cell.t0: 298.15,
            
            # DAE: Voltage Physics Constraint (benign toy OCV curve)
            self.V_cell: 2.5 + 5e-5 * self.c_s.right - 1e-3 * self.i_app,
            self.V_cell.t0: 3.72,
            self.i_app.t0: 30.0
        }


class SPM_SEIGrowth(fx.PDE):
    """
    SPM coupled with Solvent-Diffusion Limited SEI Growth.
    Replaces V1's `pybamm.lithium_ion.SPM({"SEI": "solvent-diffusion limited"})`
    """
    r = fx.Domain(bounds=(0, 5e-6), resolution=20, coord_sys="spherical")
    
    c_s = fx.State(domain=r)
    L_sei = fx.State(domain=None) # Average SEI thickness
    V_cell = fx.State(domain=None)
    i_app = fx.State(domain=None)
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        flux = -1e-14 * fx.grad(self.c_s, axis=self.r)
        j_intercalation = self.i_app / 96485.0
        
        # SEI Growth Rate: Solvent-Diffusion Limited (growth proportional to 1 / L_sei)
        # Regularized to prevent divide-by-zero at t=0
        j_sei = 1e-6 / fx.max(self.L_sei, 1e-9)
        
        return {
            fx.dt(self.c_s): -fx.div(flux, axis=self.r),
            flux.left: 0.0,
            flux.right: j_intercalation + j_sei, # SEI consumes lithium
            self.c_s.t0: 25000.0,
            
            fx.dt(self.L_sei): j_sei * 1e-5, # Molar volume conversion
            self.L_sei.t0: 1e-8, # Initial 10nm native layer
            
            self.V_cell: 2.5 + 5e-5 * self.c_s.right,
            self.V_cell.t0: 3.75,
            self.i_app.t0: 0.0
        }


# ==============================================================================
# 2. End-to-End Execution Tests
# ==============================================================================

def test_e2e_thermal_coupling_execution():
    """Proves the implicit solver successfully handles highly coupled PDE-ODE-DAE systems."""
    model = SPM_LumpedThermal()
    engine = Engine(model=model, target="cpu", mock_execution=not RUST_FFI_AVAILABLE, jacobian_bandwidth=0)
    
    # 1C discharge protocol
    protocol = Sequence([
        CC(rate=30.0, until=fx.Condition("V_cell <= 2.8")),
        Rest(time=600)
    ])
    
    res = engine.solve(protocol=protocol)
    assert res.status == "completed"
    
    if RUST_FFI_AVAILABLE:
        T_end = res["T_cell"].data[-1]
        assert T_end > 298.15 # Cell must have heated up during discharge


def test_e2e_sei_degradation_execution():
    """Proves the JIT handles classic degradation physics seamlessly."""
    model = SPM_SEIGrowth()
    engine = Engine(model=model, target="cpu", mock_execution=not RUST_FFI_AVAILABLE, jacobian_bandwidth=0)
    
    # Rest condition using t_span to allow implicit micro-stepping over the stiff initial transient
    res = engine.solve(t_span=(0, 3600), parameters={"_term_mode": 1.0, "_term_i_target": 0.0}) 
    assert res.status == "completed"
    
    if RUST_FFI_AVAILABLE:
        L_end = res["L_sei"].data[-1]
        assert L_end > 1e-8 # SEI must have grown via solvent diffusion even at rest


# ==============================================================================
# 3. Benchmarks: V2 vs V1 Architectural Paradigms
# ==============================================================================

def test_bench_v2_jit_compilation_speed(benchmark):
    """
    Benchmarks the V2 LLVM/Enzyme JIT emission.
    V1 (CasADi) can take >3 seconds to build the computational DAG for a DFN.
    V2 lowers the pure Python AST directly to C++ in milliseconds.
    """
    def compile_ast_to_cpp():
        model = SPM_LumpedThermal()
        # Bypass disk cache to test raw AST->C++ translation
        return Engine(model=model, target="cpu", cache=False, mock_execution=True)
        
    benchmark.pedantic(compile_ast_to_cpp, rounds=10, iterations=1)