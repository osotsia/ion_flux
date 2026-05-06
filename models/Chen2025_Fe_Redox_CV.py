"""
================================================================================
Replication of Figure 3a from Chen et al., 2025 (DiffEC)
"Differentiable Electrochemistry: A paradigm for uncovering hidden physical phenomena"
================================================================================
"""

import ion_flux as fx
import numpy as np
import matplotlib.pyplot as plt

class Fe_Redox_CV(fx.PDE):
    """1D Diffusion with Butler-Volmer Kinetics for the Fe3+/Fe2+ Redox Couple."""
    
    # 1. Topology with Hierarchical Sub-Meshing
    # The diffusion layer is < 5 um at 200 mV/s. We pack 150 nodes into the first 
    # 15 um to prevent numerical confinement of Fe2+, and use 50 nodes for the bulk.
    x = fx.Domain(bounds=(0, 2e-3), name="x")
    x_dense = x.region(bounds=(0, 15e-6), resolution=150, name="x_dense")
    x_sparse = x.region(bounds=(15e-6, 2e-3), resolution=50, name="x_sparse")
    
    # 2. States and Discoverable Parameters
    c_ox  = fx.State(domain=x, name="c_ox")
    c_red = fx.State(domain=x, name="c_red")
    
    k_0   = fx.Parameter(default=6.54e-5, name="k_0")     
    alpha = fx.Parameter(default=0.248, name="alpha")
    beta  = fx.Parameter(default=0.612, name="beta")
    D_avg = fx.Parameter(default=5.33e-10, name="D_avg")  
    
    E_app = fx.Parameter(default=0.8, name="E_app")       
    
    I_CV_uA = fx.Observable(domain=None, name="I_CV_uA")
    
    def math(self):
        F = 96485.3
        R = 8.314
        T = 298.15
        E_ref = 0.4336  
        Area = np.pi * (0.85e-3)**2
        
        # 3. Mass Transport
        # fx.grad automatically maps across the non-uniform regional grid strides
        flux_ox  = -self.D_avg * fx.grad(self.c_ox, axis=self.x)
        flux_red = -self.D_avg * fx.grad(self.c_red, axis=self.x)
        
        # 4. Butler-Volmer Kinetics
        overpotential = self.E_app - E_ref
        f_RT = F / (R * T)
        
        J_BV = self.k_0 * (
            fx.exp(-self.alpha * f_RT * overpotential) * self.c_ox.left - 
            fx.exp( self.beta  * f_RT * overpotential) * self.c_red.left
        )
        
        I_uA = -1.0 * F * Area * J_BV * 1e6 
        
        return {
            "equations": {
                self.c_ox:  fx.dt(self.c_ox)  == -fx.div(flux_ox, axis=self.x),
                self.c_red: fx.dt(self.c_red) == -fx.div(flux_red, axis=self.x)
            },
            "boundaries": {
                flux_ox:  {"left": -J_BV, "right": 0.0},
                flux_red: {"left":  J_BV, "right": 0.0}
            },
            "initial_conditions": {
                self.c_ox:  4.85, 
                self.c_red: 0.0
            },
            "observables": {
                self.I_CV_uA: I_uA
            }
        }

def run_diffec_figure_3a():
    model = Fe_Redox_CV()
    print("Compiling model (AOT C++ via LLVM/Enzyme)...")
    engine = fx.Engine(model, target="cpu:serial")
    
    scan_rates_mV = [200, 100, 50, 20, 10]
    
    # Note: This mapping strictly matches the physical visual traces of Fig 3a, 
    # intentionally disregarding the typographical error in the published legend.
    colors = plt.cm.viridis(np.linspace(0.95, 0.0, len(scan_rates_mV)))
    
    plt.figure(figsize=(8, 6))
    print("Executing CV sweeps via Hardware-in-the-Loop Micro-stepping...")
    
    for scan_rate_mV, color in zip(scan_rates_mV, colors):
        scan_rate = scan_rate_mV / 1000.0  
        session = engine.start_session()
        
        E_start   = 0.8    
        E_reverse = 0.1    
        
        dE_step = 0.001 
        dt = dE_step / scan_rate
        
        voltages = []
        currents = []
        current_E = E_start
        
        # Forward Scan
        while current_E >= E_reverse:
            session.step(dt=dt, inputs={"E_app": current_E})
            voltages.append(current_E)
            currents.append(session.get("I_CV_uA"))
            current_E -= scan_rate * dt
            
        # Reverse Scan
        while current_E <= E_start:
            session.step(dt=dt, inputs={"E_app": current_E})
            voltages.append(current_E)
            currents.append(session.get("I_CV_uA"))
            current_E += scan_rate * dt

        plt.plot(voltages, currents, label=rf'$\nu$ = {scan_rate_mV} mV/s', 
                 color=color, linewidth=2)
        print(f"  Completed {scan_rate_mV} mV/s sweep.")

    plt.axhline(0, color='grey', linestyle='--', alpha=0.6)
    plt.xlim(0.1, 0.8)
    plt.xlabel("Potential, V vs. SCE", fontsize=12, fontweight='bold')
    plt.ylabel(r"Current, $\mu$A", fontsize=12, fontweight='bold')
    plt.title("Replication of DiffEC Fig 3a (ion_flux)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_diffec_figure_3a()