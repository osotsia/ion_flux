import ion_flux as fx
from ion_flux.protocols import Sequence, CC, CV, Rest


class DFN(fx.PDE):
    """
    1D-1D Doyle-Fuller-Newman model utilizing explicitly coupled 
    spatial regions to guarantee numerical convergence.
    """
    # 1. Topology
    x_n = fx.Domain(bounds=(0, 40e-6), resolution=20, name="x_n")
    x_s = fx.Domain(bounds=(40e-6, 60e-6), resolution=10, name="x_s")
    x_p = fx.Domain(bounds=(60e-6, 100e-6), resolution=20, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_p") 
    
    macro_n = x_n * r_n 
    macro_p = x_p * r_p 
    
    # 2. States
    c_e_n = fx.State(domain=x_n, name="c_e_n")
    c_e_s = fx.State(domain=x_s, name="c_e_s")
    c_e_p = fx.State(domain=x_p, name="c_e_p")
    
    phi_e_n = fx.State(domain=x_n, name="phi_e_n")
    phi_e_s = fx.State(domain=x_s, name="phi_e_s")
    phi_e_p = fx.State(domain=x_p, name="phi_e_p")
    
    c_s_n = fx.State(domain=macro_n, name="c_s_n")
    c_s_p = fx.State(domain=macro_p, name="c_s_p")
    
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")
    
    V_cell = fx.State(domain=None, name="V_cell") 
    i_app = fx.State(domain=None, name="i_app")
    
    # 3. Hardware Abstraction
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        # ---------------------------------------------------------------------
        # 1. Base Parameters & Physical Constants
        # ---------------------------------------------------------------------
        De, ke = 1e-10, 1.0
        Ds_n, Ds_p = 1e-14, 1e-14
        sig_n, sig_p = 100.0, 100.0
        
        F = 96485.0
        a_n = 3.0 / 5e-6
        a_p = 3.0 / 5e-6
        aF_n = a_n * F
        aF_p = a_p * F
        
        # ---------------------------------------------------------------------
        # 2. Continuous Internal Fluxes
        # ---------------------------------------------------------------------
        N_e_n = -De * fx.grad(self.c_e_n) 
        N_e_s = -De * fx.grad(self.c_e_s) 
        N_e_p = -De * fx.grad(self.c_e_p) 
        
        i_e_n = -ke * fx.grad(self.phi_e_n) 
        i_e_s = -ke * fx.grad(self.phi_e_s) 
        i_e_p = -ke * fx.grad(self.phi_e_p) 
        
        i_s_n = -sig_n * fx.grad(self.phi_s_n) 
        i_s_p = -sig_p * fx.grad(self.phi_s_p) 
        
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n) 
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p) 
        
        # ---------------------------------------------------------------------
        # 3. Electrochemical Kinetics
        # ---------------------------------------------------------------------
        # Evaluate particle surface concentration dynamically
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p) 
        
        # Simplified OCV mappings
        U_n = 0.1 - 0.0001 * c_surf_n 
        U_p = 4.2 - 0.0001 * c_surf_p 
        
        # Overpotential
        eta_n = self.phi_s_n - self.phi_e_n - U_n 
        eta_p = self.phi_s_p - self.phi_e_p - U_p 
        
        # Volumetric exchange current (A/m^3)
        j_n = 1e6 * eta_n 
        j_p = 1e6 * eta_p 

        return {
            "regions": {
                self.x_n: [
                    fx.dt(self.c_e_n) == -fx.div(N_e_n) + (j_n / F),
                    
                    # Applying the eq_scale to pure Algebraic spatial constraints
                    0 == (fx.div(i_e_n) - j_n),
                    0 == (fx.div(i_s_n) + j_n)
                ],
                self.x_s: [
                    fx.dt(self.c_e_s) == -fx.div(N_e_s),
                    0 == (fx.div(i_e_s))
                ],
                self.x_p: [
                    fx.dt(self.c_e_p) == -fx.div(N_e_p) + (j_p / F),
                    
                    0 == (fx.div(i_e_p) - j_p),
                    0 == (fx.div(i_s_p) + j_p)
                ],
                self.macro_n: [
                    fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n)
                ],
                self.macro_p: [
                    fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p)
                ]
            },
            "boundaries": [
                # --- Electrolyte Mass Conservation ---
                N_e_n.left == 0.0, 
                N_e_n.right == N_e_s.left, 
                self.c_e_n.right == self.c_e_s.left, # State continuity
                
                N_e_s.right == N_e_p.left,
                self.c_e_s.right == self.c_e_p.left, # State continuity
                N_e_p.right == 0.0, 
                
                # --- Electrolyte Potential Conservation ---
                self.phi_e_n.left == 0.0,            # Anchor node
                
                i_e_n.right == i_e_s.left, 
                self.phi_e_n.right == self.phi_e_s.left,
                
                i_e_s.right == i_e_p.left, 
                self.phi_e_s.right == self.phi_e_p.left,
                i_e_p.right == 0.0,
                
                # --- Solid Potential (Current Collectors) ---
                i_s_n.left == -self.i_app, 
                i_s_n.right == 0.0,
                
                i_s_p.left == 0.0, 
                i_s_p.right == -self.i_app,
                
                # --- Particle Solid Diffusion (Faraday's Law) ---
                N_s_n.boundary("left", domain=self.r_n) == 0.0,  
                
                # THE FIX: Outward flux must be mathematically positive. 
                # (Removing the negative sign physically fixes the battery)
                N_s_n.boundary("right", domain=self.r_n) == j_n / aF_n, 
                
                N_s_p.boundary("left", domain=self.r_p) == 0.0,  
                N_s_p.boundary("right", domain=self.r_p) == j_p / aF_p 
            ],
            "global": [
                self.V_cell == self.phi_s_p.right - self.phi_s_n.left,
                
                # Initial Conditions
                self.c_e_n.t0 == 1000.0, self.c_e_s.t0 == 1000.0, self.c_e_p.t0 == 1000.0, 
                self.phi_e_n.t0 == 0.0, self.phi_e_s.t0 == 0.0, self.phi_e_p.t0 == 0.0,
                self.phi_s_n.t0 == 0.05, self.phi_s_p.t0 == 4.15, 
                self.c_s_n.t0 == 500.0, self.c_s_p.t0 == 500.0, 
                self.V_cell.t0 == 4.10, self.i_app.t0 == 0.0 
            ]
        }
    
# 1. Initialize your model and engine
# Assuming a standard 1D-1D Doyle-Fuller-Newman (DFN) model from the library
model = DFN()

# Explicitly set jacobian_bandwidth=0 for highly coupled multi-scale DFNs
engine = fx.Engine(model=model, target="cpu:serial", jacobian_bandwidth=0, solver_backend="native")

# 2. Define cycler parameters (1.0 Ah cell)
C_RATE = 1.0       # 1C = 1.0 Amps
V_MAX = 4.2        # Upper voltage limit
V_MIN = 2.5        # Lower voltage limit (from the XML limits)
I_CUTOFF = 0.05    # C/20 cutoff for CV phases

# 3. Build the Initial Conditioning & Capacity Check Phase
steps = [
    Rest(time=3600),
    
    # Discharge to baseline
    CC(rate=-C_RATE/3, until=model.V_cell <= V_MIN),
    Rest(time=3600),
    
    # Full CCCV Charge
    CC(rate=C_RATE/3, until=model.V_cell >= V_MAX),
    CV(voltage=V_MAX, until=model.i_app <= I_CUTOFF),
    Rest(time=3600),
    
    # Full CCCV Discharge
    CC(rate=-C_RATE/3, until=model.V_cell <= V_MIN),
    CV(voltage=V_MIN, until=model.i_app >= -I_CUTOFF),
    Rest(time=3600),
    
    # Re-Charge to 100% SOC for the pulse test
    CC(rate=C_RATE/3, until=model.V_cell >= V_MAX),
    CV(voltage=V_MAX, until=model.i_app <= I_CUTOFF),
    Rest(time=3600),
]

# 4. Build the HPPC Pulse Train (Iterating down by ~10% SOC)
# Time to discharge 10% at C/3 rate: (0.1 Ah / 0.333 A) * 3600 s = 1080 seconds
for _ in range(9):
    steps.extend([
        # Step down to next SOC level
        CC(rate=-C_RATE/3, time=1080, until=model.V_cell <= V_MIN),
        Rest(time=3600), # 1 hour equilibration
        
        # 1C Discharge Pulse (10 seconds)
        CC(rate=-C_RATE, time=10, until=model.V_cell <= V_MIN),
        Rest(time=300),
        
        # 1C Charge Pulse (10 seconds)
        CC(rate=C_RATE, time=10, until=model.V_cell >= V_MAX),
        Rest(time=3600)
    ])

# 5. Compile the protocol and solve in one shot
protocol = Sequence(steps)

print("Executing full multi-day RPT protocol...")
result = engine.solve(protocol=protocol)

# Launch the interactive dashboard to view the exact trace
result.plot_dashboard(variables=["V_cell", "i_app"])