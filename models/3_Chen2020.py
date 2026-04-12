"""
Replication of Chen 2020: "Development of Experimental Techniques for 
Parameterization of Multi-scale Lithium-ion Battery Models"

Model: Isothermal Pseudo-Two-Dimensional (P2D) Doyle-Fuller-Newman
Cell: LG M50 21700 (NMC 811 / Graphite-SiOx)
"""

import numpy as np
import matplotlib.pyplot as plt
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class Chen2020_DFN(fx.PDE):
    # =========================================================================
    # 1. Topological Sub-Meshing (Table II & Table VII)
    # =========================================================================
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=72)
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=35, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=6, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=31, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 5.86e-6), resolution=15, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5.22e-6), resolution=15, coord_sys="spherical", name="r_p") 
    
    # =========================================================================
    # 2. States 
    # =========================================================================
    c_e = fx.State(domain=cell, name="c_e")         
    phi_e = fx.State(domain=cell, name="phi_e")     
    
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")  
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")  
    
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n") 
    c_s_p = fx.State(domain=x_p * r_p, name="c_s_p") 
    
    V_cell = fx.State(domain=None, name="V_cell")    
    i_app = fx.State(domain=None, name="i_app")      
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    D_s_n = fx.Parameter(default=3.3e-14, name="D_s_n")
    
    def math(self):
        # =====================================================================
        # 3. Parameters (Table VII & Table IX Tuned Constants)
        # =====================================================================
        F, R_const, T = 96485.0, 8.314, 298.15
        A_elec = 0.1027      
        
        eps_n, eps_s, eps_p = 0.25, 0.47, 0.335
        eps_act_n, eps_act_p = 0.75, 0.665
        b_brug = 1.5                     
        
        a_n = 3.0 * eps_act_n / 5.86e-6
        a_p = 3.0 * eps_act_p / 5.22e-6
        
        c_max_n = 33133.0
        c_max_p = 63104.0
        
        sig_n, sig_p = 215.0, 0.18
        D_s_p = 4.0e-15
        t_plus = 0.2594
        k_n, k_p = 6.48e-7, 3.42e-6

        # =====================================================================
        # 4. Helper AST Functions
        # =====================================================================
        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)
            
        def sinh_ast(x):
            # Cap exponential growth at eta ~ 0.25V (x=5.0) to prevent phantom currents 
            # from annihilating the mass conservation equations during speculative Newton steps.
            x_max = 5.0 
            cosh_max = 0.5 * (np.exp(x_max) + np.exp(-x_max))
            x_safe = fx.min(fx.max(x, -x_max), x_max)
            bulk_sinh = 0.5 * (fx.exp(x_safe) - fx.exp(-x_safe))
            return bulk_sinh + cosh_max * (x - x_safe)

        # =====================================================================
        # 5. Thermodynamics & Kinetics
        # =====================================================================
        c_surf_n = fx.min(fx.max(self.c_s_n.boundary("right", domain=self.r_n), 10.0), c_max_n - 10.0)
        c_surf_p = fx.min(fx.max(self.c_s_p.boundary("right", domain=self.r_p), 10.0), c_max_p - 10.0)
        
        x_n = c_surf_n / c_max_n
        x_p = c_surf_p / c_max_p
        
        # Bound electrolyte concentration to prevent log singularities AND runaway Newton feedback loops
        ce_safe = fx.min(fx.max(self.c_e, 10.0), 5000.0)
        
        U_n = (1.9793 * fx.exp(-39.3631 * x_n) + 0.2482 
               - 0.0909 * tanh_ast(29.8538 * (x_n - 0.1234)) 
               - 0.04478 * tanh_ast(14.9159 * (x_n - 0.2769)) 
               - 0.0205 * tanh_ast(30.4444 * (x_n - 0.6103)))
               
        U_p = (-0.8090 * x_p + 4.4875 
               - 0.0428 * tanh_ast(18.5138 * (x_p - 0.5542)) 
               - 17.7326 * tanh_ast(15.7890 * (x_p - 0.3117)) 
               + 17.5842 * tanh_ast(15.9308 * (x_p - 0.3120)))

        j0_n = k_n * (ce_safe * c_surf_n * (c_max_n - c_surf_n))**0.5
        j0_p = k_p * (ce_safe * c_surf_p * (c_max_p - c_surf_p))**0.5
        
        eta_n = self.phi_s_n - self.phi_e - U_n
        eta_p = self.phi_s_p - self.phi_e - U_p
        
        F_RT = F / (2.0 * R_const * T)
        J_n = a_n * j0_n * sinh_ast(F_RT * eta_n)
        J_p = a_p * j0_p * sinh_ast(F_RT * eta_p)

        # =====================================================================
        # 6. Transport Tensors
        # =====================================================================
        N_s_n = -self.D_s_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -D_s_p * fx.grad(self.c_s_p, axis=self.r_p)
        
        i_s_n = -sig_n * fx.grad(self.phi_s_n)
        i_s_p = -sig_p * fx.grad(self.phi_s_p)
        
        c_L = ce_safe / 1000.0 
        D_e = (8.794e-11 * c_L**2 - 3.972e-10 * c_L + 4.862e-10)
        k_e = (0.1297 * c_L**3 - 2.51 * c_L**1.5 + 3.329 * c_L)
        
        D_eff_n, k_eff_n = D_e * (eps_n ** b_brug), k_e * (eps_n ** b_brug)
        D_eff_s, k_eff_s = D_e * (eps_s ** b_brug), k_e * (eps_s ** b_brug)
        D_eff_p, k_eff_p = D_e * (eps_p ** b_brug), k_e * (eps_p ** b_brug)
        
        # CRITICAL FIX: Applying the gradient to the safe, clamped variable eradicates phantom currents!
        ce_diff_term = (2.0 * R_const * T / F) * (1.0 - t_plus) * (fx.grad(ce_safe) / ce_safe)
        
        flux_ce_n = -D_eff_n * fx.grad(self.c_e)
        flux_ce_s = -D_eff_s * fx.grad(self.c_e)
        flux_ce_p = -D_eff_p * fx.grad(self.c_e)
        
        i_e_n = -k_eff_n * fx.grad(self.phi_e) + k_eff_n * ce_diff_term
        i_e_s = -k_eff_s * fx.grad(self.phi_e) + k_eff_s * ce_diff_term
        i_e_p = -k_eff_p * fx.grad(self.phi_e) + k_eff_p * ce_diff_term

        i_den = self.i_app / A_elec

        # =====================================================================
        # 7. Explicit Equation Targeting & Boundary Masking
        # =====================================================================
        return {
            "equations": {
                self.c_e: fx.Piecewise({
                    self.x_n: eps_n * fx.dt(self.c_e) == -fx.div(flux_ce_n) + (1.0 - t_plus) * J_n / F,
                    self.x_s: eps_s * fx.dt(self.c_e) == -fx.div(flux_ce_s),
                    self.x_p: eps_p * fx.dt(self.c_e) == -fx.div(flux_ce_p) + (1.0 - t_plus) * J_p / F
                }),
                
                self.phi_e: fx.Piecewise({
                    self.x_n: fx.div(i_e_n) == J_n,
                    self.x_s: fx.div(i_e_s) == 0.0,
                    self.x_p: fx.div(i_e_p) == J_p
                }),
                
                self.phi_s_n: fx.div(i_s_n) == -J_n,
                self.phi_s_p: fx.div(i_s_p) == -J_p,
                
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n),
                self.c_s_p: fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p),
                
                self.V_cell: self.V_cell == self.phi_s_p.right - self.phi_s_n.left
            },
            "boundaries": {
                self.phi_s_n: {"left": fx.Dirichlet(0.0)}, 
                
                i_s_n:        {"right": 0.0},
                i_s_p:        {"left": 0.0, "right": i_den}, 
                
                flux_ce_n:    {"left": 0.0},
                flux_ce_p:    {"right": 0.0},
                i_e_n:        {"left": 0.0},
                i_e_p:        {"right": 0.0},
                
                N_s_n:        {"left": 0.0, "right": J_n / (a_n * F)},
                N_s_p:        {"left": 0.0, "right": J_p / (a_p * F)},
            },
            "initial_conditions": {
                self.c_e: 1000.0,     
                
                # Equilibrium offsets calculated to zero out initial eta
                self.phi_s_n: 0.0,  
                self.phi_e: -0.092,
                self.phi_s_p: 4.182,
                
                self.c_s_n: 0.9014 * c_max_n,  
                self.c_s_p: 0.27 * c_max_p,
                
                self.V_cell: 4.182,    
                self.i_app: 0.0
            }
        }

if __name__ == "__main__":
    engine = fx.Engine(model=Chen2020_DFN(), target="cpu:serial", jacobian_bandwidth=0)
    
    discharge_rates = {
        "C/2":  {"current": 2.5, "D_s_n": 1.3e-14},
        "1C":   {"current": 5.0, "D_s_n": 3.3e-14},
        "1.5C": {"current": 7.5, "D_s_n": 6.3e-14}
    }
    
    results = {}
    
    for label, params in discharge_rates.items():
        print(f"Simulating {label} discharge + 2-hour relaxation...")
        protocol = Sequence([
            CC(rate=params["current"], until=engine.model.V_cell <= 2.5, time=15000),
            Rest(time=7200)
        ])
        
        res = engine.solve(protocol=protocol, parameters={"D_s_n": params["D_s_n"]})
        results[label] = res

    # =========================================================================
    # Replication of Figure 17 (Voltage Profiles)
    # =========================================================================
    fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    
    for i, (label, res) in enumerate(results.items()):
        t_hours = res["Time [s]"].data / 3600.0
        v_cell = res["V_cell"].data
        
        axs[i].plot(t_hours, v_cell, label="model", color="tab:orange", linewidth=2.5)
        
        axs[i].set_title(label, fontweight="bold", fontsize=12)
        axs[i].set_ylabel("Voltage [V]", fontsize=12)
        axs[i].set_ylim([2.4, 4.3])
        axs[i].grid(True, linestyle="--", alpha=0.5)
        
        if i == 0:
            axs[i].legend(loc="upper right", fontsize=11)
        if i == 2:
            axs[i].set_xlabel("t [h]", fontsize=12)
            
    fig.suptitle("P2D Simulation Validation (Replicating Fig. 17 Left Column)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()