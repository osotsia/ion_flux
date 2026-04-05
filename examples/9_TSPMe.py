import math
import numpy as np
import matplotlib.pyplot as plt
import ion_flux as fx
from ion_flux.protocols import Sequence, CC

class ExactTSPMe(fx.PDE):
    """
    Thermal Single Particle Model with Electrolyte (TSPMe).
    Strictly reflects the asymptotic derivation of Brosa Planella et al. (2021).
    Utilizes exact analytical integrations for Ohmic losses (Eq. 16, Eq. 17) to 
    maximize implicit solver stability and performance.
    """
    # -------------------------------------------------------------------------
    # 1. Topology
    # -------------------------------------------------------------------------
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=144)
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=71, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=10, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=63, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 5.86e-6), resolution=15, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5.22e-6), resolution=15, coord_sys="spherical", name="r_p") 
    
    # -------------------------------------------------------------------------
    # 2. States 
    # -------------------------------------------------------------------------
    c_e = fx.State(domain=cell, name="c_e")
    c_s_n = fx.State(domain=r_n, name="c_s_n")
    c_s_p = fx.State(domain=r_p, name="c_s_p")
    
    # Context Anchors: These dummy states explicitly bind the sub-regions to the AST.
    # This guarantees fx.integral() correctly applies spatial offsets (e.g. +80 for x_p)
    # when pulling values from the global c_e array.
    ctx_n = fx.State(domain=x_n, name="ctx_n")
    ctx_s = fx.State(domain=x_s, name="ctx_s")
    ctx_p = fx.State(domain=x_p, name="ctx_p")
    
    T_cell = fx.State(domain=None, name="T_cell") # 0D Lumped Thermal ODE
    V_cell = fx.State(domain=None, name="V_cell") # 0D Algebraic Voltage Constraint
    i_app = fx.State(domain=None, name="i_app")   # 0D Cycler terminal 
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        # ---------------------------------------------------------------------
        # Parameters (Table 1: LG M50)
        # ---------------------------------------------------------------------
        F, R_const, T_amb = 96485.0, 8.314, 298.15
        
        L_n, L_s, L_p = 85.2e-6, 12.0e-6, 75.6e-6
        L_cell = L_n + L_s + L_p
        A_elec = 0.1024 
        
        eps_n, eps_s, eps_p = 0.25, 0.47, 0.335
        b_brug = 1.5
        
        a_n = 3.84e5
        a_p = 3.82e5
        c_max_n, c_max_p = 33133.0, 63104.0
        sig_n, sig_p = 215.0, 0.18
        m_n, m_p = 6.48e-7, 3.42e-6
        
        De_ref, sig_e_ref, t_plus = 3e-10, 1.0, 0.2594
        
        theta_heat = 2.85e6  
        h_cool = 20.0        
        a_cool = 219.42      

        def arrh(Ea):
            return fx.exp((Ea / R_const) * (1.0 / T_amb - 1.0 / self.T_cell))

        Ds_n = 3.3e-14 * arrh(17393.0)
        Ds_p = 4.0e-15 * arrh(12047.0)

        def arcsinh_ast(x):
            return fx.log(x + (x**2 + 1.0)**0.5)

        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)

        # ---------------------------------------------------------------------
        # Core SPM Physics & Thermodynamics
        # ---------------------------------------------------------------------
        i_den = self.i_app / A_elec
        
        j_vol_n = i_den / L_n
        j_vol_p = -i_den / L_p
        
        c_surf_n = fx.min(fx.max(self.c_s_n.boundary("right", domain=self.r_n), 10.0), c_max_n - 10.0)
        c_surf_p = fx.min(fx.max(self.c_s_p.boundary("right", domain=self.r_p), 10.0), c_max_p - 10.0)
        ce_safe = fx.max(self.c_e, 1.0)
        
        x_n = c_surf_n / c_max_n
        x_p = c_surf_p / c_max_p

        U_n = (1.9793 * fx.exp(-39.3631 * x_n) + 0.2482 
               - 0.0909 * tanh_ast(29.8538 * (x_n - 0.1234)) 
               - 0.04478 * tanh_ast(14.9159 * (x_n - 0.2769)) 
               - 0.0205 * tanh_ast(30.4444 * (x_n - 0.6103)))
               
        U_p = (-0.8090 * x_p + 4.4875 
               - 0.0428 * tanh_ast(18.5138 * (x_p - 0.5542)) 
               - 17.7326 * tanh_ast(15.7890 * (x_p - 0.3117)) 
               + 17.5842 * tanh_ast(15.9308 * (x_p - 0.3120)))
        
        U_eq = U_p - U_n
        
        j0_n = m_n * (ce_safe * c_surf_n * (c_max_n - c_surf_n))**0.5 * arrh(40000.0)
        j0_p = m_p * (ce_safe * c_surf_p * (c_max_p - c_surf_p))**0.5 * arrh(24000.0)

        # ---------------------------------------------------------------------
        # Voltage Resolution (Eq. 8a-8f)
        # ---------------------------------------------------------------------
        term_n = i_den / (a_n * L_n * j0_n)
        term_p = i_den / (a_p * L_p * j0_p)
        
        eta_r_n = - (2.0 * R_const * self.T_cell / F) * (fx.integral(arcsinh_ast(term_n), over=self.x_n) / L_n)
        eta_r_p = - (2.0 * R_const * self.T_cell / F) * (fx.integral(arcsinh_ast(term_p), over=self.x_p) / L_p)
        eta_r = eta_r_n + eta_r_p
        
        eta_e = (1.0 - t_plus) * (2.0 * R_const * self.T_cell / F) * (
            fx.integral(fx.log(ce_safe), over=self.x_p) / L_p - 
            fx.integral(fx.log(ce_safe), over=self.x_n) / L_n
        )
        
        R_s_ohm = (L_n / sig_n + L_p / sig_p) / 3.0
        dPhi_s = -i_den * R_s_ohm
        
        term_n_e = L_n / (eps_n ** b_brug)
        term_s_e = 3.0 * L_s / (eps_s ** b_brug)
        term_p_e = L_p / (eps_p ** b_brug)
        R_e_ohm = (term_n_e + term_s_e + term_p_e) / (3.0 * sig_e_ref)
        dPhi_e = -i_den * R_e_ohm
                  
        V_total = U_eq + eta_r + eta_e + dPhi_e + dPhi_s

        # ---------------------------------------------------------------------
        # Heat Generation (Eq. 5c-5f)
        # ---------------------------------------------------------------------
        Q_s = (i_den ** 2) * R_s_ohm / L_cell
        Q_e = (i_den ** 2) * R_e_ohm / L_cell - (i_den * eta_e / L_cell)
        Q_irr = i_den * fx.abs(eta_r) / L_cell 
        
        Q_cool = h_cool * a_cool * (self.T_cell - T_amb)
        Q_tot = Q_s + Q_e + Q_irr

        # ---------------------------------------------------------------------
        # Tensors
        # ---------------------------------------------------------------------
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p)
        
        De_eff_n = De_ref * (eps_n ** b_brug)
        De_eff_s = De_ref * (eps_s ** b_brug)
        De_eff_p = De_ref * (eps_p ** b_brug)
        
        flux_ce_n = -De_eff_n * fx.grad(self.c_e)
        flux_ce_s = -De_eff_s * fx.grad(self.c_e)
        flux_ce_p = -De_eff_p * fx.grad(self.c_e)

        # ---------------------------------------------------------------------
        # Equation Targeting
        # ---------------------------------------------------------------------
        return {
            "equations": {
                self.c_e: fx.Piecewise({
                    self.x_n: eps_n * fx.dt(self.c_e) == -fx.div(flux_ce_n) + (1.0 - t_plus) * j_vol_n / F,
                    self.x_s: eps_s * fx.dt(self.c_e) == -fx.div(flux_ce_s),
                    self.x_p: eps_p * fx.dt(self.c_e) == -fx.div(flux_ce_p) + (1.0 - t_plus) * j_vol_p / F
                }),
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n),
                self.c_s_p: fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p),
                
                # Resolving context explicitly 
                self.ctx_n: fx.dt(self.ctx_n) == 0.0,
                self.ctx_s: fx.dt(self.ctx_s) == 0.0,
                self.ctx_p: fx.dt(self.ctx_p) == 0.0,
                
                self.T_cell: fx.dt(self.T_cell) == (Q_tot - Q_cool) / theta_heat,
                self.V_cell: self.V_cell == V_total
            },
            
            "boundaries": {
                flux_ce_n: {"left": 0.0},
                flux_ce_p: {"right": 0.0},
                N_s_n: {"left": 0.0, "right": j_vol_n / (a_n * F)},
                N_s_p: {"left": 0.0, "right": j_vol_p / (a_p * F)},
            },
            
            "initial_conditions": {
                self.c_e: 1000.0,     
                self.c_s_n: 29866.0,  
                self.c_s_p: 17038.0,
                self.ctx_n: 0.0, self.ctx_s: 0.0, self.ctx_p: 0.0,
                self.T_cell: 298.15,   
                self.V_cell: 4.10, 
                self.i_app: 0.0
            }
        }

if __name__ == "__main__":
    
    engine = fx.Engine(model=ExactTSPMe(), target="cpu:serial", jacobian_bandwidth=0)
    
    rates = {"0.5C": 2.5, "1C": 5.0, "2C": 10.0}
    results = {}
    
    for name, current in rates.items():
        print(f"Simulating {name} discharge...")
        protocol = Sequence([
            CC(rate=current, until=engine.model.V_cell <= 2.5, time=7200)
        ])
        results[name] = engine.solve(protocol=protocol)

    # -------------------------------------------------------------------------
    # Replication of Paper Figure 5
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {"0.5C": "tab:blue", "1C": "tab:orange", "2C": "tab:green"}
    
    for name, res in results.items():
        capacity_ah = (res["Time [s]"].data * rates[name]) / 3600.0
        
        axs[0].plot(capacity_ah, res["V_cell"].data, label=f"{name}", color=colors[name], linewidth=2)
        # Kelvin to Celsius conversion
        axs[1].plot(capacity_ah, res["T_cell"].data - 273.15, label=f"{name}", color=colors[name], linewidth=2)

    axs[0].set_title("TSPMe Validation: LG M50 at 25°C (Replicating Fig. 5)", fontsize=14, fontweight="bold")
    axs[0].set_ylabel("Terminal Voltage [V]", fontsize=12)
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend(fontsize=11)
    
    axs[1].set_ylabel("Cell Temperature [°C]", fontsize=12)
    axs[1].set_xlabel("Discharge Capacity [Ah]", fontsize=12)
    axs[1].grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()