import math
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class DFN(fx.PDE):
    """
    1D-1D Doyle-Fuller-Newman model.
    Strictly reflects the parameterization of the LG M50 21700 cell 
    (NMC811 / Graphite-SiOy) from O'Regan et al. (2022).
    """
    # -------------------------------------------------------------------------
    # 1. Topological Sub-Meshing (LG M50 Dimensions, Table 8)
    # Cell thickness: Negative = 85.2 um, Separator = 12 um, Positive = 75.6 um
    # Total Cell = 172.8 um. Using dx = 1.2 um ensures perfectly tiling indices.
    # -------------------------------------------------------------------------
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=144)
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=71, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=10, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=63, name="x_p")
    
    # Particle radii: Negative = 5.86 um, Positive = 5.22 um
    r_n = fx.Domain(bounds=(0, 5.86e-6), resolution=10, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5.22e-6), resolution=10, coord_sys="spherical", name="r_p") 
    
    # -------------------------------------------------------------------------
    # 2. States 
    # -------------------------------------------------------------------------
    c_e = fx.State(domain=cell, name="c_e")
    
    # Clamps restored to prevent intermediate spatial migration (grad phi) explosions
    phi_e = fx.State(domain=cell, name="phi_e", max_newton_step=0.05)
    phi_s_n = fx.State(domain=x_n, name="phi_s_n", max_newton_step=0.05)
    phi_s_p = fx.State(domain=x_p, name="phi_s_p", max_newton_step=0.05)
    V_cell = fx.State(domain=None, name="V_cell", max_newton_step=0.05) 
    
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n")
    c_s_p = fx.State(domain=x_p * r_p, name="c_s_p")
    i_app = fx.State(domain=None, name="i_app")
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    # NEW PARAMETER: Double Layer Capacitance [F/m^2]
    # Provides physical inertia to break the Index-1 DAE loop during severe depletion.
    C_dl = fx.Parameter(default=0.2, name="C_dl")
    
    def math(self):
        # Physical Constants
        F = 96485.0
        R_const = 8.314
        T = 298.15
        
        # Microstructural Parameters (Table 8)
        eps_sn, eps_en = 0.75, 0.25
        eps_es = 0.47
        eps_sp, eps_ep = 0.665, 0.335
        b_brug = 1.5
        
        # Specific Interfacial Area (a = 3 * eps_s / R_p)
        a_n = 3.0 * eps_sn / 5.86e-6
        a_p = 3.0 * eps_sp / 5.22e-6
        
        # Maximum Concentrations (Table 8)
        c_max_n = 29583.0
        c_max_p = 51765.0
        
        # Solid Conductivities (Table 8)
        sig_eff_n = 215.0
        sig_eff_p = 0.847
        
        # Electrolyte Parameters 
        De = 3e-10
        ke = 1.0
        t_plus = 0.38
        
        # Active Electrode Area 
        A_elec = 0.1024 
        
        # Solid Diffusion Coefficients 
        Ds_n = 3.3e-14
        Ds_p = 4.0e-15

        # ---------------------------------------------------------------------
        # Pre-Calculate EXACT t=0 Equilibrium Floats for Initial Conditions
        # ---------------------------------------------------------------------
        c_n_init = 28866.0
        c_p_init = 13975.0
        
        x_n_init = c_n_init / c_max_n
        x_p_init = c_p_init / c_max_p
        
        Un_0 = (1.9793 * math.exp(-39.3631 * x_n_init) + 0.2482 
               - 0.0909 * math.tanh(29.8538 * (x_n_init - 0.1234)) 
               - 0.04478 * math.tanh(14.9159 * (x_n_init - 0.2769)) 
               - 0.0205 * math.tanh(30.4444 * (x_n_init - 0.6103)))
               
        Up_0 = (-0.8090 * x_p_init + 4.4875 
               - 0.0428 * math.tanh(18.5138 * (x_p_init - 0.5542)) 
               - 17.7326 * math.tanh(15.7890 * (x_p_init - 0.3117)) 
               + 17.5842 * math.tanh(15.9308 * (x_p_init - 0.3120)))

        # ---------------------------------------------------------------------
        # Helper Functions (AST Tracers)
        # ---------------------------------------------------------------------
        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)
            
        def sinh_ast(x):
            # C1-Continuous Linear Extrapolation for sinh(x)
            # Evaluates exactly within physical bounds, ensuring flawless Jacobians
            # while shielding the f64 register from BDF polynomial predictor explosions.
            x_max = 15.0
            cosh_max = 0.5 * (math.exp(x_max) + math.exp(-x_max))
            
            x_safe = fx.min(fx.max(x, -x_max), x_max)
            bulk_sinh = 0.5 * (fx.exp(x_safe) - fx.exp(-x_safe))
            
            return bulk_sinh + cosh_max * (x - x_safe)

        # ---------------------------------------------------------------------
        # Thermodynamics & Kinetics (AST Nodes)
        # ---------------------------------------------------------------------
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p) 
        
        x_n = fx.min(fx.max(c_surf_n / c_max_n, 1e-4), 0.9999)
        x_p = fx.min(fx.max(c_surf_p / c_max_p, 1e-4), 0.9999)
        ce_safe = fx.max(self.c_e, 1e-4)
        
        # OCV Functions (O'Regan Eq S7 & S8)
        U_n = (1.9793 * fx.exp(-39.3631 * x_n) + 0.2482 
               - 0.0909 * tanh_ast(29.8538 * (x_n - 0.1234)) 
               - 0.04478 * tanh_ast(14.9159 * (x_n - 0.2769)) 
               - 0.0205 * tanh_ast(30.4444 * (x_n - 0.6103)))
               
        U_p = (-0.8090 * x_p + 4.4875 
               - 0.0428 * tanh_ast(18.5138 * (x_p - 0.5542)) 
               - 17.7326 * tanh_ast(15.7890 * (x_p - 0.3117)) 
               + 17.5842 * tanh_ast(15.9308 * (x_p - 0.3120)))

        # Exchange Current Density (O'Regan Table S6)
        i0_n = 10000.0 * 0.0002668 * (1.0 - x_n)**0.208 * x_n**0.792 * (ce_safe / 1000.0)**0.208
        i0_p = 10000.0 * 0.0005028 * (1.0 - x_p)**0.570 * x_p**0.430 * (ce_safe / 1000.0)**0.570
        
        eta_n = self.phi_s_n - self.phi_e - U_n
        eta_p = self.phi_s_p - self.phi_e - U_p
        
        # 1. Faradaic Currents (Intercalation)
        j_n = a_n * i0_n * sinh_ast((F / (2.0 * R_const * T)) * eta_n)
        j_p = a_p * i0_p * sinh_ast((F / (2.0 * R_const * T)) * eta_p)

        # 2. Double Layer Currents (Capacitive Displacement)
        j_dl_n = a_n * self.C_dl * (fx.dt(self.phi_s_n) - fx.dt(self.phi_e))
        j_dl_p = a_p * self.C_dl * (fx.dt(self.phi_s_p) - fx.dt(self.phi_e))

        # 3. Total Volumetric Currents
        j_tot_n = j_n + j_dl_n
        j_tot_p = j_p + j_dl_p

        # ---------------------------------------------------------------------
        # Tensors (AST Nodes)
        # ---------------------------------------------------------------------
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p)
        
        i_s_n = -sig_eff_n * fx.grad(self.phi_s_n)
        i_s_p = -sig_eff_p * fx.grad(self.phi_s_p)
        
        De_eff_n, ke_eff_n = De * (eps_en ** b_brug), ke * (eps_en ** b_brug)
        De_eff_s, ke_eff_s = De * (eps_es ** b_brug), ke * (eps_es ** b_brug)
        De_eff_p, ke_eff_p = De * (eps_ep ** b_brug), ke * (eps_ep ** b_brug)
        
        ce_diff_term = (2.0 * R_const * T / F) * (1.0 - t_plus) * (fx.grad(self.c_e) / ce_safe)
        
        flux_ce_n = -De_eff_n * fx.grad(self.c_e)
        flux_ce_s = -De_eff_s * fx.grad(self.c_e)
        flux_ce_p = -De_eff_p * fx.grad(self.c_e)
        
        flux_phie_n = -ke_eff_n * fx.grad(self.phi_e) + ke_eff_n * ce_diff_term
        flux_phie_s = -ke_eff_s * fx.grad(self.phi_e) + ke_eff_s * ce_diff_term
        flux_phie_p = -ke_eff_p * fx.grad(self.phi_e) + ke_eff_p * ce_diff_term

        # ---------------------------------------------------------------------
        # Explicit Equation Targeting
        # ---------------------------------------------------------------------
        return {
            "equations": {
                # --- Electrolyte ---
                self.c_e: fx.Piecewise({
                    # DL charging is non-faradaic and does not deplete Li+
                    self.x_n: eps_en * fx.dt(self.c_e) == -fx.div(flux_ce_n) + (1.0 - t_plus) * j_n / F,
                    self.x_s: eps_es * fx.dt(self.c_e) == -fx.div(flux_ce_s),
                    self.x_p: eps_ep * fx.dt(self.c_e) == -fx.div(flux_ce_p) + (1.0 - t_plus) * j_p / F
                }),
                self.phi_e: fx.Piecewise({
                    # Electric fields map directly to Total Current (Faradaic + DL)
                    # Note: Because the compiler sees fx.dt(phi_e) inside j_tot, it 
                    # automatically flags the anode/cathode regions as Stiff ODEs, 
                    # while leaving the separator region (j_tot=0.0) as an Algebraic DAE.
                    self.x_n: fx.div(flux_phie_n) == j_tot_n,
                    self.x_s: fx.div(flux_phie_s) == 0.0,
                    self.x_p: fx.div(flux_phie_p) == j_tot_p
                }),
                
                # --- Solid Phase ---
                self.phi_s_n: fx.div(i_s_n) == -j_tot_n,
                self.phi_s_p: fx.div(i_s_p) == -j_tot_p,
                
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n),
                self.c_s_p: fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p),
                
                # --- Global Algebraic ---
                self.V_cell: self.V_cell == self.phi_s_p.right - self.phi_s_n.left
            },
            
            # -----------------------------------------------------------------
            # Explicit Boundaries (Dirichlet on States, Neumann on Tensors)
            # -----------------------------------------------------------------
            "boundaries": {
                flux_ce_n:    {"left": 0.0},
                flux_ce_p:    {"right": 0.0},
                
                self.phi_s_n: {"left": fx.Dirichlet(0.0)}, # Grounded anchor node
                i_s_n:        {"right": 0.0},
                i_s_p:        {"left": 0.0, "right": self.i_app / A_elec}, 
                
                flux_phie_n:  {"left": 0.0},
                flux_phie_p:  {"right": 0.0},
                
                # Only Faradaic current crosses the particle surface to intercalate
                N_s_n:        {"left": 0.0, "right": j_n / (a_n * F)},
                N_s_p:        {"left": 0.0, "right": j_p / (a_p * F)},
            },
            
            # -----------------------------------------------------------------
            # Exact Equilibrium Initialization (Pass exact evaluated floats)
            # -----------------------------------------------------------------
            "initial_conditions": {
                self.c_e: 1000.0,     
                self.phi_s_n: 0.0,  
                self.phi_e: -Un_0,
                self.phi_s_p: Up_0 - Un_0,
                self.c_s_n: c_n_init,  
                self.c_s_p: c_p_init,
                self.V_cell: Up_0 - Un_0,    
                self.i_app: 0.0
            }
        }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Bandwidth=0 signals FAER LU/GMRES handles internal cross-domain sparsity natively
    engine = fx.Engine(model=DFN(), target="cpu:serial", jacobian_bandwidth=0, solver_backend="native")
    
    # Nominal capacity = 5 Ah
    rates = {
        "0.5C": 2.5, 
        "1C": 5.0, 
        "2C": 10.0
    }
    
    results = {}
    
    for name, current in rates.items():
        print(f"Simulating {name} discharge...")
        # Simulate Figure 15: Discharges to 2.5V cutoff
        protocol = Sequence([
            CC(rate=current, until=engine.model.V_cell <= 2.5, time=7200)
        ])
        
        res = engine.solve(protocol=protocol)
        results[name] = res

    # Plot Figure 15 style (Voltage vs Capacity/Time)
    plt.figure(figsize=(10, 6))
    
    colors = {"0.5C": "tab:blue", "1C": "tab:orange", "2C": "tab:green"}
    
    for name, res in results.items():
        t_seconds = res["Time [s]"].data
        v_cell = res["V_cell"].data
        
        # Calculate Capacity discharged in Ah
        capacity_ah = (t_seconds * rates[name]) / 3600.0
        
        plt.plot(capacity_ah, v_cell, label=f"{name} Discharge", color=colors[name], linewidth=2)

    plt.title("LG M50 21700 Discharge Curves (Simulating O'Regan et al. Fig 15)", fontsize=14)
    plt.xlabel("Discharge Capacity [Ah]", fontsize=12)
    plt.ylabel("Terminal Voltage [V]", fontsize=12)
    plt.xlim(left=0)
    plt.ylim([2.4, 4.3])
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # You can also launch the interactive 2x4 full-state dashboard for a specific run!
    # print("Launching comprehensive internal state dashboard for 1C...")
    # results["1C"].plot_dashboard()