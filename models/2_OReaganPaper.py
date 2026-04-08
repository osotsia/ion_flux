r"""
Thermal-Electrochemical Doyle-Fuller-Newman (DFN) Model

Reference:
O'Regan, K., Brosa Planella, F., Widanage, W. D., & Kendrick, E. (2022).
"Thermal-electrochemical parameters of a high energy lithium-ion cylindrical battery."
Electrochimica Acta.

This script rigorously implements the parameterization of the LG M50 21700 cell 
(NMC811 / Graphite-SiOy) directly from the experimental datasets provided in the paper.
It has been upgraded to a fully non-linear Thermal-Electrochemical model, coupling 
Arrhenius transport kinetics, entropic heating, and Joule losses to a lumped 0D thermal state.

The electrolyte transport properties ($D_e, \kappa_e$) use a simplified Arrhenius 
scaling rather than the incredibly dense, 9-term empirical Gasteiger polynomials (Eq 18-21) 
to keep the AST compilation efficient.

This model currently runs, but the output figure diverges from the figure in the paper. Still some bugs left.
"""

import math
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class ThermalDFN(fx.PDE):
    # =========================================================================
    # 1. Topological Sub-Meshing (LG M50 Dimensions, Table 8)
    # =========================================================================
    # Cell thickness: Negative = 85.2 um, Separator = 12 um, Positive = 75.6 um
    # Total Cell = 172.8 um. Using dx = 1.2 um ensures perfectly tiling indices.
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=144)
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=71, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=10, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=63, name="x_p")
    
    # Particle radii: Negative = 5.86 um, Positive = 5.22 um
    r_n = fx.Domain(bounds=(0, 5.86e-6), resolution=50, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5.22e-6), resolution=50, coord_sys="spherical", name="r_p") 
    
    # =========================================================================
    # 2. States (Table S2: Doyle-Fuller-Newman model unknowns)
    # =========================================================================
    c_e = fx.State(domain=cell, name="c_e")         # Electrolyte concentration
    phi_e = fx.State(domain=cell, name="phi_e")     # Electrolyte potential
    
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")  # Negative solid potential
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")  # Positive solid potential
    
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n") # Negative particle concentration
    c_s_p = fx.State(domain=x_p * r_p, name="c_s_p") # Positive particle concentration
    
    V_cell = fx.State(domain=None, name="V_cell")    # 0D Terminal Voltage
    i_app = fx.State(domain=None, name="i_app")      # 0D Terminal Current
    T_cell = fx.State(domain=None, name="T_cell")    # 0D Lumped Cell Temperature
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    # Double Layer Capacitance [F/m^2] (Numerical inertia for DAEs)
    C_dl = fx.Parameter(default=0.2, name="C_dl")
    
    def math(self):
        # =====================================================================
        # 3. Parameters (Table 8 & Physical Constants)
        # =====================================================================
        F, R_const, T_ref = 96485.0, 8.314, 298.15
        A_elec = 0.1024  # Active Electrode Area [m^2]
        L_cell = 172.8e-6 # Total cell thickness [m]
        
        # Microstructural Parameters (Table 8)
        eps_sn, eps_en = 0.75, 0.25      
        eps_es = 0.47                    
        eps_sp, eps_ep = 0.665, 0.335    
        b_brug = 1.5                     
        
        a_n = 3.0 * eps_sn / 5.86e-6
        a_p = 3.0 * eps_sp / 5.22e-6
        
        c_max_n = 29583.0
        c_max_p = 51765.0
        
        # Transport Parameters (Table 8)
        sig_eff_n = 215.0       

        # =====================================================================
        # 4. Helper Functions (AST Tracers)
        # =====================================================================
        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)
            
        def sinh_ast(x):
            x_max = 15.0
            cosh_max = 0.5 * (math.exp(x_max) + math.exp(-x_max))
            x_safe = fx.min(fx.max(x, -x_max), x_max)
            bulk_sinh = 0.5 * (fx.exp(x_safe) - fx.exp(-x_safe))
            return bulk_sinh + cosh_max * (x - x_safe)
            
        def arrh(Ea):
            """Arrhenius Temperature Scaling [Eq 6]."""
            return fx.exp((Ea / R_const) * (1.0 / T_ref - 1.0 / self.T_cell))

        # =====================================================================
        # 5. Thermodynamics & Kinetics (Arrhenius Coupled)
        # =====================================================================
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p) 
        
        x_n = fx.min(fx.max(c_surf_n / c_max_n, 1e-4), 0.9999)
        x_p = fx.min(fx.max(c_surf_p / c_max_p, 1e-4), 0.9999)
        ce_safe = fx.max(self.c_e, 1e-4)
        
        # OCV Functions (Eq. S7 and Eq. S8)
        U_n = (1.9793 * fx.exp(-39.3631 * x_n) + 0.2482 
               - 0.0909 * tanh_ast(29.8538 * (x_n - 0.1234)) 
               - 0.04478 * tanh_ast(14.9159 * (x_n - 0.2769)) 
               - 0.0205 * tanh_ast(30.4444 * (x_n - 0.6103)))
               
        U_p = (-0.8090 * x_p + 4.4875 
               - 0.0428 * tanh_ast(18.5138 * (x_p - 0.5542)) 
               - 17.7326 * tanh_ast(15.7890 * (x_p - 0.3117)) 
               + 17.5842 * tanh_ast(15.9308 * (x_p - 0.3120)))

        # Entropic Heating Term (dU/dT) [mV/K -> V/K] (Eq 16 & 17, Table S7)
        dUdT_n = 1e-3 * (-0.1112 * x_n + 0.02914 + 0.3561 * fx.exp(-((x_n - 0.08309)**2) / 0.004616))
        dUdT_p = 1e-3 * (0.04006 * fx.exp(-((x_p - 0.2828)**2) / 0.0009855) - 0.06656 * fx.exp(-((x_p - 0.8032)**2) / 0.02179))

        # Exchange Current Density w/ Arrhenius (Eq. 13 & 14)
        i0_n = 2.668 * (1.0 - x_n)**0.208 * x_n**0.792 * (ce_safe / 1000.0)**0.208 * arrh(40000.0)
        i0_p = 5.028 * (1.0 - x_p)**0.570 * x_p**0.430 * (ce_safe / 1000.0)**0.570 * arrh(24010.0)
        
        eta_n = self.phi_s_n - self.phi_e - U_n
        eta_p = self.phi_s_p - self.phi_e - U_p
        
        # Faradaic Currents (Exact Asymmetric Butler-Volmer kinetics)
        alpha_n, alpha_p = 0.792, 0.43
        F_RT = F / (R_const * self.T_cell)
        
        # Bound overpotentials to prevent exponential overflow during early Newton iterations
        eta_n_safe = fx.min(fx.max(eta_n, -1.0), 1.0)
        eta_p_safe = fx.min(fx.max(eta_p, -1.0), 1.0)
        
        j_n = a_n * i0_n * (fx.exp(alpha_n * F_RT * eta_n_safe) - fx.exp(-(1.0 - alpha_n) * F_RT * eta_n_safe))
        j_p = a_p * i0_p * (fx.exp(alpha_p * F_RT * eta_p_safe) - fx.exp(-(1.0 - alpha_p) * F_RT * eta_p_safe))

        j_dl_n = a_n * self.C_dl * (fx.dt(self.phi_s_n) - fx.dt(self.phi_e))
        j_dl_p = a_p * self.C_dl * (fx.dt(self.phi_s_p) - fx.dt(self.phi_e))

        j_tot_n = j_n + j_dl_n
        j_tot_p = j_p + j_dl_p

        # =====================================================================
        # 6. Transport PDEs (Thermally Coupled & Stoichiometry Dependent)
        # =====================================================================
        # Solid phase transport (O'Regan 2022 Stoichiometry fits)
        x_bulk_n = fx.min(fx.max(self.c_s_n / c_max_n, 1e-4), 0.9999)
        x_bulk_p = fx.min(fx.max(self.c_s_p / c_max_p, 1e-4), 0.9999)
        
        # Bypassing LLVM llvm.exp10.f64 intrinsic error using 10^x = exp(x * ln(10))
        ln10 = math.log(10.0)
        
        D_ref_n = fx.exp((
            11.17 * x_bulk_n - 15.11
            - 1.553 * fx.exp(-((x_bulk_n - 0.2031)**2) / 0.0006091)
            - 6.136 * fx.exp(-((x_bulk_n - 0.5375)**2) / 0.06438)
            - 9.725 * fx.exp(-((x_bulk_n - 0.9144)**2) / 0.0578)
            + 1.85 * fx.exp(-((x_bulk_n - 0.5953)**2) / 0.001356)
        ) * ln10) * 3.0321
        
        D_ref_p = fx.exp((
            -13.96
            - 0.9231 * fx.exp(-((x_bulk_p - 0.3216)**2) / 0.002534)
            - 0.4066 * fx.exp(-((x_bulk_p - 0.4532)**2) / 0.003926)
            - 0.993 * fx.exp(-((x_bulk_p - 0.8098)**2) / 0.09924)
        ) * ln10) * 2.7
        
        Ds_n = D_ref_n * fx.exp(2092.0 * (1.0 / 298.15 - 1.0 / self.T_cell))
        Ds_p = D_ref_p * fx.exp(1449.0 * (1.0 / 298.15 - 1.0 / self.T_cell))
        
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p)
        
        # Electronic conductivity (Thermally coupled for positive electrode)
        sig_eff_p = 0.8473 * fx.exp(3500.0 / R_const * (1.0 / 298.15 - 1.0 / self.T_cell))
        
        i_s_n = -sig_eff_n * fx.grad(self.phi_s_n)
        i_s_p = -sig_eff_p * fx.grad(self.phi_s_p)
        
        # Electrolyte transport (Landesfeind 2019 Empirical Polynomials)
        c_L = ce_safe / 1000.0
        T_L = self.T_cell
        
        # Transference Number
        t_plus = -12.8 - 6.12*c_L + 0.0821*T_L + 0.904*c_L**2 + 0.0318*c_L*T_L - 1.27e-4*T_L**2 + 0.0175*c_L**3 - 0.00312*c_L**2*T_L - 3.96e-5*c_L*T_L**2
        
        # Thermodynamic Factor
        TDF = 25.7 - 45.1*c_L - 0.177*T_L + 1.94*c_L**2 + 0.295*c_L*T_L + 3.08e-4*T_L**2 + 0.259*c_L**3 - 0.00946*c_L**2*T_L - 4.54e-4*c_L*T_L**2
        
        # Diffusivity (m2/s)
        De = 1010.0 * fx.exp(1.01 * c_L) * fx.exp(-1560.0 / T_L) * fx.exp(-487.0 * c_L / T_L) * 1e-10
        
        # Conductivity (S/m) (Converted from mS/cm by / 10.0)
        ke_A = 0.521 * (1.0 + (T_L - 228.0))
        ke_B = 1.0 - 1.06 * c_L**0.5 + 0.353 * (1.0 - 0.00359 * fx.exp(1000.0 / T_L)) * c_L
        ke_C = 1.0 + c_L**4 * (0.00148 * fx.exp(1000.0 / T_L))
        ke = (ke_A * c_L * ke_B / ke_C) / 10.0
        
        De_eff_n, ke_eff_n = De * (eps_en ** b_brug), ke * (eps_en ** b_brug)
        De_eff_s, ke_eff_s = De * (eps_es ** b_brug), ke * (eps_es ** b_brug)
        De_eff_p, ke_eff_p = De * (eps_ep ** b_brug), ke * (eps_ep ** b_brug)
        
        # TDF properly included in the diffusion migration term
        ce_diff_term = (2.0 * R_const * self.T_cell / F) * (1.0 - t_plus) * TDF * (fx.grad(self.c_e) / ce_safe)
        
        flux_ce_n = -De_eff_n * fx.grad(self.c_e)
        flux_ce_s = -De_eff_s * fx.grad(self.c_e)
        flux_ce_p = -De_eff_p * fx.grad(self.c_e)
        
        flux_phie_n = -ke_eff_n * fx.grad(self.phi_e) + ke_eff_n * ce_diff_term
        flux_phie_s = -ke_eff_s * fx.grad(self.phi_e) + ke_eff_s * ce_diff_term
        flux_phie_p = -ke_eff_p * fx.grad(self.phi_e) + ke_eff_p * ce_diff_term

        # =====================================================================
        # 7. Energy Conservation (Heat Generation & Cooling)
        # =====================================================================
        # Reversible (Entropic) & Irreversible (Reaction) Heating [W/m^3]
        Q_rxn_n = j_n * eta_n + j_n * self.T_cell * dUdT_n
        Q_rxn_p = j_p * eta_p + j_p * self.T_cell * dUdT_p
        
        # Ohmic Joule Heating [W/m^3] (Corrected negative sign for the ce_diff_term cross-multiplication)
        Q_ohm_n = sig_eff_n * (fx.grad(self.phi_s_n)**2) + ke_eff_n * (fx.grad(self.phi_e)**2) - ke_eff_n * ce_diff_term * fx.grad(self.phi_e)
        Q_ohm_s = ke_eff_s * (fx.grad(self.phi_e)**2) - ke_eff_s * ce_diff_term * fx.grad(self.phi_e)
        Q_ohm_p = sig_eff_p * (fx.grad(self.phi_s_p)**2) + ke_eff_p * (fx.grad(self.phi_e)**2) - ke_eff_p * ce_diff_term * fx.grad(self.phi_e)

        # Integrate spatially and average over the total cell volume
        Q_total_area = (fx.integral(Q_rxn_n + Q_ohm_n, over=self.x_n) + 
                        fx.integral(Q_ohm_s, over=self.x_s) + 
                        fx.integral(Q_rxn_p + Q_ohm_p, over=self.x_p))
        Q_vol = Q_total_area / L_cell
        
        # Convective Newton Cooling
        # Effective Cell Area (5.31e-3) / Volume (2.42e-5) = ~219.42 m^-1 (Table 8)
        h_cool = 10.0 
        Q_cool_vol = h_cool * 219.42 * (self.T_cell - T_ref)
        
        # Lumped Volumetric Heat Capacity (rho * Cp = 2682 * 866 = 2.3226e6 J / m^3 K)
        rho_cp = 2.3226e6 

        # =====================================================================
        # 8. Explicit Equation Targeting
        # =====================================================================
        return {
            "equations": {
                # --- Electrolyte Mass & Charge Conservation ---
                self.c_e: fx.Piecewise({
                    self.x_n: eps_en * fx.dt(self.c_e) == -fx.div(flux_ce_n) + (1.0 - t_plus) * j_n / F,
                    self.x_s: eps_es * fx.dt(self.c_e) == -fx.div(flux_ce_s),
                    self.x_p: eps_ep * fx.dt(self.c_e) == -fx.div(flux_ce_p) + (1.0 - t_plus) * j_p / F
                }),
                self.phi_e: fx.Piecewise({
                    self.x_n: fx.div(flux_phie_n) == j_tot_n,
                    self.x_s: fx.div(flux_phie_s) == 0.0,
                    self.x_p: fx.div(flux_phie_p) == j_tot_p
                }),
                
                # --- Solid Phase Mass & Charge Conservation ---
                self.phi_s_n: fx.div(i_s_n) == -j_tot_n,
                self.phi_s_p: fx.div(i_s_p) == -j_tot_p,
                
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n),
                self.c_s_p: fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p),
                
                # --- 0D Thermal & Electrical Dynamics ---
                self.T_cell: fx.dt(self.T_cell) == (Q_vol - Q_cool_vol) / rho_cp,
                self.V_cell: self.V_cell == self.phi_s_p.right - self.phi_s_n.left
            },
            
            # =================================================================
            # 9. Boundary Conditions
            # =================================================================
            "boundaries": {
                flux_ce_n:    {"left": 0.0},
                flux_ce_p:    {"right": 0.0},
                
                self.phi_s_n: {"left": fx.Dirichlet(0.0)}, 
                i_s_n:        {"right": 0.0},
                i_s_p:        {"left": 0.0, "right": self.i_app / A_elec}, 
                
                flux_phie_n:  {"left": 0.0},
                flux_phie_p:  {"right": 0.0},
                
                N_s_n:        {"left": 0.0, "right": j_n / (a_n * F)},
                N_s_p:        {"left": 0.0, "right": j_p / (a_p * F)},
            },
            
            # =================================================================
            # 10. Initial Conditions
            # =================================================================
            "initial_conditions": {
                self.c_e: 1000.0,     
                self.phi_s_n: 0.0,  
                self.phi_e: -0.103,
                self.phi_s_p: 4.07,
                self.c_s_n: 28866.0,  
                self.c_s_p: 13975.0,
                self.T_cell: 298.15,
                self.V_cell: 4.173,    
                self.i_app: 0.0
            }
        }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    # Bandwidth=0 signals FAER LU/GMRES handles internal cross-domain sparsity natively
    engine = fx.Engine(model=ThermalDFN(), target="cpu:serial", jacobian_bandwidth=0, solver_backend="native")
    
    # Nominal capacity = 5 Ah
    rates = {
        "0.5C": 2.5, 
        "1C": 5.0, 
        "2C": 10.0
    }
    
    results = {}
    start_time = time.perf_counter()

    for name, current in rates.items():
        print(f"Simulating {name} discharge + relaxation...")
        
        # Protocol: Discharge to 2.5V, followed by 2-hour Rest
        protocol = Sequence([
            CC(rate=current, until=engine.model.V_cell <= 2.5, time=15000),
            Rest(time=7200) # 2 hours of relaxation
        ])
        
        res = engine.solve(protocol=protocol)
        results[name] = res

    elapsed = time.perf_counter() - start_time
    print(f"✅ Completed in {elapsed:.4f} seconds.")

    # -------------------------------------------------------------------------
    # Replication of Paper Figure 15 (Voltage and Temperature vs Time)
    # -------------------------------------------------------------------------
    fig, (ax_v, ax_t) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {"0.5C": "tab:blue", "1C": "tab:orange", "2C": "tab:green"}
    
    for name, res in results.items():
        t_seconds = res["Time [s]"].data
        v_cell = res["V_cell"].data
        t_celsius = res["T_cell"].data - 273.15 # Kelvin to Celsius
        
        # Plotting exactly as shown in O'Regan et al. Figure 15
        ax_v.plot(t_seconds, v_cell, label=f"{name}", color=colors[name], linewidth=2)
        ax_t.plot(t_seconds, t_celsius, label=f"{name}", color=colors[name], linewidth=2)

    # Aesthetics to mirror the paper
    ax_v.set_title("LG M50 21700 Voltage Profiles", fontsize=14, fontweight="bold")
    ax_v.set_xlabel("Time / s", fontsize=12)
    ax_v.set_ylabel("Voltage / V", fontsize=12)
    ax_v.set_xlim(left=0)
    ax_v.set_ylim([2.4, 4.3])
    ax_v.legend(fontsize=12)
    ax_v.grid(True, linestyle="--", alpha=0.7)
    
    ax_t.set_title("LG M50 21700 Temperature Profiles", fontsize=14, fontweight="bold")
    ax_t.set_xlabel("Time / s", fontsize=12)
    ax_t.set_ylabel("Cell temperature / °C", fontsize=12)
    ax_t.set_xlim(left=0)
    ax_t.legend(fontsize=12)
    ax_t.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # You can also launch the interactive 2x4 full-state dashboard for a specific run!
    # print("Launching comprehensive internal state dashboard for 2C...")
    # results["2C"].plot_dashboard()