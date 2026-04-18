r"""
Isothermal Pseudo-Two-Dimensional (P2D) Doyle-Fuller-Newman Model

Reference:
Chen, C.-H., Brosa Planella, F., O'Regan, K., Gastol, D., Widanage, W. D., & Kendrick, E. (2020).
"Development of experimental techniques for parameterization of multi-scale lithium-ion battery models."
Journal of The Electrochemical Society, 167(8), 080534.

This module implements the comprehensive 35-parameter dataset for the commercial
LG M50 21700 cell (NMC 811 / Graphite-SiOx). It leverages hierarchical macro-micro
composite domains to model solid diffusion, exposing severe electrolyte starvation,
reaction heterogeneity, and cathode surface saturation during high C-rate discharges.
"""

import numpy as np
import matplotlib.pyplot as plt
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class Chen2020_DFN(fx.PDE):
    # =========================================================================
    # 1. Topological Sub-Meshing (Table II & Table VII)
    # =========================================================================
    # L_cell = 172.8 um (Total thickness). Discretized into 72 macro nodes.
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=72)
    
    # Regional Sub-Domains (L_n = 85.2 um, L_s = 12.0 um, L_p = 75.6 um)
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=35, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=6, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=31, name="x_p")
    
    # Microscopic Particle Domains (R_n = 5.86 um, R_p = 5.22 um)
    r_n = fx.Domain(bounds=(0, 5.86e-6), resolution=15, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5.22e-6), resolution=15, coord_sys="spherical", name="r_p") 
    
    # =========================================================================
    # 2. States & Observables
    # =========================================================================
    c_e = fx.State(domain=cell, name="c_e")         # Electrolyte Li+ concentration [mol/m^3]
    
    phi_e = fx.State(domain=cell, name="phi_e")     # Electrolyte potential [V]
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")  # Solid potential (Anode) [V]
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")  # Solid potential (Cathode) [V]
    
    # Hierarchical states (Macro x Micro) representing solid particle concentrations [mol/m^3]
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n") 
    c_s_p = fx.State(domain=x_p * r_p, name="c_s_p") 
    
    V_cell = fx.State(domain=None, name="V_cell")    # 0D Terminal Voltage [V]
    i_app = fx.State(domain=None, name="i_app")      # 0D Terminal Current [A]
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    # Tunable parameter (Table IX) to match discharge final voltage curves
    D_s_n = fx.Parameter(default=3.3e-14, name="D_s_n")

    # Analytical Telemetry Trackers for Volumetric Current Densities (J_k) [A/m^3]
    J_n_obs = fx.Observable(domain=x_n, name="J_n_obs")
    J_p_obs = fx.Observable(domain=x_p, name="J_p_obs")
    
    def math(self):
        # =====================================================================
        # 3. Parameters (Table VII & Table IX Tuned Constants)
        # =====================================================================
        F, R_const, T = 96485.0, 8.314, 298.15
        A_elec = 0.1027  # Electrode Plate Area [m^2]
        
        # Microstructural Volume Fractions
        eps_n, eps_s, eps_p = 0.25, 0.47, 0.335
        eps_act_n, eps_act_p = 0.75, 0.665
        
        # Bruggeman tortuosity scaling (Noted on Page 18: "Theoretical value of 1.5 for packed spheres")
        b_brug = 1.5                     
        
        # Specific active surface area [1/m] (Equation 26: a_k = 3 * eps_act_k / R_k)
        a_n = 3.0 * eps_act_n / 5.86e-6
        a_p = 3.0 * eps_act_p / 5.22e-6
        
        # Maximum Solid Concentrations [mol/m^3]
        c_max_n = 33133.0
        c_max_p = 63104.0
        
        # Transport & Kinetic Constants
        sig_n, sig_p = 215.0, 0.18    # Electronic conductivities [S/m]
        D_s_p = 4.0e-15               # Cathode solid diffusivity [m^2/s]
        t_plus = 0.2594               # Transference number
        k_n, k_p = 6.48e-7, 3.42e-6   # Reaction rate constants [A/m^2 (m^3/mol)^1.5]

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
        # Extract surface concentrations natively from the Micro-Domains
        c_surf_n = fx.min(fx.max(self.c_s_n.boundary("right", domain=self.r_n), 10.0), c_max_n - 10.0)
        c_surf_p = fx.min(fx.max(self.c_s_p.boundary("right", domain=self.r_p), 10.0), c_max_p - 10.0)
        
        # Active particle stoichiometries
        x_n = c_surf_n / c_max_n
        x_p = c_surf_p / c_max_p
        
        # Bound electrolyte concentration to prevent log singularities AND runaway Newton feedback loops
        ce_safe = fx.min(fx.max(self.c_e, 10.0), 5000.0)
        
        # Equilibrium Open Circuit Potentials [V] (Equations 8 & 9)
        U_n = (1.9793 * fx.exp(-39.3631 * x_n) + 0.2482 
               - 0.0909 * tanh_ast(29.8538 * (x_n - 0.1234)) 
               - 0.04478 * tanh_ast(14.9159 * (x_n - 0.2769)) 
               - 0.0205 * tanh_ast(30.4444 * (x_n - 0.6103)))
               
        U_p = (-0.8090 * x_p + 4.4875 
               - 0.0428 * tanh_ast(18.5138 * (x_p - 0.5542)) 
               - 17.7326 * tanh_ast(15.7890 * (x_p - 0.3117)) 
               + 17.5842 * tanh_ast(15.9308 * (x_p - 0.3120)))

        # Exchange Current Densities [A/m^2] (Table I: Reaction Kinetics)
        j0_n = k_n * (ce_safe * c_surf_n * (c_max_n - c_surf_n))**0.5
        j0_p = k_p * (ce_safe * c_surf_p * (c_max_p - c_surf_p))**0.5
        
        # Local Overpotentials [V] (Table I)
        eta_n = self.phi_s_n - self.phi_e - U_n
        eta_p = self.phi_s_p - self.phi_e - U_p
        
        # Volumetric Butler-Volmer Reaction Current Densities (J_k) [A/m^3] (Table I)
        F_RT = F / (2.0 * R_const * T)  # Using alpha = 0.5 symmetry
        J_n = a_n * j0_n * sinh_ast(F_RT * eta_n)
        J_p = a_p * j0_p * sinh_ast(F_RT * eta_p)

        # =====================================================================
        # 6. Transport Tensors
        # =====================================================================
        # Solid Fickian Diffusion Fluxes [mol/m^2 s] (Table I: Mass Conservation)
        N_s_n = -self.D_s_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -D_s_p * fx.grad(self.c_s_p, axis=self.r_p)
        
        # Solid Ohmic Current Fluxes [A/m^2] (Table I: Charge Conservation)
        i_s_n = -sig_n * fx.grad(self.phi_s_n)
        i_s_p = -sig_p * fx.grad(self.phi_s_p)
        
        # Empirical Electrolyte Formulations (Equations 23 & 24)
        c_L = ce_safe / 1000.0  # Convert to mol/dm^3 for the polynomial fits
        D_e = (8.794e-11 * c_L**2 - 3.972e-10 * c_L + 4.862e-10)
        k_e = (0.1297 * c_L**3 - 2.51 * c_L**1.5 + 3.329 * c_L)
        
        # Effective Electrolyte Diffusivities & Conductivities (Bruggeman scaled)
        D_eff_n, k_eff_n = D_e * (eps_n ** b_brug), k_e * (eps_n ** b_brug)
        D_eff_s, k_eff_s = D_e * (eps_s ** b_brug), k_e * (eps_s ** b_brug)
        D_eff_p, k_eff_p = D_e * (eps_p ** b_brug), k_e * (eps_p ** b_brug)
        
        # Electrolyte Concentration Polarization Term (From Table I: Charge Conservation)
        # Represents: (2 * (1-t+) * R * T / F) * grad(ln(c_e)). Note: grad(ln(x)) = grad(x)/x
        ce_diff_term = (2.0 * R_const * T / F) * (1.0 - t_plus) * (fx.grad(ce_safe) / ce_safe)
        
        # Electrolyte Li+ Mass Flux [mol/m^2 s]
        flux_ce_n = -D_eff_n * fx.grad(self.c_e)
        flux_ce_s = -D_eff_s * fx.grad(self.c_e)
        flux_ce_p = -D_eff_p * fx.grad(self.c_e)
        
        # Electrolyte Current Flux [A/m^2] (Ohmic + Concentration Migration)
        i_e_n = -k_eff_n * fx.grad(self.phi_e) + k_eff_n * ce_diff_term
        i_e_s = -k_eff_s * fx.grad(self.phi_e) + k_eff_s * ce_diff_term
        i_e_p = -k_eff_p * fx.grad(self.phi_e) + k_eff_p * ce_diff_term

        i_den = self.i_app / A_elec

        # =====================================================================
        # 7. Explicit Equation Targeting (Mapping directly to Table I)
        # =====================================================================
        return {
            "equations": {
                # --- Electrolyte Mass Conservation ---
                self.c_e: fx.Piecewise({
                    self.x_n: eps_n * fx.dt(self.c_e) == -fx.div(flux_ce_n) + (1.0 - t_plus) * J_n / F,
                    self.x_s: eps_s * fx.dt(self.c_e) == -fx.div(flux_ce_s),
                    self.x_p: eps_p * fx.dt(self.c_e) == -fx.div(flux_ce_p) + (1.0 - t_plus) * J_p / F
                }),
                
                # --- Electrolyte Charge Conservation (Spatial DAE) ---
                self.phi_e: fx.Piecewise({
                    self.x_n: fx.div(i_e_n) == J_n,
                    self.x_s: fx.div(i_e_s) == 0.0,
                    self.x_p: fx.div(i_e_p) == J_p
                }),
                
                # --- Solid Charge Conservation (Spatial DAE) ---
                self.phi_s_n: fx.div(i_s_n) == -J_n,
                self.phi_s_p: fx.div(i_s_p) == -J_p,
                
                # --- Solid Mass Conservation (Macro-Micro Unrolled PDE) ---
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n),
                self.c_s_p: fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p),
                
                # --- Global Terminal Voltage Algebraic Mapping ---
                self.V_cell: self.V_cell == self.phi_s_p.right - self.phi_s_n.left
            },
            
            # =================================================================
            # 8. Boundary Conditions
            # =================================================================
            "boundaries": {
                # Reference potential anchoring point (Prevents singular DAEs)
                self.phi_s_n: {"left": fx.Dirichlet(0.0)}, 
                
                # Zero flux through current collectors (Except terminal i_app at cathode)
                i_s_n:        {"right": 0.0},
                i_s_p:        {"left": 0.0, "right": i_den}, 
                
                # Impermeable boundaries for electrolyte mass and charge
                flux_ce_n:    {"left": 0.0},
                flux_ce_p:    {"right": 0.0},
                i_e_n:        {"left": 0.0},
                i_e_p:        {"right": 0.0},
                
                # Volumetric faradaic current injected natively into micro-particle surfaces
                N_s_n:        {"left": 0.0, "right": J_n / (a_n * F)},
                N_s_p:        {"left": 0.0, "right": J_p / (a_p * F)},
            },
            
            # =================================================================
            # 9. Initial Conditions
            # =================================================================
            "initial_conditions": {
                self.c_e: 1000.0,     
                
                # Equilibrium potential offsets calculated to zero out initial eta
                self.phi_s_n: 0.0,  
                self.phi_e: -0.092,
                self.phi_s_p: 4.182,
                
                # Based on the 100% SOC stoichiometries from Table VII
                self.c_s_n: 0.9014 * c_max_n,  
                self.c_s_p: 0.27 * c_max_p,
                
                self.V_cell: 4.182,    
                self.i_app: 0.0
            },
            "observables": {
                self.J_n_obs: J_n,
                self.J_p_obs: J_p
            }
        }

if __name__ == "__main__":
    engine = fx.Engine(model=Chen2020_DFN(), target="cpu:serial", solver_backend="native")
    
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
        
        results[label] = engine.solve(protocol=protocol, parameters={"D_s_n": params["D_s_n"]})

    # =========================================================================
    # Spatial Coordinate Definitions
    # =========================================================================
    x_cell = np.linspace(0, 172.8, 72)
    x_anode = np.linspace(0, 85.2, 35)
    r_cathode = np.linspace(0, 5.22, 15)

    # =========================================================================
    # FIGURE: Combined Macro & Micro Diagnostics
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Chen et al. (2020) - P2D DFN: Macro & Micro Diagnostics", fontsize=15, fontweight="bold")
    
    colors_macro = {"C/2": "tab:blue", "1C": "tab:orange", "1.5C": "tab:red"}
    
    # --- Top Row: Macro Validation (Across All C-Rates) ---
    for label, res in results.items():
        t_hours = res["Time [s]"].data / 3600.0
        mask = res["i_app"].data > 0.1
        c_e_end = res["c_e"].data[mask][-1]
        
        axs[0, 0].plot(t_hours, res["V_cell"].data, label=label, color=colors_macro[label], linewidth=2)
        axs[0, 1].plot(x_cell, c_e_end, label=f"{label} (End of Discharge)", color=colors_macro[label], linewidth=2)

    axs[0, 0].set(title="Voltage Profiles (Rep. Fig 17)", xlabel="Time [h]", ylabel="Terminal Voltage [V]", ylim=[2.4, 4.3])
    axs[0, 1].set(title="Electrolyte Concentration Polarization", xlabel="Distance from Anode Collector $[\mu m]$", ylabel="Concentration $[mol/m^3]$")
    axs[0, 1].axvline(85.2, color='k', linestyle='--', alpha=0.5, label="Separator Interface")
    axs[0, 1].axvline(97.2, color='k', linestyle='--', alpha=0.5)

    # --- Bottom Row: Internal Micro-Diagnostics (1.5C Only) ---
    res_15c = results["1.5C"]
    t_15c = res_15c["Time [s]"].data
    mask_discharge = res_15c["i_app"].data > 0.1
    t_discharge = t_15c[mask_discharge]
    
    idx_10 = np.searchsorted(t_discharge, t_discharge[-1] * 0.10)
    idx_50 = np.searchsorted(t_discharge, t_discharge[-1] * 0.50)
    idx_90 = np.searchsorted(t_discharge, t_discharge[-1] * 0.90)
    
    J_n_history = res_15c["J_n_obs"].data
    c_s_p_history = res_15c["c_s_p"].data
    
    colors_micro = ["tab:blue", "tab:orange", "tab:red"]
    labels_micro = ["10% DOD", "50% DOD", "90% DOD"]
    
    for idx, label_text, color in zip([idx_10, idx_50, idx_90], labels_micro, colors_micro):
        axs[1, 0].plot(x_anode, J_n_history[idx], color=color, linewidth=2.5, label=label_text)
        
        # The cathode has 31 macro nodes. The particle exactly at the separator is index 0.
        # Reshape the flat 1D array into the 2D hierarchical geometry (31 macro x 15 micro)
        c_radial_profile = c_s_p_history[idx].reshape((31, 15))[0, :] 
        axs[1, 1].plot(r_cathode, c_radial_profile, color=color, linewidth=2.5, label=label_text)

    axs[1, 0].set(title="Anode Reaction Current Heterogeneity (1.5C)", xlabel="Distance from Anode Collector $[\mu m]$", ylabel="Volumetric Current, $J_n$ $[A/m^3]$")
    axs[1, 1].set(title="Cathode Core-Shell Saturation (1.5C @ Separator)", xlabel="Particle Radius, $r$ $[\mu m]$ (0=Core)", ylabel="Solid Concentration $[mol/m^3]$")
    axs[1, 1].axhline(63104.0, color='k', linestyle=':', linewidth=2.5, label="Saturation Limit ($c_{max}$)")

    for ax in axs.flat:
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        
    fig.tight_layout()
    plt.show()