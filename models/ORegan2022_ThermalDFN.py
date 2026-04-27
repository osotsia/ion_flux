r"""
Thermal-Electrochemical Doyle-Fuller-Newman (DFN) Model

Reference:
O'Regan, K., Brosa Planella, F., Widanage, W. D., & Kendrick, E. (2022).
"Thermal-electrochemical parameters of a high energy lithium-ion cylindrical battery."
Electrochimica Acta.

This script implements the parameterization of the LG M50 21700 cell.
It leverages Async Task Parallelism to execute multiple state-machine protocols 
concurrently, and extracts internal spatial fields to diagnose rate-limiting transport phenomena.

TODO: change the current uniform mesh to exponential
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class MeshConfig:
    """
    Configuration class for the spatial discretization mesh.
    """
    # Macro-Scale Dimensions (Thicknesses in meters)
    L_n = 85.2e-6
    L_s = 12.0e-6
    L_p = 75.6e-6
    L_cell = L_n + L_s + L_p

    # Micro-Scale Dimensions (Particle Radii in meters)
    R_n = 5.86e-6
    R_p = 5.22e-6

    # Macro-Scale Resolutions (Number of spatial nodes)
    # STRICTLY ALIGNED to physical lengths to prevent mismatched `dx` scales 
    # between regional states (phi_s) and global states (c_e).
    res_n = 71
    res_s = 10
    res_p = 63
    res_cell = res_n + res_s + res_p

    # Micro-Scale Resolutions (Radial nodes inside each particle)
    res_r_n = 15
    res_r_p = 15


class ThermalDFN(fx.PDE):
    # =========================================================================
    # 1. Topological Sub-Meshing (LG M50 Dimensions, Table 8)
    # =========================================================================
    cell = fx.Domain(bounds=(0, MeshConfig.L_cell), resolution=MeshConfig.res_cell)
    x_n = cell.region(bounds=(0, MeshConfig.L_n), resolution=MeshConfig.res_n, name="x_n")
    x_s = cell.region(bounds=(MeshConfig.L_n, MeshConfig.L_n + MeshConfig.L_s), resolution=MeshConfig.res_s, name="x_s")
    x_p = cell.region(bounds=(MeshConfig.L_n + MeshConfig.L_s, MeshConfig.L_cell), resolution=MeshConfig.res_p, name="x_p")
    
    r_n = fx.Domain(bounds=(0, MeshConfig.R_n), resolution=MeshConfig.res_r_n, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, MeshConfig.R_p), resolution=MeshConfig.res_r_p, coord_sys="spherical", name="r_p") 
    
    # =========================================================================
    # 2. States & Observables
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

    # Observables for spatial reaction diagnostics
    J_n_obs = fx.Observable(domain=x_n, name="J_n_obs")
    J_p_obs = fx.Observable(domain=x_p, name="J_p_obs")
    
    def math(self):
        # =====================================================================
        # 3. Parameters (Table 8 & Physical Constants)
        # =====================================================================
        F, R_const, T_ref = 96485.0, 8.314, 298.15
        A_elec = 0.1024  # Active Electrode Area [m^2]
        
        # Microstructural Parameters (Table 8)
        eps_sn, eps_en = 0.75, 0.25      
        eps_es = 0.47                    
        eps_sp, eps_ep = 0.665, 0.335
        b_brug = 1.5                     
        
        a_n = 3.0 * eps_sn / MeshConfig.R_n
        a_p = 3.0 * eps_sp / MeshConfig.R_p
        
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

        j_tot_n = j_n
        j_tot_p = j_p

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
        
        # 1. Evaluate electrolyte current fluxes first
        flux_phie_n = -ke_eff_n * fx.grad(self.phi_e) + ke_eff_n * ce_diff_term
        flux_phie_s = -ke_eff_s * fx.grad(self.phi_e) + ke_eff_s * ce_diff_term
        flux_phie_p = -ke_eff_p * fx.grad(self.phi_e) + ke_eff_p * ce_diff_term

        # 2. Strict conservative Li+ fluxes incorporating migration (t_plus * i_e / F)
        flux_ce_n = -De_eff_n * fx.grad(self.c_e) + (t_plus * flux_phie_n) / F
        flux_ce_s = -De_eff_s * fx.grad(self.c_e) + (t_plus * flux_phie_s) / F
        flux_ce_p = -De_eff_p * fx.grad(self.c_e) + (t_plus * flux_phie_p) / F

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
        Q_vol = Q_total_area / MeshConfig.L_cell
        
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
                    # Using the strictly conservative formulation: source is purely faradaic injection (j/F)
                    self.x_n: eps_en * fx.dt(self.c_e) == -fx.div(flux_ce_n) + j_n / F,
                    self.x_s: eps_es * fx.dt(self.c_e) == -fx.div(flux_ce_s),
                    self.x_p: eps_ep * fx.dt(self.c_e) == -fx.div(flux_ce_p) + j_p / F
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
            },
            "observables": {
                self.J_n_obs: j_n,
                self.J_p_obs: j_p
            }
        }


def run_parallel_processes():
    print("Compiling DFN Math to Native C++ Binary...")
    # NOTE: Set solver_backend to 'sundials' to utilize the robust IDA solver
    compiler_engine = fx.Engine(model=ThermalDFN(), target="cpu:serial", solver_backend="sundials")
    
    rates = {"0.5C": 2.5, "1C": 5.0, "2C": 10.0}
    params_list = []
    protocols_list = []
    keys_order = []
    
    for name, current in rates.items():
        keys_order.append(name)
        params_list.append({}) # Target parameters can be overridden here
        protocols_list.append(Sequence([
            CC(rate=current, until=compiler_engine.model.V_cell <= 2.5, time=15000),
            Rest(time=7200)
        ]))
    
    # Drop into Rust, utilizing Rayon thread-pool to distribute state-machine evaluation.
    print("\nInitiating Native Rayon Batch Execution...")
    start_time = time.perf_counter()
    batch_results = compiler_engine.solve_batch(parameters=params_list, protocols=protocols_list, max_workers=3, show_progress=True)
    print(f"\nAll batch processes completed in {time.perf_counter() - start_time:.2f}s")
    
    return {keys_order[i]: res for i, res in enumerate(batch_results)}


if __name__ == "__main__":
    results = run_parallel_processes()

    res_2c = results["2C"]
    t_2c = res_2c["Time [s]"].data
    mask_discharge = res_2c["i_app"].data > 0.1
    t_discharge = t_2c[mask_discharge]
    
    idx_10 = np.searchsorted(t_discharge, t_discharge[-1] * 0.10)
    idx_50 = np.searchsorted(t_discharge, t_discharge[-1] * 0.50)
    idx_90 = np.searchsorted(t_discharge, t_discharge[-1] * 0.90)
    dod_indices = [idx_10, idx_50, idx_90]
    dod_labels = ["10% DoD", "50% DoD", "90% DoD"]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    # =========================================================================
    # COMBINED FIGURE: Macro Validation & Internal Spatial Diagnostics
    # =========================================================================
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("O'Regan et al. (2022) - Thermal DFN Simulation Diagnostics", fontsize=16, fontweight="bold")

    # --- ROW 0: Macro Validation (Voltage & Temperature) ---
    for name, res in results.items():
        t_sec = res["Time [s]"].data
        c = {"0.5C": "tab:blue", "1C": "tab:orange", "2C": "tab:green"}[name]
        
        axs[0, 0].plot(t_sec, res["V_cell"].data, label=name, color=c, linewidth=2)
        axs[0, 1].plot(t_sec, res["T_cell"].data - 273.15, label=name, color=c, linewidth=2)

    axs[0, 0].set(title="Terminal Voltage Profiles", xlabel="Time [s]", ylabel="Voltage [V]")
    axs[0, 1].set(title="Lumped Temperature Profiles", xlabel="Time [s]", ylabel="Temperature [°C]")

    # --- ROW 1 & 2 Setup: Spatial Diagnostics (2C Discharge) ---
    x_cell = np.linspace(0, MeshConfig.L_cell * 1e6, MeshConfig.res_cell)
    x_anode = np.linspace(0, MeshConfig.L_n * 1e6, MeshConfig.res_n)
    x_cathode = np.linspace((MeshConfig.L_n + MeshConfig.L_s) * 1e6, MeshConfig.L_cell * 1e6, MeshConfig.res_p)
    r_p_arr = np.linspace(0, MeshConfig.R_p * 1e6, MeshConfig.res_r_p)

    sep_start = MeshConfig.L_n * 1e6
    sep_end = (MeshConfig.L_n + MeshConfig.L_s) * 1e6

    c_e_history = res_2c["c_e"].data
    c_s_p_history = res_2c["c_s_p"].data
    j_n_history = res_2c["J_n_obs"].data
    j_p_history = res_2c["J_p_obs"].data

    for idx, label, color in zip(dod_indices, dod_labels, colors):
        # [1, 0] Electrolyte Polarization
        axs[1, 0].plot(x_cell, c_e_history[idx], color=color, linewidth=2, label=label)
        
        # [1, 1] Cathode Particle Saturation
        c_radial = c_s_p_history[idx].reshape((MeshConfig.res_p, MeshConfig.res_r_p))[0, :]
        axs[1, 1].plot(r_p_arr, c_radial, color=color, linewidth=2, label=label)

        # [2, 0] Anode Volumetric Current
        axs[2, 0].plot(x_anode, j_n_history[idx] / 1e6, color=color, linewidth=2, label=label)
        
        # [2, 1] Cathode Volumetric Current (Absolute)
        axs[2, 1].plot(x_cathode, abs(j_p_history[idx]) / 1e6, color=color, linewidth=2, label=label)

    # Styling Row 1
    axs[1, 0].set(title="Electrolyte Starvation", xlabel="Distance from Anode [µm]", ylabel="Concentration [mol/m³]")
    axs[1, 0].axvline(sep_start, color='k', linestyle='--', alpha=0.5, label="Separator")
    axs[1, 0].axvline(sep_end, color='k', linestyle='--', alpha=0.5)
    axs[1, 1].set(title="Cathode Particle Saturation (Separator Interface)", xlabel="Radial Distance [µm]", ylabel="Concentration [mol/m³]")
    axs[1, 1].axhline(51765.0, color='k', linestyle=':', label="Saturation Limit ($c_{max}$)")
    
    # Styling Row 2
    axs[2, 0].set(title="Anode Volumetric Current", xlabel="Distance [µm]", ylabel="Reaction Current [A/cm³]")
    axs[2, 1].set(title="Cathode Volumetric Current (Absolute)", xlabel="Distance [µm]", ylabel="Reaction Current [A/cm³]")

    # Apply generic styling to all 6 subplots
    for ax in axs.flat:
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    plt.show()