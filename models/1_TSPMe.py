"""
Thermal Single Particle Model with Electrolyte (TSPMe)

Reference:
Brosa Planella, F., Sheikh, M., & Widanage, W. D. (2021).
"Systematic derivation and validation of a reduced thermal-electrochemical 
model for lithium-ion batteries using asymptotic methods."
Electrochimica Acta, 388, 138524. 
https://doi.org/10.1016/j.electacta.2021.138524
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class ExactTSPMe(fx.PDE):
    """
    Strictly reflects the asymptotic derivation of Brosa Planella et al. (2021).
    Utilizes exact analytical integrations for Ohmic losses (Section 3.1.2) to 
    maximize implicit solver stability and performance.
    """
    # -------------------------------------------------------------------------
    # 1. Topology (Fig. 4 Geometry)
    # -------------------------------------------------------------------------
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=144)
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=71, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=10, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=63, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 5.86e-6), resolution=15, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5.22e-6), resolution=15, coord_sys="spherical", name="r_p") 
    
    # -------------------------------------------------------------------------
    # 2. States & Observables
    # -------------------------------------------------------------------------
    c_e = fx.State(domain=cell, name="c_e")
    c_s_n = fx.State(domain=r_n, name="c_s_n")
    c_s_p = fx.State(domain=r_p, name="c_s_p")
    
    T_cell = fx.State(domain=None, name="T_cell") # 0D Lumped Thermal ODE
    V_cell = fx.State(domain=None, name="V_cell") # 0D Algebraic Voltage Constraint
    i_app = fx.State(domain=None, name="i_app")   # 0D Cycler terminal 
    
    # --- Diagnostic Telemetry (Zero overhead on the implicit solver) ---
    U_eq = fx.Observable(domain=None, name="U_eq")
    eta_r = fx.Observable(domain=None, name="eta_r")
    eta_e = fx.Observable(domain=None, name="eta_e")
    dPhi_s = fx.Observable(domain=None, name="dPhi_s")
    dPhi_e = fx.Observable(domain=None, name="dPhi_e")
    
    Q_s = fx.Observable(domain=None, name="Q_s")
    Q_e = fx.Observable(domain=None, name="Q_e")
    Q_irr = fx.Observable(domain=None, name="Q_irr")
    
    T_amb = fx.Parameter(default=298.15, name="T_amb")
    Ds_n = fx.Parameter(default=3.3e-14, name="Ds_n")  # <-- ADDED
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        # =====================================================================
        # Parameters (Table 1: LG M50 & Table 4: Tuned Thermal Parameters)
        # =====================================================================
        F, R_const, T_ref = 96485.0, 8.314, 298.15
        
        # Macro dimensions [m] & Electrode Area [m^2]
        L_n, L_s, L_p = 85.2e-6, 12.0e-6, 75.6e-6
        L_cell = L_n + L_s + L_p
        A_elec = 0.10269  # Corrected to exactly match 5Ah / 48.69 A/m^2 (Table 1)
        
        # Microstructure (Porosity & Bruggeman factor, Eq. 4)
        eps_n, eps_s, eps_p = 0.25, 0.47, 0.335
        b_brug = 1.5
        
        # Electrode Particle Properties
        a_n, a_p = 3.84e5, 3.82e5           # Surface area density [1/m]
        c_max_n, c_max_p = 33133.0, 63104.0 # Max lithium concentration [mol/m^3]
        sig_n, sig_p = 215.0, 0.18          # Electronic conductivity [S/m]
        m_n, m_p = 6.48e-7, 3.42e-6         # Reaction rate [(A/m^2)(mol/m^3)^-1.5]
        
        # Electrolyte Properties
        De_ref, sig_e_ref, t_plus = 3e-10, 1.0, 0.2594
        
        # Thermal Properties (Table 1 of Brosa Planella 2021)
        theta_heat = 2.85e6  # Baseline volumetric heat capacity [J / (K m^3)]
        h_cool = 20.0        # Baseline heat exchange coefficient [W / (K m^2)]
        a_cool = 219.42      # Cooling surface area density [1/m]

        # Solid Diffusion (Paper uses 'effective' constants per C-rate/Temp rather than Arrhenius)
        # Note: To replicate the paper's plots exactly at a specific temperature,
        # you would manually tune these constants per run. Here we use the base reference values.
        # Ds_n = 3.3e-14 
        Ds_p = 4.0e-15

        # Helper AST macros
        def arcsinh_ast(x): return fx.log(x + (x**2 + 1.0)**0.5)
        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)

        # =====================================================================
        # Core Electrochemistry & Thermodynamics
        # =====================================================================
        i_den = self.i_app / A_elec
        
        # Volumetric current densities [A/m^3] (Eq. 2)
        j_vol_n = i_den / L_n
        j_vol_p = -i_den / L_p
        
        # Surface concentrations clamped for numerical safety
        c_surf_n = fx.min(fx.max(self.c_s_n.boundary("right", domain=self.r_n), 10.0), c_max_n - 10.0)
        c_surf_p = fx.min(fx.max(self.c_s_p.boundary("right", domain=self.r_p), 10.0), c_max_p - 10.0)
        ce_safe = fx.max(self.c_e, 1.0)
        
        x_n = c_surf_n / c_max_n
        x_p = c_surf_p / c_max_p

        # Open Circuit Potentials (Appendix D, Fig D.1)
        U_n = (1.9793 * fx.exp(-39.3631 * x_n) + 0.2482 
               - 0.0909 * tanh_ast(29.8538 * (x_n - 0.1234)) 
               - 0.04478 * tanh_ast(14.9159 * (x_n - 0.2769)) 
               - 0.0205 * tanh_ast(30.4444 * (x_n - 0.6103)))
               
        U_p = (-0.8090 * x_p + 4.4875 
               - 0.0428 * tanh_ast(18.5138 * (x_p - 0.5542)) 
               - 17.7326 * tanh_ast(15.7890 * (x_p - 0.3117)) 
               + 17.5842 * tanh_ast(15.9308 * (x_p - 0.3120)))
        
        # Exchange current densities (Eq. 7)
        def arrh(Ea):
            return fx.exp((Ea / R_const) * (1.0 / T_ref - 1.0 / self.T_cell))
        
        j0_n = m_n * (ce_safe * c_surf_n * (c_max_n - c_surf_n))**0.5 * arrh(40000.0)
        j0_p = m_p * (ce_safe * c_surf_p * (c_max_p - c_surf_p))**0.5 * arrh(24000.0)

        # =====================================================================
        # Voltage Resolution (Eq. 8a)
        # =====================================================================
        
        # 1. Equilibrium Potential (Eq. 8b)
        U_eq = U_p - U_n
        
        # 2. Reaction Overpotential (Eq. 8c)
        term_n = i_den / (a_n * L_n * j0_n)
        term_p = i_den / (a_p * L_p * j0_p)
        eta_r_n = - (2.0 * R_const * self.T_cell / F) * (fx.integral(arcsinh_ast(term_n), over=self.x_n) / L_n)
        eta_r_p = - (2.0 * R_const * self.T_cell / F) * (fx.integral(arcsinh_ast(term_p), over=self.x_p) / L_p)
        eta_r = eta_r_n + eta_r_p
        
        # 3. Electrolyte Concentration Overpotential (Eq. 8d)
        eta_e = (1.0 - t_plus) * (2.0 * R_const * self.T_cell / F) * (
            fx.integral(fx.log(ce_safe), over=self.x_p) / L_p - 
            fx.integral(fx.log(ce_safe), over=self.x_n) / L_n
        )
        
        # 4. Solid Ohmic Drop (Eq. 8f)
        R_s_ohm = (L_n / sig_n + L_p / sig_p) / 3.0
        dPhi_s = -i_den * R_s_ohm
        
        # 5. Electrolyte Ohmic Drop (Eq. 16: Analytical limit from Sec 3.1.2)
        term_n_e = L_n / (eps_n ** b_brug)
        term_s_e = 3.0 * L_s / (eps_s ** b_brug)
        term_p_e = L_p / (eps_p ** b_brug)
        R_e_ohm = (term_n_e + term_s_e + term_p_e) / (3.0 * sig_e_ref)
        dPhi_e = -i_den * R_e_ohm
                  
        V_total = U_eq + eta_r + eta_e + dPhi_e + dPhi_s

        # =====================================================================
        # Heat Generation (Eq. 5a)
        # =====================================================================
        
        Q_s = (i_den ** 2) * R_s_ohm / L_cell                      # Solid Joule Heating (Eq. 5c)
        Q_e = (i_den ** 2) * R_e_ohm / L_cell - (i_den * eta_e / L_cell) # Electrolyte Heating (Eq. 17)
        # FIXED: Absolute value wraps the entire product to correctly act as a heat source during both charge and discharge
        Q_irr = fx.abs(i_den * eta_r) / L_cell                     # Irreversible Reaction Heating (Eq. 5e)
        
        Q_cool = h_cool * a_cool * (self.T_cell - self.T_amb)      # Newton Cooling to Ambient (Eq. 5a)
        Q_tot = Q_s + Q_e + Q_irr                                  # Total Heat Source

        # =====================================================================
        # PDEs & Spatial Tensors
        # =====================================================================
        
        N_s_n = -self.Ds_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p)
        
        De_eff_n = De_ref * (eps_n ** b_brug)
        De_eff_s = De_ref * (eps_s ** b_brug)
        De_eff_p = De_ref * (eps_p ** b_brug)
        
        flux_ce_n = -De_eff_n * fx.grad(self.c_e)
        flux_ce_s = -De_eff_s * fx.grad(self.c_e)
        flux_ce_p = -De_eff_p * fx.grad(self.c_e)

        return {
            "equations": {
                # --- Electrolyte Transport (Eq. 3a) ---
                self.c_e: fx.Piecewise({
                    self.x_n: eps_n * fx.dt(self.c_e) == -fx.div(flux_ce_n) + (1.0 - t_plus) * j_vol_n / F,
                    self.x_s: eps_s * fx.dt(self.c_e) == -fx.div(flux_ce_s),
                    self.x_p: eps_p * fx.dt(self.c_e) == -fx.div(flux_ce_p) + (1.0 - t_plus) * j_vol_p / F
                }),
                # --- Solid Particle Transport (Eq. 1a) ---
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n),
                self.c_s_p: fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p),
                
                # --- Lumped Thermal & Voltage Constraints (Eq. 5a & 8a) ---
                self.T_cell: fx.dt(self.T_cell) == (Q_tot - Q_cool) / theta_heat,
                self.V_cell: self.V_cell == V_total
            },
            
            "boundaries": {
                # Electrolyte boundaries (Eq. 3b)
                flux_ce_n: {"left": 0.0},
                flux_ce_p: {"right": 0.0},
                # Solid surface flux boundaries (Eq. 1c)
                N_s_n: {"left": 0.0, "right": j_vol_n / (a_n * F)},
                N_s_p: {"left": 0.0, "right": j_vol_p / (a_p * F)},
            },
            
            "initial_conditions": {
                self.c_e: 1000.0,     
                self.c_s_n: 29866.0,  
                self.c_s_p: 17038.0,
                self.T_cell: self.T_amb,   
                self.V_cell: 4.10, 
                self.i_app: 0.0
            },
            
            # Telemetry tracking
            "observables": {
                self.U_eq: U_eq,
                self.eta_r: eta_r,
                self.eta_e: eta_e,
                self.dPhi_s: dPhi_s,
                self.dPhi_e: dPhi_e,
                self.Q_s: Q_s,
                self.Q_e: Q_e,
                self.Q_irr: Q_irr
            }
        }

if __name__ == "__main__":
    
    # Bandwidth=0 signals FAER LU/GMRES to handle internal cross-domain sparsity natively
    engine = fx.Engine(model=ExactTSPMe(), target="cpu:serial", jacobian_bandwidth=0)
    
    rates = {"0.5C": 2.5, "1C": 5.0, "2C": 10.0}
    tuned_Dn = {"0.5C": 0.9e-14, "1C": 2.0e-14, "2C": 6.0e-14}  # <-- ADDED (from Table 4)
    results = {}
    
    for name, current in rates.items():
        print(f"Simulating {name} discharge + 2-hour rest...")
        # Discharge to 2.5V, followed by 7200s (2-hour) relaxation
        protocol = Sequence([
            CC(rate=current, until=engine.model.V_cell <= 2.5, time=15000),
            Rest(time=7200)
        ])
        
        # <-- UPDATED: Inject the tuned parameter
        results[name] = engine.solve(protocol=protocol, parameters={"Ds_n": tuned_Dn[name]})


    # =========================================================================
    # FIGURE 1: Replication of Paper Figure 5
    # (Voltage and Temperature vs Capacity during Discharge Phase)
    # =========================================================================
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {"0.5C": "tab:blue", "1C": "tab:orange", "2C": "tab:green"}
    
    for name, res in results.items():
        # Strictly mask the active CC discharge phase (ignoring the Rest step)
        discharge_mask = res["i_app"].data > 0.1
        t_discharge = res["Time [s]"].data[discharge_mask]
        
        # Calculate capacity mapped strictly to the discharge segment
        capacity_ah = (t_discharge * rates[name]) / 3600.0
        v_cell = res["V_cell"].data[discharge_mask]
        t_celsius = res["T_cell"].data[discharge_mask] - 273.15
        
        axs1[0].plot(capacity_ah, v_cell, label=f"{name}", color=colors[name], linewidth=2)
        axs1[1].plot(capacity_ah, t_celsius, label=f"{name}", color=colors[name], linewidth=2)

    axs1[0].set_title("TSPMe Validation: LG M50 at 25°C (Replicating Fig. 5)", fontsize=14, fontweight="bold")
    axs1[0].set_ylabel("Terminal Voltage [V]", fontsize=12)
    axs1[0].grid(True, linestyle="--", alpha=0.7)
    axs1[0].legend(fontsize=11)
    
    axs1[1].set_ylabel("Cell Temperature [°C]", fontsize=12)
    axs1[1].set_xlabel("Discharge Capacity [Ah]", fontsize=12)
    axs1[1].grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()


    # =========================================================================
    # FIGURE 2: Replication of Paper Figure 8 Layout
    # (Voltage and Cell Temperature vs Time over Full Discharge + Rest)
    # =========================================================================
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 10))
    
    for i, (name, current) in enumerate(rates.items()):
        res = results[name]
        t_seconds = res["Time [s]"].data
        v_cell = res["V_cell"].data
        t_celsius = res["T_cell"].data - 273.15
        
        # --- Left Column: Voltage ---
        axs2[i, 0].plot(t_seconds, v_cell, label="TSPMe", color="black", linewidth=2)
        axs2[i, 0].set_ylabel("Voltage (V)", fontsize=12)
        axs2[i, 0].grid(True, linestyle="--", alpha=0.7)
        axs2[i, 0].text(-0.15, 0.5, name, transform=axs2[i, 0].transAxes, 
                       fontsize=14, fontweight="bold", va="center")
        
        # --- Right Column: Temperature ---
        axs2[i, 1].plot(t_seconds, t_celsius, label="TSPMe", color="black", linewidth=2)
        axs2[i, 1].set_ylabel("Cell temperature (°C)", fontsize=12)
        axs2[i, 1].grid(True, linestyle="--", alpha=0.7)
        
        if i == 2:
            axs2[i, 0].set_xlabel("Time (s)", fontsize=12)
            axs2[i, 1].set_xlabel("Time (s)", fontsize=12)

    fig2.suptitle("TSPMe Validation: LG M50 at 25°C (Replicating Fig. 8 Layout)", fontsize=15, fontweight="bold")
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.92, left=0.12)


    # =========================================================================
    # FIGURE 3: Anatomy of the Voltage Cliff (Deconstruction)
    # =========================================================================
    print("Generating Voltage Deconstruction analysis...")
    fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, rate in enumerate(["0.5C", "2C"]):
        res = results[rate]
        
        # Mask strictly to the discharge phase
        mask = res["i_app"].data > 0.1
        cap = (res["Time [s]"].data[mask] * rates[rate]) / 3600.0
        
        # Extract telemetry
        u_eq = res["U_eq"].data[mask]
        eta_r = res["eta_r"].data[mask]
        eta_e = res["eta_e"].data[mask]
        dphi_e = res["dPhi_e"].data[mask]
        dphi_s = res["dPhi_s"].data[mask]
        v_cell = res["V_cell"].data[mask]
        
        ax = axs3[i]
        
        # Calculate sequential drop layers
        l1 = u_eq
        l2 = l1 + eta_r
        l3 = l2 + eta_e
        l4 = l3 + dphi_e
        l5 = l4 + dphi_s # Matches v_cell
        
        ax.plot(cap, l1, color='k', linestyle='--', linewidth=2, label="Thermodynamic $U_{eq}$")
        ax.plot(cap, v_cell, color='k', linewidth=2, label="Terminal $V_{cell}$")
        
        ax.fill_between(cap, l1, l2, color="tab:red", alpha=0.5, label="Reaction Kinetics ($\eta_r$)")
        ax.fill_between(cap, l2, l3, color="tab:orange", alpha=0.5, label="Electrolyte Transport ($\eta_e$)")
        ax.fill_between(cap, l3, l4, color="tab:blue", alpha=0.5, label="Electrolyte Ohmic Drop ($\Delta\Phi_e$)")
        ax.fill_between(cap, l4, l5, color="tab:gray", alpha=0.5, label="Solid Ohmic Drop ($\Delta\Phi_s$)")
        
        ax.set_title(f"Voltage Penalty Breakdown at {rate}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Discharge Capacity [Ah]", fontsize=12)
        if i == 0:
            ax.set_ylabel("Voltage [V]", fontsize=12)
            ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

    fig3.tight_layout()


    # =========================================================================
    # FIGURE 4: The Thermal Engine (Heat Apportionment)
    # =========================================================================
    print("Generating Heat Apportionment analysis...")
    fig4, axs4 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, rate in enumerate(["0.5C", "2C"]):
        res = results[rate]
        
        mask = res["i_app"].data > 0.1
        cap = (res["Time [s]"].data[mask] * rates[rate]) / 3600.0
        
        q_s = res["Q_s"].data[mask]
        q_e = res["Q_e"].data[mask]
        q_irr = res["Q_irr"].data[mask]
        
        ax = axs4[i]
        
        ax.stackplot(cap, q_s, q_e, q_irr, 
                     labels=["Solid Ohmic ($Q_s$)", "Electrolyte Ohmic ($Q_e$)", "Irreversible Rxn ($Q_{irr}$)"],
                     colors=["tab:gray", "tab:blue", "tab:red"], alpha=0.7)
        
        ax.set_title(f"Heat Source Apportionment at {rate}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Discharge Capacity [Ah]", fontsize=12)
        if i == 0:
            ax.set_ylabel("Volumetric Heat Gen [$W/m^3$]", fontsize=12)
            ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

    fig4.tight_layout()
    
    plt.show()