r"""
1+1D Single Particle Model with Electrolyte (SPMe)

Reference:
Marquis, S. G., Timms, R., Sulzer, V., Please, C. P., & Chapman, S. J. (2020).
"A Suite of Reduced-Order Models of a Single-Layer Lithium-ion Pouch Cell."
arXiv preprint arXiv:2008.03691.

This model resolves the macroscopic transverse geometry (the z-axis of the current 
collectors) while employing an asymptotically reduced through-cell model (the SPMe).
It captures electrolyte concentration gradients, non-uniform current distributions (I_loc),
and fully coupled Arrhenius thermal-electrochemical feedback loops (Table 6 & 7).
"""

import numpy as np
import matplotlib.pyplot as plt
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class Marquis1Plus1D_SPMe(fx.PDE):
    """
    Pseudo-3D architecture capturing transverse (z), through-cell (x), 
    and micro-radial (r) dynamics using independent 2-level composite domains.
    """
    # =========================================================================
    # 1. Topology & Domains (Table 6)
    # =========================================================================
    # Transverse Domain (Tall Pouch Cell Height: 137 mm)
    z = fx.Domain(bounds=(0, 0.137), resolution=15, name="z")
    
    # Through-Cell Domain (Total Thickness: 225 um)
    x_cell = fx.Domain(bounds=(0, 225e-6), resolution=25, name="x_cell")
    x_n = x_cell.region(bounds=(0, 100e-6), resolution=10, name="x_n")
    x_s = x_cell.region(bounds=(100e-6, 125e-6), resolution=5, name="x_s")
    x_p = x_cell.region(bounds=(125e-6, 225e-6), resolution=10, name="x_p")
    
    # Micro-Particle Domains
    r_n = fx.Domain(bounds=(0, 10e-6), resolution=10, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 10e-6), resolution=10, coord_sys="spherical", name="r_p") 
    
    # Composite Hierarchies
    z_x = z * x_cell
    z_rn = z * r_n
    z_rp = z * r_p

    # =========================================================================
    # 2. States & Observables
    # =========================================================================
    # 2D Fields
    c_e = fx.State(domain=z_x, name="c_e")       # Electrolyte Conc. [mol/m^3]
    c_sn = fx.State(domain=z_rn, name="c_sn")    # Anode Solid Conc. [mol/m^3]
    c_sp = fx.State(domain=z_rp, name="c_sp")    # Cathode Solid Conc. [mol/m^3]
    
    # 1D Transverse Fields
    phi_cn = fx.State(domain=z, name="phi_cn")   # Anode Current Collector [V]
    phi_cp = fx.State(domain=z, name="phi_cp")   # Cathode Current Collector [V]
    T_cell = fx.State(domain=z, name="T_cell")   # Local Temperature [K]
    
    # The crucial coupling state: Through-cell current density [A/m^2]
    I_loc = fx.State(domain=z, name="I_loc")     
    
    # 0D Global States
    V_term = fx.State(domain=None, name="V_term")
    I_app = fx.State(domain=None, name="I_app")
    
    terminal = fx.Terminal(current=I_app, voltage=V_term)

    def math(self):
        # =====================================================================
        # 3. Parameters (Table 6: LGM50 Parameterization)
        # =====================================================================
        F, R_g, T_inf = 96487.0, 8.314, 298.15
        
        L_n, L_s, L_p = 100e-6, 25e-6, 100e-6
        L_cell = L_n + L_s + L_p
        L_cn, L_cp = 25e-6, 25e-6
        L_y = 0.207 
        
        eps_n, eps_s, eps_p = 0.3, 1.0, 0.3
        b_brug = 1.5
        a_n, a_p = 0.18e6, 0.15e6
        c_n_max, c_p_max = 24983.2619938437, 51217.9257309275
        
        sig_n, sig_p = 100.0, 10.0
        sig_cn, sig_cp = 5.96e7, 3.55e7
        t_plus = 0.4
        
        rho_eff, lambda_eff = 1.812e6, 59.396

        L_tab = 40e-3
        L_total = L_cell + L_cn + L_cp

        # =====================================================================
        # 4. Helper AST Functions & Arrhenius Kinetics
        # =====================================================================
        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)
            
        def cosh_ast(x): return 0.5 * (fx.exp(x) + fx.exp(-x))
        def sech2_ast(x): return 1.0 / (cosh_ast(x)**2)
        def arcsinh_ast(x): return fx.log(x + (x**2 + 1.0)**0.5)
        
        # Arrhenius multiplier for thermal feedback (Table 6 & 7)
        def arrh(E_a):
            # Bound T_cell to prevent exponential NaN explosions during Newton stepping
            T_safe = fx.min(fx.max(self.T_cell, 200.0), 400.0)
            return fx.exp((E_a / R_g) * (1.0 / T_inf - 1.0 / T_safe))

        # =====================================================================
        # 5. Thermodynamics & Non-Linear Transport (Table 7)
        # =====================================================================
        c_sn_surf = fx.min(fx.max(self.c_sn.boundary("right", domain=self.r_n), 10.0), c_n_max - 10.0)
        c_sp_surf = fx.min(fx.max(self.c_sp.boundary("right", domain=self.r_p), 10.0), c_p_max - 10.0)
        
        # Boundaries for electrolyte concentration overpotential (Eq 3.25)
        ce_x0 = fx.max(self.c_e.boundary("left", domain=self.x_cell), 10.0)
        ce_xL = fx.max(self.c_e.boundary("right", domain=self.x_cell), 10.0)
        
        # Averages for exchange current density
        ce_avg_n = fx.max(fx.integral(self.c_e, over=self.x_n) / L_n, 10.0)
        ce_avg_p = fx.max(fx.integral(self.c_e, over=self.x_p) / L_p, 10.0)
        ce_avg_cell = fx.max(fx.integral(self.c_e, over=self.x_cell) / L_cell, 10.0)

        x_n = c_sn_surf / c_n_max
        x_p = c_sp_surf / c_p_max
        x_p_str = 1.062 * x_p

        # Open Circuit Potentials (Exact from Table 7)
        U_n = 0.194 + 1.5 * fx.exp(-120.0 * x_n) \
              + 0.0351 * tanh_ast((x_n - 0.286) / 0.083) \
              - 0.0045 * tanh_ast((x_n - 0.849) / 0.119) \
              - 0.035 * tanh_ast((x_n - 0.9233) / 0.05) \
              - 0.0147 * tanh_ast((x_n - 0.5) / 0.034) \
              - 0.102 * tanh_ast((x_n - 0.194) / 0.142) \
              - 0.022 * tanh_ast((x_n - 0.9) / 0.0164) \
              - 0.011 * tanh_ast((x_n - 0.124) / 0.0226) \
              + 0.0155 * tanh_ast((x_n - 0.105) / 0.029)
              
        U_p = 2.16216 + 0.07645 * tanh_ast(30.834 - 54.4806 * x_p_str) \
              + 2.1581 * tanh_ast(52.294 - 50.294 * x_p_str) \
              - 0.14169 * tanh_ast(11.0923 - 19.8543 * x_p_str) \
              + 0.2051 * tanh_ast(1.4684 - 5.4888 * x_p_str) \
              + 0.2531 * tanh_ast((-x_p_str + 0.56478) / 0.1316) \
              - 0.02167 * tanh_ast((x_p_str - 0.525) / 0.006)

        # Exchange Current Densities with Arrhenius Thermal Feedback (Eq 3.23 / 3.24)
        j0_n = 2e-5 * arrh(3.748e4) * (ce_avg_n * c_sn_surf * (c_n_max - c_sn_surf)) ** 0.5
        j0_p = 6e-7 * arrh(3.957e4) * (ce_avg_p * c_sp_surf * (c_p_max - c_sp_surf)) ** 0.5

        # Eq 3.21 & 3.22: Reaction Overpotentials
        eta_n = (2.0 * R_g * self.T_cell / F) * arcsinh_ast(self.I_loc / (a_n * j0_n * L_n))
        eta_p = -(2.0 * R_g * self.T_cell / F) * arcsinh_ast(self.I_loc / (a_p * j0_p * L_p))
        eta_r = eta_p - eta_n
        
        # Eq 3.25: Non-Linear Electrolyte Concentration Overpotential
        eta_c = 2.0 * (1.0 - t_plus) * (R_g * self.T_cell / F) * fx.log(ce_avg_p / ce_avg_n)
        
        # Exact Non-Linear Electrolyte Conductivity (Table 7)
        ce_m = ce_avg_cell * 1e-3
        kappa_e = (0.0911 + 1.9101 * ce_m - 1.052 * (ce_m**2) + 0.1554 * (ce_m**3)) * arrh(3.47e4)
        
        # Eq 3.26 & 3.27: Ohmic Drops
        R_elec = (1.0 / kappa_e) * (L_n / (3 * eps_n**b_brug) + L_s / (eps_s**b_brug) + L_p / (3 * eps_p**b_brug))
        R_solid = (1.0 / 3.0) * (L_p / sig_p + L_n / sig_n)
        
        dPhi_elec = -self.I_loc * R_elec
        dPhi_solid = -self.I_loc * R_solid
        
        # Eq 3.18: Total Electrochemical Voltage Map
        V_ec = (U_p - U_n) + eta_r + eta_c + dPhi_elec + dPhi_solid

        # =====================================================================
        # 6. Transport Physics
        # =====================================================================
        # 1. Solid Fickian Diffusion
        D_sn = 3.9e-14 * arrh(4.277e4)
        D_sp = 1e-13 * arrh(1.855e4)
        flux_sn = -D_sn * fx.grad(self.c_sn, axis=self.r_n)
        flux_sp = -D_sp * fx.grad(self.c_sp, axis=self.r_p)

        # 2. Exact Non-Linear Electrolyte Mass Transport (Table 7 & Appendix B.1)
        # Binds the local continuous c_e state dynamically instead of taking averages
        ce_local_m = fx.max(self.c_e, 10.0) * 1e-3
        D_e = (5.34e-10) * fx.exp(-0.65 * ce_local_m) * arrh(3.704e4)
        
        # Eq B.1 Mass Conservation Source Terms integrated into FVM interfaces
        flux_en = -D_e * (eps_n**b_brug) * fx.grad(self.c_e, axis=self.x_cell) + (self.x_cell.coords * t_plus * self.I_loc) / (F * L_n)
        flux_es = -D_e * (eps_s**b_brug) * fx.grad(self.c_e, axis=self.x_cell) + (t_plus * self.I_loc) / F
        flux_ep = -D_e * (eps_p**b_brug) * fx.grad(self.c_e, axis=self.x_cell) + ((L_cell - self.x_cell.coords) * t_plus * self.I_loc) / (F * L_p)

        # 3. Transverse Current Collectors (Eq 2.1a)
        i_cn = -sig_cn * fx.grad(self.phi_cn, axis=self.z)
        i_cp = -sig_cp * fx.grad(self.phi_cp, axis=self.z)
        
        # 4. Transverse Thermal Energy Balance (Eq 2.1d, 3.3, 3.8)
        flux_T = -lambda_eff * fx.grad(self.T_cell, axis=self.z)
        
        # Heat Sources: Ohmic
        Q_ohm_cc = (L_cn / L_total) * sig_cn * (fx.grad(self.phi_cn, axis=self.z)**2) + \
                   (L_cp / L_total) * sig_cp * (fx.grad(self.phi_cp, axis=self.z)**2)
        Q_ohm_through = fx.abs((self.I_loc ** 2) * (R_elec + R_solid) / L_total)
        
        # Heat Sources: Irreversible Reaction
        Q_rxn = fx.abs(self.I_loc * eta_r) / L_total
        
        # Heat Sources: Reversible Entropic Heating (Table 7)
        # Note: The paper accidentally published dU/dc as dU/dT in Table 7. 
        # Leaving the dU_dT formulas as-is preserves mathematical parity with the text.

        dU_dT_n = -1.5 * (120.0 / c_n_max) * fx.exp(-120.0 * x_n) \
                  + (0.0351 / (0.083 * c_n_max)) * sech2_ast((x_n - 0.286) / 0.083) \
                  - (0.0045 / (0.119 * c_n_max)) * sech2_ast((x_n - 0.849) / 0.119) \
                  - (0.035 / (0.05 * c_n_max)) * sech2_ast((x_n - 0.9233) / 0.05) \
                  - (0.0147 / (0.034 * c_n_max)) * sech2_ast((x_n - 0.5) / 0.034) \
                  - (0.102 / (0.142 * c_n_max)) * sech2_ast((x_n - 0.194) / 0.142) \
                  - (0.022 / (0.0164 * c_n_max)) * sech2_ast((x_n - 0.9) / 0.0164) \
                  - (0.011 / (0.0226 * c_n_max)) * sech2_ast((x_n - 0.124) / 0.0226) \
                  + (0.0155 / (0.029 * c_n_max)) * sech2_ast((x_n - 0.105) / 0.029)
                  
        dU_dT_p = 0.07645 * (-54.4806 / c_p_max) * sech2_ast(30.834 - 54.4806 * x_p_str) \
                  + 2.1581 * (-50.294 / c_p_max) * sech2_ast(52.294 - 50.294 * x_p_str) \
                  + 0.14169 * (19.854 / c_p_max) * sech2_ast(11.0923 - 19.8543 * x_p_str) \
                  - 0.2051 * (5.4888 / c_p_max) * sech2_ast(1.4684 - 5.4888 * x_p_str) \
                  - (0.2531 / 0.1316 / c_p_max) * sech2_ast((-x_p_str + 0.56478) / 0.1316) \
                  - (0.02167 / 0.006 / c_p_max) * sech2_ast((x_p_str - 0.525) / 0.006)
                  
        Q_rev = (self.I_loc * self.T_cell / L_total) * (dU_dT_n - dU_dT_p)
        
        # The paper specifies strictly adiabatic conditions on all non-tab boundaries.
        Q_cool_face = 0.0
        
        Q_total = Q_ohm_through + Q_rxn + Q_rev + Q_ohm_cc - Q_cool_face
        
        # Calculate the geometrically scaled effective tab cooling coefficient for the 1D projection
        h_tab_eff = 1000.0 * (L_tab * (L_cn + L_cp)) / (L_y * L_total)

        return {
            "equations": {
                # --- Through-Cell Physics ---
                self.c_sn: fx.dt(self.c_sn) == -fx.div(flux_sn, axis=self.r_n),
                self.c_sp: fx.dt(self.c_sp) == -fx.div(flux_sp, axis=self.r_p),
                
                self.c_e: fx.Piecewise({
                    self.x_n: eps_n * fx.dt(self.c_e) == -fx.div(flux_en, axis=self.x_cell) + self.I_loc / (F * L_n),
                    self.x_s: eps_s * fx.dt(self.c_e) == -fx.div(flux_es, axis=self.x_cell),
                    self.x_p: eps_p * fx.dt(self.c_e) == -fx.div(flux_ep, axis=self.x_cell) - self.I_loc / (F * L_p)
                }),
                
                # --- Transverse Physics ---
                self.phi_cn: fx.div(i_cn, axis=self.z) == -self.I_loc / L_cn,
                self.phi_cp: fx.div(i_cp, axis=self.z) == self.I_loc / L_cp,
                self.T_cell: rho_eff * fx.dt(self.T_cell) == (-fx.div(flux_T, axis=self.z) + Q_total),
                
                # --- Spatial DAE Coupling (I_loc Root Finding) ---
                self.I_loc: (self.phi_cp - self.phi_cn) == V_ec,
                
                # --- Global Terminal Definition ---
                self.V_term: self.V_term == self.phi_cp.boundary("right", domain=self.z) - self.phi_cn.boundary("right", domain=self.z)
            },
            "boundaries": {
                # Solid Particle Fluxes (Eq 3.10)
                flux_sn: {"left": 0.0, "right": self.I_loc / (F * a_n * L_n)},
                flux_sp: {"left": 0.0, "right": -self.I_loc / (F * a_p * L_p)},
                
                # Electrolyte Flux (Sealed Outer Edges)
                flux_en: {"left": 0.0},
                flux_ep: {"right": 0.0},
                
                # Current Collector Tabs (Placed at Top, z=0.137)
                self.phi_cn: {"right": fx.Dirichlet(0.0)},
                i_cn: {"left": 0.0},
                
                # Geometric current density flowing OUT of the tab cross-section
                i_cp: {"left": 0.0, "right": self.I_app / (L_y * L_cp)},
                
                # Tab Cooling Boundary (Eq 4.3)
                flux_T: {"left": 0.0, "right": h_tab_eff * (self.T_cell.boundary("right", domain=self.z) - T_inf)}
            },
            "initial_conditions": {
                self.c_sn: 19986.609595075,
                self.c_sp: 30730.7554385565,
                self.c_e: 1000.0,
                self.phi_cn: 0.0,
                self.phi_cp: 4.1,
                self.T_cell: T_inf,
                self.I_loc: 1e-6, #to push it off the flat gradient plateau at t=0
                self.V_term: 4.1,
                self.I_app: 0.0
            }
        }

if __name__ == "__main__":
    print("Compiling 1+1D Thermally-Coupled SPMe to Native Machine Code...")
    model = Marquis1Plus1D_SPMe()
    engine = fx.Engine(model=model, target="cpu", solver_backend="native")
    
    protocol = Sequence([
        CC(rate=2.041848, until=model.V_term <= 3.0, time=3600),
        Rest(time=600)
    ])
    
    print("\nExecuting 3C Discharge Protocol with Tab Cooling...")
    res = engine.solve(protocol=protocol)

    # =========================================================================
    # FIGURE 1: Full Cell 1+1D Diagnostics
    # =========================================================================
    t_mask = res["I_app"].data > 1.0 
    t_eval = res["Time [s]"].data[t_mask]
    capacity_ah = (t_eval * 2.041848) / 3600.0
    z_coords = np.linspace(0, 137, 15)
    x_coords = np.linspace(0, 225, 25)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Marquis et al. (2020) - 1+1D SPMe: Transverse Tab Diagnostics (3C)", fontsize=14, fontweight="bold")

    # [0, 0] Terminal Voltage & Cell Temperature (Rep Fig 2a/6a)
    ax_v = axs[0, 0]
    ax_t = ax_v.twinx()
    ax_v.plot(capacity_ah, res["V_term"].data[t_mask], 'tab:blue', linewidth=2, label="Terminal Voltage")
    T_avg = np.mean(res["T_cell"].data[t_mask], axis=1)
    ax_t.plot(capacity_ah, T_avg, 'tab:red', linewidth=2, label="Avg Temperature")
    ax_v.set(title="Cell Output Dynamics (Rep. Fig 2a/6a)", xlabel="Discharge Capacity [Ah]", ylabel="Voltage [V]")
    ax_t.set_ylabel("Temperature [K]", color='tab:red')
    ax_v.grid(True, linestyle="--", alpha=0.6)

    # [0, 1] Through-Cell Current Density (Rep Fig 10b)
    ax_i = axs[0, 1]
    I_loc_history = res["I_loc"].data[t_mask]
    for pct in [0.1, 0.5, 0.9]:
        idx = int(len(I_loc_history) * pct)
        ax_i.plot(z_coords, I_loc_history[idx], linewidth=2, label=f"{int(pct*100)}% DoD")
    ax_i.set(title="Transverse Current Distribution (Rep. Fig 10b)", xlabel="z [mm]", ylabel=r"Local Current Density, $\mathcal{I}$ [A/m$^2$]")
    ax_i.legend(); ax_i.grid(True, linestyle="--", alpha=0.6)

    # [1, 0] Electrolyte Concentration Variance
    ax_ce = axs[1, 0]
    c_e_final = res["c_e"].data[t_mask][-1].reshape((15, 25))
    # Tab is now physically located at z=137mm (index -1)
    ax_ce.plot(x_coords, c_e_final[-1, :], 'tab:purple', linewidth=2, label="Top (Near Tab)")
    ax_ce.plot(x_coords, c_e_final[0, :], 'tab:orange', linewidth=2, label="Bottom (Away from Tab)")
    ax_ce.axvline(100, color='k', linestyle=':', alpha=0.5, label="Separator Interfaces")
    ax_ce.axvline(125, color='k', linestyle=':', alpha=0.5)
    ax_ce.set(title="Electrolyte Polarization Variance (Transverse Z-Axis)", xlabel="Through-Cell Distance, x [um]", ylabel=r"Electrolyte Conc. [mol/m$^3$]")
    ax_ce.legend(); ax_ce.grid(True, linestyle="--", alpha=0.6)

    # [1, 1] Transverse Potential Drop (Rep Fig 5b)
    ax_phi = axs[1, 1]
    idx_50 = int(len(t_eval) * 0.5)
    phi_cn_50 = res["phi_cn"].data[t_mask][idx_50] * 1000.0 # Convert to mV
    ax_phi.plot(z_coords, phi_cn_50, 'k-', linewidth=2)
    ax_phi.set(title="Ohmic Drop in Negative Current Collector (Rep. Fig 5b)", xlabel="z [mm]", ylabel=r"Potential, $\phi_{s,cn}$ [mV]")
    ax_phi.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.show()