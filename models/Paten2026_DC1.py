"""
Reproduction of Figure 8 from Paten et al. (2026), J. Electrochem. Soc. 173 030503:
"Dual-Continuum Models of Lithium-Ion Batteries are Fast and Accurate Alternatives 
to the Doyle-Fuller-Newman Approach: I. Derivation and Validation"

Comparison of DFN vs. First-Order Dual-Continuum (DC1) models on a commercial 
LG M50 21700 cell (NMC 811 / Graphite-SiOx) at 0.5C, 1.0C, and 1.5C discharge rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import ion_flux as fx
from ion_flux.protocols import Sequence, CC


# =============================================================================
# 1. First-Order Dual-Continuum Model (DC1) - Paten et al. (2026)
# =============================================================================
class Paten2026_DC1(fx.PDE):
    """
    Dual-Continuum First-Order Model (DC1).
    AM mass balance is volume-averaged (purely 1D in x), eliminating the microscale 
    radial dimension r. Surface concentration is computed algebraically via Eq. 47.
    """
    # Macroscale 1D Domain Topology (Chen et al. 2020, Table II)
    cell = fx.Domain(bounds=(0, 172.8e-6))
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=35, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=6, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=31, name="x_p")

    # Continuum State Variables
    c_e = fx.State(domain=cell, name="c_e")         # Electrolyte concentration [mol/m^3]
    phi_e = fx.State(domain=cell, name="phi_e")     # Electrolyte potential [V]
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")  # Solid potential Anode [V]
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")  # Solid potential Cathode [V]

    # AM Phase Concentrations: Purely 1D (No r-dimension)
    c_k_n = fx.State(domain=x_n, name="c_k_n")          # Volume-averaged conc <c_n>^n [mol/m^3]
    c_surf_n = fx.State(domain=x_n, name="c_surf_n")    # Surface conc <c_n>_A [mol/m^3] (Spatial DAE)
    c_k_p = fx.State(domain=x_p, name="c_k_p")          # Volume-averaged conc <c_p>^p [mol/m^3]
    c_surf_p = fx.State(domain=x_p, name="c_surf_p")    # Surface conc <c_p>_A [mol/m^3] (Spatial DAE)

    V_cell = fx.State(domain=None, name="V_cell")
    i_app = fx.State(domain=None, name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    # Parameters from Paten et al. (2026), Table IV (DC1 Column)
    k_n = fx.Parameter(default=9.480e-7, name="k_n")
    k_p = fx.Parameter(default=2.053e-6, name="k_p")
    tau_e_n = fx.Parameter(default=2.735, name="tau_e_n")
    tau_e_p = fx.Parameter(default=1.944, name="tau_e_p")
    a_n = fx.Parameter(default=441818.3, name="a_n")
    a_p = fx.Parameter(default=412729.1, name="a_p")
    s_A_n = fx.Parameter(default=-335.06, name="s_A_n")   # Closure mapping <s>_A,n [mol m^-1 A^-1]
    s_A_p = fx.Parameter(default=-500.38, name="s_A_p")   # Closure mapping <s>_A,p [mol m^-1 A^-1]
    c_max_n = fx.Parameter(default=33304.8, name="c_max_n")
    c_max_p = fx.Parameter(default=57399.3, name="c_max_p")

    def math(self):
        F, R_const, T = 96485.0, 8.314, 298.15
        A_elec = 0.1027
        i_den = self.i_app / A_elec

        # Porosities & Volume Fractions (Chen et al. 2020)
        eps_n, eps_s, eps_p = 0.25, 0.47, 0.335
        eps_act_n, eps_act_p = 0.75, 0.665
        sig_n, sig_p = 215.0, 0.18
        t_plus = 0.2594

        # Numerical stabilization helpers
        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)

        def sinh_ast(x):
            x_max = 5.0
            cosh_max = 0.5 * (np.exp(x_max) + np.exp(-x_max))
            x_safe = fx.clamp(x, lower=-x_max, upper=x_max)
            return 0.5 * (fx.exp(x_safe) - fx.exp(-x_safe)) + cosh_max * (x - x_safe)

        # Interfacial Concentration State Clamping
        c_s_n_safe = fx.clamp(self.c_surf_n, lower=10.0, upper=self.c_max_n - 10.0)
        c_s_p_safe = fx.clamp(self.c_surf_p, lower=10.0, upper=self.c_max_p - 10.0)
        x_stoich_n = c_s_n_safe / self.c_max_n
        x_stoich_p = c_s_p_safe / self.c_max_p
        ce_safe = fx.clamp(self.c_e, lower=10.0, upper=5000.0)

        # Open-Circuit Potentials (Chen et al. 2020, Eqs. 8-9)
        U_n = (1.9793 * fx.exp(-39.3631 * x_stoich_n) + 0.2482
               - 0.0909 * tanh_ast(29.8538 * (x_stoich_n - 0.1234))
               - 0.04478 * tanh_ast(14.9159 * (x_stoich_n - 0.2769))
               - 0.0205 * tanh_ast(30.4444 * (x_stoich_n - 0.6103)))

        U_p = (-0.8090 * x_stoich_p + 4.4875
               - 0.0428 * tanh_ast(18.5138 * (x_stoich_p - 0.5542))
               - 17.7326 * tanh_ast(15.7890 * (x_stoich_p - 0.3117))
               + 17.5842 * tanh_ast(15.9308 * (x_stoich_p - 0.3120)))

        # Exchange Current Densities [A/m^2]
        j0_n = self.k_n * fx.sqrt(ce_safe * c_s_n_safe * (self.c_max_n - c_s_n_safe))
        j0_p = self.k_p * fx.sqrt(ce_safe * c_s_p_safe * (self.c_max_p - c_s_p_safe))

        eta_n = self.phi_s_n - self.phi_e - U_n
        eta_p = self.phi_s_p - self.phi_e - U_p

        F_RT = F / (2.0 * R_const * T)
        j_n = 2.0 * j0_n * sinh_ast(F_RT * eta_n)
        j_p = 2.0 * j0_p * sinh_ast(F_RT * eta_p)

        J_n = self.a_n * j_n  # Volumetric current density [A/m^3]
        J_p = self.a_p * j_p

        # Electrolyte Transport Properties (Chen et al. 2020, Eqs. 23-24)
        c_L = ce_safe / 1000.0
        D_e = 8.794e-11 * c_L**2 - 3.972e-10 * c_L + 4.862e-10
        k_e = 0.1297 * c_L**3 - 2.51 * c_L**1.5 + 3.329 * c_L

        # Effective Transport Scaled by Tortuosity (Table IV)
        D_eff_n, k_eff_n = D_e * (eps_n / self.tau_e_n), k_e * (eps_n / self.tau_e_n)
        D_eff_s, k_eff_s = D_e * (eps_s ** 1.5),         k_e * (eps_s ** 1.5)
        D_eff_p, k_eff_p = D_e * (eps_p / self.tau_e_p), k_e * (eps_p / self.tau_e_p)

        ce_diff_term = (2.0 * R_const * T / F) * (1.0 - t_plus) * (fx.grad(ce_safe) / ce_safe)
        flux_ce_n = -D_eff_n * fx.grad(self.c_e)
        flux_ce_s = -D_eff_s * fx.grad(self.c_e)
        flux_ce_p = -D_eff_p * fx.grad(self.c_e)

        i_e_n = -k_eff_n * fx.grad(self.phi_e) + k_eff_n * ce_diff_term
        i_e_s = -k_eff_s * fx.grad(self.phi_e) + k_eff_s * ce_diff_term
        i_e_p = -k_eff_p * fx.grad(self.phi_e) + k_eff_p * ce_diff_term

        i_s_n = -sig_n * fx.grad(self.phi_s_n)
        i_s_p = -sig_p * fx.grad(self.phi_s_p)

        return {
            "equations": {
                # Electrolyte Conservation
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

                # DC1 Volume-Averaged AM Mass Conservation [Eqs. 37, 46]
                self.c_k_n: eps_act_n * fx.dt(self.c_k_n) == -J_n / F,
                self.c_k_p: eps_act_p * fx.dt(self.c_k_p) == -J_p / F,

                # DC1 First-Order Surface Mapping [Eq. 47] (Spatial Algebraic Constraint)
                self.c_surf_n: self.c_surf_n == self.c_k_n + self.s_A_n * j_n,
                self.c_surf_p: self.c_surf_p == self.c_k_p + self.s_A_p * j_p,

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
            },
            "initial_conditions": {
                self.c_e: 1000.0,
                self.phi_s_n: 0.0,
                self.phi_e: -0.092,
                self.phi_s_p: 4.182,
                self.c_k_n: 29901.3,     # Table IV
                self.c_surf_n: 29901.3,
                self.c_k_p: 18051.3,     # Table IV
                self.c_surf_p: 18051.3,
                self.V_cell: 4.182,
                self.i_app: 0.0
            }
        }


# =============================================================================
# 2. Benchmark Doyle-Fuller-Newman (DFN) Model - Paten et al. (2026)
# =============================================================================
class Paten2026_DFN(fx.PDE):
    """
    Standard Pseudo-Two-Dimensional (P2D) DFN Model with Table IV parameters.
    Solves 1D Fickian spherical diffusion inside particles across macro nodes.
    """
    cell = fx.Domain(bounds=(0, 172.8e-6))
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=35, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=6, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=31, name="x_p")

    # Microscale Radial Domains (Table IV: R_n = 6.401 um, R_p = 5.266 um)
    r_n = fx.Domain(bounds=(0, 6.401e-6), resolution=15, coord_sys="spherical", name="r_n")
    r_p = fx.Domain(bounds=(0, 5.266e-6), resolution=15, coord_sys="spherical", name="r_p")

    c_e = fx.State(domain=cell, name="c_e")
    phi_e = fx.State(domain=cell, name="phi_e")
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")

    # Hierarchical Macro x Micro Domain
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n")
    c_s_p = fx.State(domain=x_p * r_p, name="c_s_p")

    V_cell = fx.State(domain=None, name="V_cell")
    i_app = fx.State(domain=None, name="i_app")
    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    # Parameters from Paten et al. (2026), Table IV (DFN Column)
    k_n = fx.Parameter(default=4.908e-7, name="k_n")
    k_p = fx.Parameter(default=4.414e-6, name="k_p")
    tau_e_n = fx.Parameter(default=2.738, name="tau_e_n")
    tau_e_p = fx.Parameter(default=2.802, name="tau_e_p")
    D_s_n = fx.Parameter(default=1.332e-13, name="D_s_n")
    D_s_p = fx.Parameter(default=5.126e-15, name="D_s_p")
    c_max_n = fx.Parameter(default=32694.2, name="c_max_n")
    c_max_p = fx.Parameter(default=62266.0, name="c_max_p")

    def math(self):
        F, R_const, T = 96485.0, 8.314, 298.15
        A_elec = 0.1027
        i_den = self.i_app / A_elec

        eps_n, eps_s, eps_p = 0.25, 0.47, 0.335
        eps_act_n, eps_act_p = 0.75, 0.665
        sig_n, sig_p = 215.0, 0.18
        t_plus = 0.2594

        # a_k = 3 * eps_act_k / R_k
        a_n = 3.0 * eps_act_n / 6.401e-6
        a_p = 3.0 * eps_act_p / 5.266e-6

        def tanh_ast(x):
            e2x = fx.exp(2.0 * x)
            return (e2x - 1.0) / (e2x + 1.0)

        def sinh_ast(x):
            x_max = 5.0
            cosh_max = 0.5 * (np.exp(x_max) + np.exp(-x_max))
            x_safe = fx.clamp(x, lower=-x_max, upper=x_max)
            return 0.5 * (fx.exp(x_safe) - fx.exp(-x_safe)) + cosh_max * (x - x_safe)

        # Micro-surface extraction [Eq. 51b, Table I]
        c_surf_n = fx.clamp(self.c_s_n.boundary("right", domain=self.r_n), lower=10.0, upper=self.c_max_n - 10.0)
        c_surf_p = fx.clamp(self.c_s_p.boundary("right", domain=self.r_p), lower=10.0, upper=self.c_max_p - 10.0)
        x_stoich_n = c_surf_n / self.c_max_n
        x_stoich_p = c_surf_p / self.c_max_p
        ce_safe = fx.clamp(self.c_e, lower=10.0, upper=5000.0)

        U_n = (1.9793 * fx.exp(-39.3631 * x_stoich_n) + 0.2482
               - 0.0909 * tanh_ast(29.8538 * (x_stoich_n - 0.1234))
               - 0.04478 * tanh_ast(14.9159 * (x_stoich_n - 0.2769))
               - 0.0205 * tanh_ast(30.4444 * (x_stoich_n - 0.6103)))

        U_p = (-0.8090 * x_stoich_p + 4.4875
               - 0.0428 * tanh_ast(18.5138 * (x_stoich_p - 0.5542))
               - 17.7326 * tanh_ast(15.7890 * (x_stoich_p - 0.3117))
               + 17.5842 * tanh_ast(15.9308 * (x_stoich_p - 0.3120)))

        j0_n = self.k_n * fx.sqrt(ce_safe * c_surf_n * (self.c_max_n - c_surf_n))
        j0_p = self.k_p * fx.sqrt(ce_safe * c_surf_p * (self.c_max_p - c_surf_p))

        eta_n = self.phi_s_n - self.phi_e - U_n
        eta_p = self.phi_s_p - self.phi_e - U_p

        F_RT = F / (2.0 * R_const * T)
        J_n = a_n * j0_n * sinh_ast(F_RT * eta_n)
        J_p = a_p * j0_p * sinh_ast(F_RT * eta_p)

        c_L = ce_safe / 1000.0
        D_e = 8.794e-11 * c_L**2 - 3.972e-10 * c_L + 4.862e-10
        k_e = 0.1297 * c_L**3 - 2.51 * c_L**1.5 + 3.329 * c_L

        D_eff_n, k_eff_n = D_e * (eps_n / self.tau_e_n), k_e * (eps_n / self.tau_e_n)
        D_eff_s, k_eff_s = D_e * (eps_s ** 1.5),         k_e * (eps_s ** 1.5)
        D_eff_p, k_eff_p = D_e * (eps_p / self.tau_e_p), k_e * (eps_p / self.tau_e_p)

        ce_diff_term = (2.0 * R_const * T / F) * (1.0 - t_plus) * (fx.grad(ce_safe) / ce_safe)
        flux_ce_n = -D_eff_n * fx.grad(self.c_e)
        flux_ce_s = -D_eff_s * fx.grad(self.c_e)
        flux_ce_p = -D_eff_p * fx.grad(self.c_e)

        i_e_n = -k_eff_n * fx.grad(self.phi_e) + k_eff_n * ce_diff_term
        i_e_s = -k_eff_s * fx.grad(self.phi_e) + k_eff_s * ce_diff_term
        i_e_p = -k_eff_p * fx.grad(self.phi_e) + k_eff_p * ce_diff_term

        i_s_n = -sig_n * fx.grad(self.phi_s_n)
        i_s_p = -sig_p * fx.grad(self.phi_s_p)

        # Microscale Solid Diffusion Flux [Eq. 52]
        N_s_n = -self.D_s_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -self.D_s_p * fx.grad(self.c_s_p, axis=self.r_p)

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

                # P2D Radial Transport [Eq. 51a]
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

                # Faradaic Current Injection into Micro-Particle Surface [Eq. 51b]
                N_s_n:        {"left": 0.0, "right": J_n / (a_n * F)},
                N_s_p:        {"left": 0.0, "right": J_p / (a_p * F)},
            },
            "initial_conditions": {
                self.c_e: 1000.0,
                self.phi_s_n: 0.0,
                self.phi_e: -0.092,
                self.phi_s_p: 4.182,
                self.c_s_n: 29541.5,     # Table IV
                self.c_s_p: 16842.2,     # Table IV
                self.V_cell: 4.182,
                self.i_app: 0.0
            }
        }


# =============================================================================
# 3. Execution Pipeline & Plotting (Figure 8 Reproduction)
# =============================================================================
def main():
    rates = {
        "0.5C": {"current": 2.5, "t_max": 7500},
        "1C":   {"current": 5.0, "t_max": 3800},
        "1.5C": {"current": 7.5, "t_max": 2600}
    }

    print("Compiling Paten et al. (2026) DC1 Model via ion_flux AOT Toolchain...")
    engine_dc1 = fx.Engine(model=Paten2026_DC1(), target="cpu:serial")

    print("Compiling Paten et al. (2026) DFN Model via ion_flux AOT Toolchain...")
    engine_dfn = fx.Engine(model=Paten2026_DFN(), target="cpu:serial")

    results_dc1 = {}
    results_dfn = {}

    for label, cfg in rates.items():
        current = cfg["current"]
        t_max = cfg["t_max"]
        prot = Sequence([CC(rate=current, until=engine_dc1.model.V_cell <= 2.5, time=t_max)])

        print(f"Simulating {label} ({current}A) DC1 discharge...")
        results_dc1[label] = engine_dc1.solve(protocol=prot, show_progress=False)

        prot_dfn = Sequence([CC(rate=current, until=engine_dfn.model.V_cell <= 2.5, time=t_max)])
        print(f"Simulating {label} ({current}A) DFN discharge...")
        results_dfn[label] = engine_dfn.solve(protocol=prot_dfn, show_progress=False)

    # Visual Reproduction Aligned with Fig. 8 (JES style)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "Reproduction of Figure 8: Experimental Cell Voltage vs DFN and DC1 Models\n"
        "(Paten et al. 2026, J. Electrochem. Soc. 173 030503)",
        fontsize=13, fontweight="bold"
    )

    panels = [("0.5C", "(a) 0.5C", axs[0]), ("1C", "(b) 1C", axs[1]), ("1.5C", "(c) 1.5C", axs[2])]

    for key, title, ax in panels:
        # DC1 prediction
        res_dc1 = results_dc1[key]
        ax.plot(res_dc1["Time [s]"].data, res_dc1["V_cell"].data, 'b--', linewidth=2.0, label="DC1 (Paten 2026)")

        # DFN prediction
        res_dfn = results_dfn[key]
        ax.plot(res_dfn["Time [s]"].data, res_dfn["V_cell"].data, 'r:', linewidth=2.0, label="DFN (Doyle 1993)")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylim(2.4, 4.3)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9)

    axs[0].set_ylabel("Cell Voltage [V]", fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()