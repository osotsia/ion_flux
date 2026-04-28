"""
CPR Exactness Oracle

This suite proves that the Hybrid Graph Coloring (Phase 2) and the Forward/Reverse-Mode 
AD evaluation (Phase 3) perfectly reconstruct the exact analytical Jacobian. 

It explicitly compares the newly assembled CPR Jacobian against the legacy LLVM 
block-wise Sparse Jacobian. If these matrices diverge by even 1e-10, the VJP/JVP 
scaling or color maps are flawed.
"""

import pytest
import numpy as np
import ctypes
import shutil
import platform
from ion_flux.runtime.engine import Engine
import ion_flux as fx
import math

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


def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

@REQUIRES_COMPILER
def test_cpr_hybrid_jacobian_matches_legacy_llvm_jacobian():
    """
    PROBE: Reconstructs the Jacobian using the legacy LLVM `evaluate_jacobian_sparse`
    and compares it to the new Python/Rust CPR hybrid `evaluate_jacobian`.
    """
    engine = Engine(model=ThermalDFN(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    
    # Randomize states to ensure no structural zero is hidden by 0.0 derivatives
    np.random.seed(42)
    y = np.random.uniform(0.1, 1.0, N).tolist()
    ydot = np.random.uniform(0.1, 1.0, N).tolist()
    c_j = 12.34 # Arbitrary implicit scaling factor
    
    p_list = engine._pack_parameters({})
    m_list = engine.layout.get_mesh_data()
    
    # -------------------------------------------------------------------------
    # 1. Legacy LLVM Sparse Jacobian (The Ground Truth)
    # -------------------------------------------------------------------------
    rows = (ctypes.c_int * (N * 50))()
    cols = (ctypes.c_int * (N * 50))()
    vals = (ctypes.c_double * (N * 50))()
    nnz = ctypes.c_int(0)
    
    y_arr = (ctypes.c_double * N)(*y)
    ydot_arr = (ctypes.c_double * N)(*ydot)
    p_arr = (ctypes.c_double * len(p_list))(*p_list)
    m_arr = (ctypes.c_double * len(m_list))(*m_list)
    
    engine.runtime.dll.evaluate_jacobian_sparse(
        y_arr, ydot_arr, p_arr, m_arr, ctypes.c_double(c_j), 
        rows, cols, vals, ctypes.byref(nnz)
    )
    
    J_legacy = np.zeros((N, N))
    for i in range(nnz.value):
        J_legacy[rows[i], cols[i]] = vals[i]
        
    # -------------------------------------------------------------------------
    # 2. New CPR Hybrid Jacobian (JVP + VJP)
    # -------------------------------------------------------------------------
    # `engine.evaluate_jacobian` was updated in Phase 3 to execute the exact 
    # CPR logic used by the Rust backend.
    J_cpr = np.array(engine.evaluate_jacobian(y, ydot, c_j, parameters={}))
    
    # -------------------------------------------------------------------------
    # 3. Assert Exactness
    # -------------------------------------------------------------------------
    # We verify that both matrices are mathematically identical.
    np.testing.assert_allclose(
        J_cpr, J_legacy, 
        rtol=1e-10, atol=1e-12,
        err_msg="FATAL: The CPR assembled Jacobian (JVP+VJP) diverges from the legacy LLVM Jacobian! "
                "The Reverse-Mode VJP dense row extraction or Forward-Mode JVP color map is flawed."
    )

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])