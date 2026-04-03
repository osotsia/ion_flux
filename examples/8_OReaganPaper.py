import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class DFN(fx.PDE):
    """
    1D-1D Doyle-Fuller-Newman model utilizing Topological Sub-Meshing, 
    Discrete IR formulation, and Explicit Equation Targeting.
    
    Updated to strictly reflect the parameterization of the LG M50 21700 cell 
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
    phi_e = fx.State(domain=cell, name="phi_e")
    
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")
    
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n")
    c_s_p = fx.State(domain=x_p * r_p, name="c_s_p")
    
    V_cell = fx.State(domain=None, name="V_cell") 
    i_app = fx.State(domain=None, name="i_app")
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        import math

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
        
        # Solid Conductivities (Effective via Bruggeman)
        sig_eff_n = 215.0 * (eps_sn ** b_brug)
        sig_eff_p = 0.847 * (eps_sp ** b_brug)
        
        # Electrolyte Parameters (Approximate baseline constants)
        De = 3e-10
        ke = 1.0
        t_plus = 0.38
        
        # Active Electrode Area (From paper dimensions: 80 cm x 6.4 cm, double-sided)
        A_elec = 0.1024 
        
        # Solid Diffusion Coefficients (Approx constants at 25C)
        Ds_n = 3.3e-14
        Ds_p = 4.0e-15

        # ---------------------------------------------------------------------
        # Pre-Calculate EXACT t=0 Equilibrium Floats for Initial Conditions
        # ---------------------------------------------------------------------
        c_n_init = 28866.0
        c_p_init = 13975.0
        
        x_n_init = c_n_init / c_max_n
        x_p_init = c_p_init / c_max_p
        
        # Evaluate standard math.tanh floats for numerical stability at t=0
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
            ex = fx.exp(x)
            return 0.5 * (ex - 1.0 / ex)

        # ---------------------------------------------------------------------
        # Thermodynamics & Kinetics (AST Nodes)
        # ---------------------------------------------------------------------
        # Safe extraction of variables to prevent NaNs in fractional powers
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p) 
        
        x_n = fx.min(fx.max(c_surf_n / c_max_n, 1e-4), 0.9999)
        x_p = fx.min(fx.max(c_surf_p / c_max_p, 1e-4), 0.9999)
        ce_safe = fx.max(self.c_e, 1e-4)
        
        # OCV Functions (O'Regan Eq S7 & S8 mapped into the computational graph)
        U_n = (1.9793 * fx.exp(-39.3631 * x_n) + 0.2482 
               - 0.0909 * tanh_ast(29.8538 * (x_n - 0.1234)) 
               - 0.04478 * tanh_ast(14.9159 * (x_n - 0.2769)) 
               - 0.0205 * tanh_ast(30.4444 * (x_n - 0.6103)))
               
        U_p = (-0.8090 * x_p + 4.4875 
               - 0.0428 * tanh_ast(18.5138 * (x_p - 0.5542)) 
               - 17.7326 * tanh_ast(15.7890 * (x_p - 0.3117)) 
               + 17.5842 * tanh_ast(15.9308 * (x_p - 0.3120)))

        # Exchange Current Density (Butler-Volmer per O'Regan Table S6)
        # Note: 10000.0 multiplier converts from A/cm^2 to A/m^2
        i0_n = 10000.0 * 0.0002668 * (1.0 - x_n)**0.208 * x_n**0.792 * (ce_safe / 1000.0)**0.208
        i0_p = 10000.0 * 0.0005028 * (1.0 - x_p)**0.570 * x_p**0.430 * (ce_safe / 1000.0)**0.570
        
        eta_n = self.phi_s_n - self.phi_e - U_n
        eta_p = self.phi_s_p - self.phi_e - U_p
        
        j_n = a_n * i0_n * sinh_ast((F / (2.0 * R_const * T)) * eta_n)
        j_p = a_p * i0_p * sinh_ast((F / (2.0 * R_const * T)) * eta_p)

        # ---------------------------------------------------------------------
        # Tensors (AST Nodes)
        # ---------------------------------------------------------------------
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p)
        
        i_s_n = -sig_eff_n * fx.grad(self.phi_s_n)
        i_s_p = -sig_eff_p * fx.grad(self.phi_s_p)
        
        # Spatial masking for effective properties over the continuous cell domain
        x = self.cell.coords
        is_n = x <= 85.2e-6
        is_p = x >= 97.2e-6
        is_s = 1.0 - is_n - is_p
        
        eps_e = is_n * eps_en + is_s * eps_es + is_p * eps_ep
        De_eff = De * (eps_e ** b_brug)
        ke_eff = ke * (eps_e ** b_brug)
        
        grad_ce = fx.grad(self.c_e)
        grad_phie = fx.grad(self.phi_e)
        
        # Concentration gradient term for electrolyte charge conservation
        ce_diff_term = (2.0 * R_const * T / F) * (1.0 - t_plus) * (grad_ce / ce_safe)
        
        # SINGLE, globally stitched FVM fluxes to prevent mass discontinuity
        flux_ce = -De_eff * grad_ce
        flux_phie = -ke_eff * grad_phie + ke_eff * ce_diff_term

        # ---------------------------------------------------------------------
        # Explicit Equation Targeting
        # ---------------------------------------------------------------------
        return {
            "equations": {
                # --- Electrolyte (Continuous Piecewise Physics) ---
                self.c_e: fx.Piecewise({
                    self.x_n: eps_en * fx.dt(self.c_e) == -fx.div(flux_ce) + (1.0 - t_plus) * j_n / F,
                    self.x_s: eps_es * fx.dt(self.c_e) == -fx.div(flux_ce),
                    self.x_p: eps_ep * fx.dt(self.c_e) == -fx.div(flux_ce) + (1.0 - t_plus) * j_p / F
                }),
                self.phi_e: fx.Piecewise({
                    self.x_n: fx.div(flux_phie) == j_n,
                    self.x_s: fx.div(flux_phie) == 0.0,
                    self.x_p: fx.div(flux_phie) == j_p
                }),
                
                # --- Solid Phase ---
                self.phi_s_n: fx.div(i_s_n) == -j_n,
                self.phi_s_p: fx.div(i_s_p) == -j_p,
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n),
                self.c_s_p: fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p),
                
                # --- Global Algebraic ---
                self.V_cell: self.V_cell == self.phi_s_p.right - self.phi_s_n.left
            },
            
            # -----------------------------------------------------------------
            # Explicit Boundaries (Dirichlet on States, Neumann on Tensors)
            # -----------------------------------------------------------------
            "boundaries": {
                flux_ce:      {"left": 0.0, "right": 0.0},
                
                self.phi_s_n: {"left": fx.Dirichlet(0.0)}, # Grounded anchor node
                i_s_n:        {"right": 0.0},
                # Map cyclic total current [A] to geometric current density [A/m^2]
                i_s_p:        {"left": 0.0, "right": self.i_app / A_elec}, 
                
                flux_phie:    {"left": 0.0, "right": 0.0},
                
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
    # Bandwidth=0 signals FAER LU/GMRES handles internal cross-domain sparsity natively
    engine = fx.Engine(model=DFN(), target="cpu:serial", jacobian_bandwidth=0)

    #import json
    #print(engine.ast_payload)
    
    # Simulate Figure 15: 1C Discharge (5A nominal capacity) down to 2.5V cutoff
    protocol = Sequence([
        CC(rate=5.0, until=engine.model.V_cell <= 2.5, time=3600), 
        Rest(time=600)
    ])
    
    res = engine.solve(protocol=protocol)
    
    res.plot_dashboard(variables=[
        ["V_cell"],
        ["phi_s_n", "phi_e", "phi_s_p"],
        ["c_s_n", "c_s_p"],
        ["c_e"]
    ])