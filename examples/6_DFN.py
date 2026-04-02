import ion_flux as fx
from ion_flux.protocols import Sequence, CC, CV, Rest

class DFN(fx.PDE):
    """
    1D-1D Doyle-Fuller-Newman model utilizing Topological Sub-Meshing, 
    Discrete IR formulation, and Explicit Equation Targeting.
    """
    # -------------------------------------------------------------------------
    # 1. Topological Sub-Meshing
    # Enforces internal FVM continuity (State & Flux) across region boundaries.
    # -------------------------------------------------------------------------
    cell = fx.Domain(bounds=(0, 100e-6), resolution=50)
    x_n = cell.region(bounds=(0, 40e-6), resolution=20, name="x_n")
    x_s = cell.region(bounds=(40e-6, 60e-6), resolution=10, name="x_s")
    x_p = cell.region(bounds=(60e-6, 100e-6), resolution=20, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_p") 
    
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
        De, ke = 1e-10, 1.0
        Ds_n, Ds_p = 1e-14, 1e-14
        sig_n, sig_p = 100.0, 100.0
        F = 96485.0
        aF_n = (3.0 / 5e-6) * F
        aF_p = (3.0 / 5e-6) * F
        
        # ---------------------------------------------------------------------
        # Strongly Typed Tensors (grad naturally evaluates to CellFaces)
        # ---------------------------------------------------------------------
        N_e = -De * fx.grad(self.c_e)
        i_e = -ke * fx.grad(self.phi_e)
        i_s_n = -sig_n * fx.grad(self.phi_s_n)
        i_s_p = -sig_p * fx.grad(self.phi_s_p)
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n)
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p)
        
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p) 
        
        U_n = 0.1 - 0.0001 * c_surf_n 
        U_p = 4.2 - 0.0001 * c_surf_p 
        j_n = 1e6 * (self.phi_s_n - self.phi_e - U_n) 
        j_p = 1e6 * (self.phi_s_p - self.phi_e - U_p)

        # ---------------------------------------------------------------------
        # Explicit Equation Targeting
        # ---------------------------------------------------------------------
        return {
            "equations": {
                # --- Electrolyte (Continuous Piecewise Physics) ---
                self.c_e: fx.Piecewise({
                    self.x_n: fx.dt(self.c_e) == -fx.div(N_e) + (j_n / F),
                    self.x_s: fx.dt(self.c_e) == -fx.div(N_e),
                    self.x_p: fx.dt(self.c_e) == -fx.div(N_e) + (j_p / F)
                }),
                self.phi_e: fx.Piecewise({
                    self.x_n: fx.div(i_e) == j_n,
                    self.x_s: fx.div(i_e) == 0.0,
                    self.x_p: fx.div(i_e) == j_p
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
                N_e:          {"left": 0.0, "right": 0.0},
                self.phi_e:   {"left": fx.Dirichlet(0.0)}, # Anchor node
                
                i_s_n:        {"left": self.i_app, "right": 0.0},
                i_s_p:        {"left": 0.0, "right": self.i_app},
                
                N_s_n:        {"left": 0.0, "right": j_n / aF_n},
                N_s_p:        {"left": 0.0, "right": j_p / aF_p},
            },
            
            "initial_conditions": {
                self.c_e: 1000.0,     self.phi_e: 0.0,
                self.phi_s_n: 0.05,   self.phi_s_p: 4.15,
                self.c_s_n: 500.0,    self.c_s_p: 500.0,
                self.V_cell: 4.10,    self.i_app: 0.0
            }
        }

if __name__ == "__main__":
    # Bandwidth=0 signals FAER LU/GMRES handles internal cross-domain sparsity natively
    engine = fx.Engine(model=DFN(), target="cpu:serial", jacobian_bandwidth=0)
    protocol = Sequence([CC(rate=30.0, time=3600), Rest(time=600)])
    res = engine.solve(protocol=protocol)
    res.plot_dashboard()