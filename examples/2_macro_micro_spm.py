import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class MacroMicroSPM(fx.PDE):
    """
    1D-1D Macro-Micro Model.
    Validates hierarchical cross-product meshes and regional algebraic constraints (DAEs).
    Excludes electrolyte transport to guarantee convergence.
    """
    x_n = fx.Domain(bounds=(0, 40e-6), resolution=10, name="x_n")
    x_p = fx.Domain(bounds=(60e-6, 100e-6), resolution=10, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_p") 
    
    macro_n = x_n * r_n 
    macro_p = x_p * r_p 
    
    c_s_n = fx.State(domain=macro_n, name="c_s_n")
    c_s_p = fx.State(domain=macro_p, name="c_s_p")
    
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")
    
    V_cell = fx.State(name="V_cell") 
    i_app = fx.State(name="i_app")
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    def math(self):
        Ds_n, Ds_p = 1e-14, 1e-14
        sig_n, sig_p = 100.0, 100.0

        i_s_n = -sig_n * fx.grad(self.phi_s_n, axis=self.x_n) 
        i_s_p = -sig_p * fx.grad(self.phi_s_p, axis=self.x_p) 
        
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n) 
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p) 
        
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p) 
        
        U_n = 0.1 - 0.0001 * c_surf_n 
        U_p = 4.2 - 0.0001 * c_surf_p 
        
        # Multiply by 1e6 to represent a physically realistic specific area * exchange current
        j_n = 1e6 * (self.phi_s_n - U_n) 
        j_p = 1e6 * (self.phi_s_p - U_p) 

        # AST Equilibration: Scale massive spatial DAEs to O(1) to bypass f64 ULP noise
        eq_scale = 1e-12
        
        # Faraday Conversion: Volumetric current (A/m^3) to Area flux (mol/m^2 s)
        # a = 3 / R_p = 6e5. F = 96485. a*F = ~5.78e10
        aF = 5.78e10

        return {
            "regions": {
                self.x_n: [ 0 == (fx.div(i_s_n, axis=self.x_n) + j_n) * eq_scale ],
                self.x_p: [ 0 == (fx.div(i_s_p, axis=self.x_p) + j_p) * eq_scale ],
                self.macro_n: [ fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n) ],
                self.macro_p: [ fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p) ]
            },
            "boundaries": [
                i_s_n.left == -self.i_app, i_s_n.right == 0.0,
                i_s_p.left == 0.0, i_s_p.right == -self.i_app,
                
                N_s_n.boundary("left", domain=self.r_n) == 0.0,  
                N_s_n.boundary("right", domain=self.r_n) == -j_n / aF, 
                N_s_p.boundary("left", domain=self.r_p) == 0.0,  
                N_s_p.boundary("right", domain=self.r_p) == -j_p / aF 
            ],
            "global": [
                self.V_cell == self.phi_s_p.right - self.phi_s_n.left,
                
                self.phi_s_n.t0 == 0.05, 
                self.phi_s_p.t0 == 4.15, 
                self.c_s_n.t0 == 500.0, 
                self.c_s_p.t0 == 500.0, 
                self.V_cell.t0 == 4.10, 
                self.i_app.t0 == 0.0 
            ]
        }

if __name__ == "__main__":

    model=MacroMicroSPM()
    engine = fx.Engine(model, target="cpu:serial", solver_backend="native")
    
    protocol = Sequence([
        CC(rate=30.0, until=model.V_cell <= 3.0, time=3600),
        Rest(time=600)
    ])
    
    print("Executing Macro-Micro SPM protocol...")
    res = engine.solve(protocol=protocol)
    print(f"Simulation Complete. Final Voltage: {res['V_cell'].data[-1]:.3f} V")

    print("Launching Dashboard.")
    res.plot_dashboard()