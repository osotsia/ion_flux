import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class DFN(fx.PDE):
    """
    1D-1D Doyle-Fuller-Newman model utilizing explicitly coupled 
    spatial regions to guarantee numerical convergence.
    """
    # 1. Topology
    x_n = fx.Domain(bounds=(0, 40e-6), resolution=20, name="x_n")
    x_s = fx.Domain(bounds=(40e-6, 60e-6), resolution=10, name="x_s")
    x_p = fx.Domain(bounds=(60e-6, 100e-6), resolution=20, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_n") 
    r_p = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r_p") 
    
    macro_n = x_n * r_n 
    macro_p = x_p * r_p 
    
    # 2. States
    c_e_n = fx.State(domain=x_n, name="c_e_n")
    c_e_s = fx.State(domain=x_s, name="c_e_s")
    c_e_p = fx.State(domain=x_p, name="c_e_p")
    
    phi_e_n = fx.State(domain=x_n, name="phi_e_n")
    phi_e_s = fx.State(domain=x_s, name="phi_e_s")
    phi_e_p = fx.State(domain=x_p, name="phi_e_p")
    
    c_s_n = fx.State(domain=macro_n, name="c_s_n")
    c_s_p = fx.State(domain=macro_p, name="c_s_p")
    
    phi_s_n = fx.State(domain=x_n, name="phi_s_n")
    phi_s_p = fx.State(domain=x_p, name="phi_s_p")
    
    V_cell = fx.State(domain=None, name="V_cell") 
    i_app = fx.State(domain=None, name="i_app")
    
    # 3. Hardware Abstraction
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        # Base parameters
        De, ke = 1e-10, 1.0
        Ds_n, Ds_p = 1e-14, 1e-14
        sig_n, sig_p = 100.0, 100.0
        
        # Continuous internal fluxes
        N_e_n = -De * fx.grad(self.c_e_n) 
        N_e_s = -De * fx.grad(self.c_e_s) 
        N_e_p = -De * fx.grad(self.c_e_p) 
        
        i_e_n = -ke * fx.grad(self.phi_e_n) 
        i_e_s = -ke * fx.grad(self.phi_e_s) 
        i_e_p = -ke * fx.grad(self.phi_e_p) 
        
        i_s_n = -sig_n * fx.grad(self.phi_s_n) 
        i_s_p = -sig_p * fx.grad(self.phi_s_p) 
        
        N_s_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n) 
        N_s_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p) 
        
        # Evaluate particle surface concentration dynamically
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n) 
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p) 
        
        # OCV mapping
        U_n = 0.1 - 0.0001 * c_surf_n 
        U_p = 4.2 - 0.0001 * c_surf_p 
        
        # Overpotential
        eta_n = self.phi_s_n - self.phi_e_n - U_n 
        eta_p = self.phi_s_p - self.phi_e_p - U_p 
        
        # Butler-Volmer
        j_n = 1.0 * eta_n 
        j_p = 1.0 * eta_p 
        
        return {
            "regions": {
                self.x_n: [
                    fx.dt(self.c_e_n) == -fx.div(N_e_n) + j_n,
                    0 == fx.div(i_e_n) - j_n,
                    0 == fx.div(i_s_n) + j_n
                ],
                self.x_s: [
                    fx.dt(self.c_e_s) == -fx.div(N_e_s),
                    0 == fx.div(i_e_s)
                ],
                self.x_p: [
                    fx.dt(self.c_e_p) == -fx.div(N_e_p) + j_p,
                    0 == fx.div(i_e_p) - j_p,
                    0 == fx.div(i_s_p) + j_p
                ],
                self.macro_n: [
                    fx.dt(self.c_s_n) == -fx.div(N_s_n, axis=self.r_n)
                ],
                self.macro_p: [
                    fx.dt(self.c_s_p) == -fx.div(N_s_p, axis=self.r_p)
                ]
            },
            "boundaries": [
                # Electrolyte Flux Matching
                N_e_n.left == 0.0, N_e_p.right == 0.0, 
                N_e_n.right == N_e_s.left, 
                N_e_s.right == N_e_p.left,
                
                # Electrolyte State Continuity (Crucial for 2nd-order PDEs)
                self.c_e_n.right == self.c_e_s.left, 
                self.c_e_s.right == self.c_e_p.left,
                self.phi_e_n.right == self.phi_e_s.left, 
                self.phi_e_s.right == self.phi_e_p.left,
                
                # Electrolyte Potential Boundary
                self.phi_e_n.left == 0.0,
                i_e_n.right == i_e_s.left, 
                i_e_s.right == i_e_p.left, 
                i_e_p.right == 0.0,
                
                # Solid Potential Boundaries
                i_s_n.left == -self.i_app, i_s_n.right == 0.0,
                i_s_p.left == 0.0, i_s_p.right == self.i_app,
                
                # Particle Boundaries
                N_s_n.boundary("left", domain=self.r_n) == 0.0,  
                N_s_n.boundary("right", domain=self.r_n) == -j_n, 
                N_s_p.boundary("left", domain=self.r_p) == 0.0,  
                N_s_p.boundary("right", domain=self.r_p) == -j_p 
            ],
            "global": [
                self.V_cell == self.phi_s_p.right - self.phi_s_n.left,
                self.c_e_n.t0 == 1000.0, self.c_e_s.t0 == 1000.0, self.c_e_p.t0 == 1000.0, 
                self.phi_e_n.t0 == 0.0, self.phi_e_s.t0 == 0.0, self.phi_e_p.t0 == 0.0,
                self.phi_s_n.t0 == 0.05, self.phi_s_p.t0 == 4.15, 
                self.c_s_n.t0 == 500.0, self.c_s_p.t0 == 500.0, 
                self.V_cell.t0 == 4.10, self.i_app.t0 == 0.0 
            ]
        }

if __name__ == "__main__":
    # Instantiate the compilation engine and target the local CPU
    engine = fx.Engine(model=DFN(), target="cpu:serial")

    # Define a 1-hour (3600s) Constant Current (CC) discharge
    # The applied rate (e.g., 30.0A) is assumed to correspond to the cell's 1C capacity
    protocol = Sequence([
        CC(rate=30.0, time=3600),
        Rest(time=600)
    ])

    print("Executing protocol...")
    res = engine.solve(protocol=protocol)

    print("Simulation Complete. Launching Dashboard.")
    res.plot_dashboard()