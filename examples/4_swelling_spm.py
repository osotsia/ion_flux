import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class SwellingSPM(fx.PDE):
    """
    Single Particle Model with physical particle swelling.
    Validates Arbitrary Lagrangian-Eulerian (ALE) kinematics and moving boundaries (Stefan problems).
    Replaces PyBaMM's pybamm.lithium_ion.SPM({"particle mechanics": "swelling only"}).
    """
    r_p = fx.Domain(bounds=(0, 10e-6), resolution=20, coord_sys="spherical", name="r_p")

    c_s_p = fx.State(domain=r_p, name="c_s_p")
    R_particle = fx.State(name="R_particle")
    V_cell = fx.State(name="V_cell")
    i_app = fx.State(name="i_app")

    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    def math(self):
        Ds_p = 1e-14
        flux_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p)
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p)

        j_flux = self.i_app / 96485.0
        
        # Swelling ODE proportional to internal state integral
        average_c = fx.integral(self.c_s_p, over=self.r_p)

        return {
            "regions": {
                self.r_p: [ fx.dt(self.c_s_p) == -fx.div(flux_p, axis=self.r_p) ]
            },
            "boundaries": [
                # Dynamic moving boundary binding (Compiler automatically applies ALE grid convection)
                self.r_p.right == self.R_particle,
                
                flux_p.boundary("left", domain=self.r_p) == 0.0,
                flux_p.boundary("right", domain=self.r_p) == j_flux
            ],
            "global": [
                # Particle swelling ODE
                fx.dt(self.R_particle) == 1e-12 * average_c,
                
                self.V_cell == 4.2 - 0.0001 * c_surf_p - 0.05 * self.i_app,
                
                self.c_s_p.t0 == 200.0,
                self.R_particle.t0 == 10e-6,
                self.V_cell.t0 == 4.18,
                self.i_app.t0 == 0.0
            ]
        }

if __name__ == "__main__":
    engine = fx.Engine(model=SwellingSPM(), target="cpu:serial")
    
    protocol = Sequence([
        CC(rate=10.0, until=fx.Condition("V_cell <= 3.0"), time=3600),
        Rest(time=600)
    ])
    
    print("Executing Swelling SPM protocol...")
    res = engine.solve(protocol=protocol)
    print(f"Simulation Complete. Final Voltage: {res['V_cell'].data[-1]:.3f} V")
    print(f"Final Particle Radius: {res['R_particle'].data[-1]*1e6:.3f} um")

    print("Launching Dashboard.")
    res.plot_dashboard()