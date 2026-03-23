import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class SingleParticleModel(fx.PDE):
    """
    Lumped Single Particle Model (SPM).
    Two 1D spherical particles representing the negative and positive electrodes.
    Replaces PyBaMM's pybamm.lithium_ion.SPM().
    """
    r_n = fx.Domain(bounds=(0, 10e-6), resolution=15, coord_sys="spherical", name="r_n")
    r_p = fx.Domain(bounds=(0, 10e-6), resolution=15, coord_sys="spherical", name="r_p")

    c_s_n = fx.State(domain=r_n, name="c_s_n")
    c_s_p = fx.State(domain=r_p, name="c_s_p")
    V_cell = fx.State(name="V_cell")
    i_app = fx.State(name="i_app")

    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    def math(self):
        Ds_n, Ds_p = 1e-14, 1e-14
        
        flux_n = -Ds_n * fx.grad(self.c_s_n, axis=self.r_n)
        flux_p = -Ds_p * fx.grad(self.c_s_p, axis=self.r_p)

        # Extract boundaries to calculate cell voltage
        c_surf_n = self.c_s_n.boundary("right", domain=self.r_n)
        c_surf_p = self.c_s_p.boundary("right", domain=self.r_p)

        # Simplified OCV mappings
        U_n = 0.1 - 0.0001 * c_surf_n
        U_p = 4.2 - 0.0001 * c_surf_p

        # Assuming Faraday's flux conversion parameterization
        j_flux = self.i_app / 96485.0

        return {
            "regions": {
                self.r_n: [ fx.dt(self.c_s_n) == -fx.div(flux_n, axis=self.r_n) ],
                self.r_p: [ fx.dt(self.c_s_p) == -fx.div(flux_p, axis=self.r_p) ]
            },
            "boundaries": [
                flux_n.boundary("left", domain=self.r_n) == 0.0,
                flux_n.boundary("right", domain=self.r_n) == -j_flux,
                
                flux_p.boundary("left", domain=self.r_p) == 0.0,
                flux_p.boundary("right", domain=self.r_p) == j_flux
            ],
            "global": [
                self.V_cell == (U_p - U_n) - 0.02 * self.i_app,
                
                self.c_s_n.t0 == 800.0,
                self.c_s_p.t0 == 200.0,
                self.V_cell.t0 == 4.18,
                self.i_app.t0 == 0.0
            ]
        }

if __name__ == "__main__":
    engine = fx.Engine(model=SingleParticleModel(), target="cpu:serial")
    
    protocol = Sequence([
        CC(rate=10.0, until=fx.Condition("V_cell <= 3.0"), time=3600),
        Rest(time=600)
    ])
    
    print("Executing Single Particle Model protocol...")
    res = engine.solve(protocol=protocol)
    print(f"Simulation Complete. Final Voltage: {res['V_cell'].data[-1]:.3f} V")

    print("Launching Dashboard.")
    res.plot_dashboard()