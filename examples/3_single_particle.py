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
            "equations": {
                # --- Solid Phase PDEs ---
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(flux_n, axis=self.r_n),
                self.c_s_p: fx.dt(self.c_s_p) == -fx.div(flux_p, axis=self.r_p),
                
                # --- Global Algebraic ---
                self.V_cell: self.V_cell == (U_p - U_n) - 0.02 * self.i_app
            },
            "boundaries": {
                flux_n: {"left": 0.0, "right": -j_flux},
                flux_p: {"left": 0.0, "right": j_flux}
            },
            "initial_conditions": {
                self.c_s_n: 800.0,
                self.c_s_p: 200.0,
                self.V_cell: 4.18,
                self.i_app: 0.0
            }
        }

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    model=SingleParticleModel()
    engine = fx.Engine(model, target="cpu:serial")
    
    protocol = Sequence([
        CC(rate=10.0, until=model.V_cell <= 3.0, time=3600),
        Rest(time=600)
    ])
    
    print("Executing Single Particle Model protocol...")
    res = engine.solve(protocol=protocol)
    print(f"Simulation Complete. Final Voltage: {res['V_cell'].data[-1]:.3f} V")

    t_hours = res["Time [s]"].data / 3600.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Lumped Single Particle Model (10A Discharge)", fontsize=14, fontweight="bold")
    
    ax1.plot(t_hours, res["V_cell"].data, linewidth=2, color="tab:blue")
    ax1.set(xlabel="Time [h]", ylabel="Voltage [V]", title="Terminal Voltage")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Extract the internal solid concentrations at the end of the discharge step
    discharge_end_idx = np.searchsorted(t_hours, 1.0) - 1 # Roughly 1 hour mark
    r_n = np.linspace(0, 10, 15)
    r_p = np.linspace(0, 10, 15)
    
    ax2.plot(r_n, res["c_s_n"].data[discharge_end_idx], label="Anode $c_s$", color="tab:orange", linewidth=2)
    ax2.plot(r_p, res["c_s_p"].data[discharge_end_idx], label="Cathode $c_s$", color="tab:green", linewidth=2)
    ax2.set(xlabel="Radius [µm] (0 = Core)", ylabel="Concentration [mol/m³]", title="End of Discharge Concentrations")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best")
    
    plt.tight_layout()
    plt.show()