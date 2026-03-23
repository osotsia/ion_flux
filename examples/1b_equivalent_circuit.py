import ion_flux as fx
from ion_flux.protocols import Sequence, CC

class ThermoCoupledECM(fx.PDE):
    """
    0D Equivalent Circuit Model with lumped thermal dynamics and entropic heating.
    Directly replicates the physics embedded in PyBaMM's Thevenin() model.
    """
    soc = fx.State(name="soc")
    v_rc = fx.State(name="v_rc")
    T_cell = fx.State(name="T_cell")
    T_jig = fx.State(name="T_jig")
    V_cell = fx.State(domain=None, name="V_cell")
    i_app = fx.State(domain=None, name="i_app")

    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    # Electrical Parameters (Mapped to a 100Ah cell)
    Q = fx.Parameter(default=100.0)      # Capacity [Ah]
    R0 = fx.Parameter(default=0.001)     # Ohmic resistance [Ohm]
    R1 = fx.Parameter(default=0.001)     # RC resistance [Ohm]
    tau = fx.Parameter(default=100.0)    # RC time constant [s]

    # Thermal Parameters
    cth_cell = fx.Parameter(default=3500.0)   # Cell heat capacity [J/K]
    cth_jig = fx.Parameter(default=15000.0)   # Jig heat capacity [J/K]
    k_cell_jig = fx.Parameter(default=2.5)    # Convection coefficient (Cell to Jig) [W/K]
    k_jig_air = fx.Parameter(default=1.5)     # Convection coefficient (Jig to Ambient) [W/K]
    T_amb = fx.Parameter(default=25.0)        # Ambient temperature [degC]

    def math(self):
        # Empirical OCV and Entropic Change (dU/dT) approximations
        ocv = 3.4 + 0.6 * self.soc - 0.1 * fx.exp(-30.0 * self.soc)
        dUdT = -0.00015 + 0.001 * fx.exp(-20.0 * self.soc)
        
        # Heat Generation
        Q_irr = (self.i_app ** 2) * self.R0 + self.i_app * self.v_rc
        Q_rev = -self.i_app * (self.T_cell + 273.15) * dUdT
        
        # Newton Cooling
        Q_cell_cool = -self.k_cell_jig * (self.T_cell - self.T_jig)
        Q_jig_cool = -self.k_jig_air * (self.T_jig - self.T_amb)

        return {
            "global": [
                # Core electrical ODEs
                fx.dt(self.soc) == -self.i_app / (self.Q * 3600.0),
                fx.dt(self.v_rc) == (self.i_app * self.R1 - self.v_rc) / self.tau,
                
                # Thermal ODEs
                fx.dt(self.T_cell) == (Q_irr + Q_rev + Q_cell_cool) / self.cth_cell,
                fx.dt(self.T_jig) == (Q_jig_cool - Q_cell_cool) / self.cth_jig,
                
                # Algebraic Terminal Constraint
                self.V_cell == ocv - self.v_rc - self.i_app * self.R0,
                
                # Initial Conditions (matching PyBaMM simulation states)
                self.soc.t0 == 0.5,
                self.v_rc.t0 == 0.0,
                self.T_cell.t0 == 25.0,
                self.T_jig.t0 == 25.0,
                self.V_cell.t0 == 3.65,
                self.i_app.t0 == 0.0
            ]
        }

if __name__ == "__main__":
    engine = fx.Engine(model=ThermoCoupledECM(), target="cpu:serial")
    
    # Matching PyBaMM's 100A discharge profile
    protocol = Sequence([
        CC(rate=100.0, until=fx.Condition("V_cell <= 3.2"), time=3600)
    ])
    
    res = engine.solve(protocol=protocol)
    
    # Isolate variables for the dashboard to mirror the PyBaMM layout
    variables_to_plot = [
        "i_app",
        ["V_cell"], # OCV can be tracked natively if added as an output state
        "soc",
        ["T_cell", "T_jig"]
    ]
    res.plot_dashboard(variables=variables_to_plot)