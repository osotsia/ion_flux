import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class EquivalentCircuit(fx.PDE):
    """
    0D Equivalent Circuit Model (Thevenin-like).
    Validates pure ODEs and 0D algebraic constraints without spatial topology.
    Replaces PyBaMM's pybamm.equivalent_circuit.Thevenin().
    """
    soc = fx.State(name="soc")
    v_rc = fx.State(name="v_rc")
    V_cell = fx.State(name="V_cell")
    i_app = fx.State(name="i_app")

    terminal = fx.Terminal(current=i_app, voltage=V_cell)

    Q = fx.Parameter(default=5.0)       # Cell capacity [Ah]
    R0 = fx.Parameter(default=0.015)    # Series resistance [Ohm]
    R1 = fx.Parameter(default=0.01)     # RC resistance [Ohm]
    tau = fx.Parameter(default=100.0)   # RC time constant [s]

    def math(self):
        # Linear approximation of OCV for simplicity
        ocv = 3.0 + 1.2 * self.soc  
        
        return {
            "global": [
                # ODEs
                fx.dt(self.soc) == -self.i_app / (self.Q * 3600.0),
                fx.dt(self.v_rc) == (self.i_app * self.R1 - self.v_rc) / self.tau,
                
                # Algebraic Voltage Constraint (DAE)
                self.V_cell == ocv - self.v_rc - self.i_app * self.R0,
                
                # Initial Conditions
                self.soc.t0 == 1.0,
                self.v_rc.t0 == 0.0,
                self.V_cell.t0 == 4.2,
                self.i_app.t0 == 0.0
            ]
        }

if __name__ == "__main__":

    model=EquivalentCircuit()
    engine = fx.Engine(model, target="cpu:serial", solver_backend="native")
    
    protocol = Sequence([
        CC(rate=5.0, until=model.V_cell <= 3.2, time=3600),
        Rest(time=600)
    ])
    
    print("Executing Equivalent Circuit protocol...")
    res = engine.solve(protocol=protocol)
    print(f"Final Voltage: {res['V_cell'].data[-1]:.3f} V")

    print("Launching Dashboard.")
    res.plot_dashboard()