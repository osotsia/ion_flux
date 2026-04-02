#### Step A: Define Reusable Submodels
import ion_flux as fx
from ion_flux.protocols import Sequence, CC, Rest

class FickianParticle(fx.PDE):
    """A highly reusable submodel for solid diffusion."""
    r = fx.Domain(bounds=(0, 5e-6), resolution=15, coord_sys="spherical")
    c_s = fx.State(domain=r)
    D_s = fx.Parameter(default=1e-14)
    
    # The Interface Contract: This submodel NEEDS an external flux to function.
    def math(self, j_flux: fx.Node):
        flux = -self.D_s * fx.grad(self.c_s, axis=self.r)
        
        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(flux, axis=self.r)
            },
            "boundaries": {
                flux: {"left": 0.0, "right": j_flux}
            },
            "initial_conditions": {}
        }

#### Step B: Compose the Full Model
class ModularSPM(fx.PDE):
    """Composes the full cell using instantiated submodels."""
    
    # 1. Instantiate submodels as attributes
    # The framework will automatically namespace their states (e.g., 'neg_particle_c_s')
    neg_particle = FickianParticle()
    pos_particle = FickianParticle()
    
    # 2. Define Macro states
    V_cell = fx.State(domain=None)
    i_app = fx.State(domain=None)
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        # 3. Define the coupling physics (Faraday's Law)
        j_n = self.i_app / 96485.0
        j_p = -self.i_app / 96485.0
        
        # 4. Extract surface concentrations from the submodels for the OCV calculation
        c_surf_n = self.neg_particle.c_s.right
        c_surf_p = self.pos_particle.c_s.right
        
        macro_physics = {
            "equations": {
                self.V_cell: self.V_cell == (4.2 - 0.001 * c_surf_p) - (0.1 - 0.001 * c_surf_n)
            },
            "boundaries": {},
            "initial_conditions": {
                self.V_cell: 4.1,
                self.i_app: 0.0,
                # Initial conditions for submodels can be targeted explicitly
                self.neg_particle.c_s: 800.0,
                self.pos_particle.c_s: 200.0
            }
        }
        
        # 5. Merge all ASTs into a single seamless dictionary
        return fx.merge(
            macro_physics,
            self.neg_particle.math(j_flux=j_n),
            self.pos_particle.math(j_flux=j_p)
        )
    
if __name__ == "__main__":
    # Standard compilation and execution payload
    model=ModularSPM()
    engine = fx.Engine(model, target="cpu:serial")
        
    protocol = Sequence([
        CC(rate=1.0, until=model.V_cell <= -5, time=3600),
        Rest(time=600)
    ])

    res = engine.solve(protocol=protocol)
    res.plot_dashboard(variables=["V_cell"])