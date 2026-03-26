"""
Middle-End Codegen: ALE Moving Meshes

Validates the injection of Arbitrary Lagrangian-Eulerian (ALE) advection velocity
terms for moving boundaries (Stefan problems) and traps grid inversion conditions.
"""

import pytest
import ion_flux as fx
from ion_flux.runtime.engine import Engine
import re

class HollowParticlePDE(fx.PDE):
    """
    Isolates Spherical Origin Singularity Logic.
    Models a hollow core particle where the inner radius > 0.
    """
    # Note the bounds: domain starts at 5 microns, NOT 0.
    r = fx.Domain(bounds=(5e-6, 10e-6), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    
    def math(self):
        return {
            "regions": {
                self.r: [ fx.dt(self.c) == fx.div(fx.grad(self.c)) ]
            },
            "boundaries": [
                self.c.left == 1.0, self.c.right == 0.0
            ]
        }

class ALEAdvectionPDE(fx.PDE):
    """
    Isolates Advective Instability in Moving Meshes (Stefan Problems).
    """
    x = fx.Domain(bounds=(0, 1), resolution=10, name="x")
    c = fx.State(domain=x, name="c")
    thickness = fx.State(domain=None, name="thickness")
    
    def math(self):
        return {
            "regions": {
                self.x: [ fx.dt(self.c) == fx.grad(self.c) ]
            },
            "boundaries": [
                self.x.right == self.thickness,  # Triggers ALE grid velocity injection
                self.c.left == 0.0, self.c.right == 0.0
            ],
            "global": [
                fx.dt(self.thickness) == 1.0, 
                self.thickness.t0 == 1.0
            ]
        }

def test_hollow_particle_avoids_origin_singularity_logic():
    """
    X-Ray for Spherical L'Hopital constraints.
    Proves that a spherical domain bounding away from r=0 does not 
    blindly apply the 3*flux/dr symmetry condition at the first node.
    """
    model = HollowParticlePDE()
    engine = Engine(model=model, target="cpu", mock_execution=True)
    cpp = engine.cpp_source
    
    assert "((i) == 0 ? (3.0 *" not in cpp, (
        "Numerical Singularity Flaw: Compiler injected L'Hopital's symmetry "
        "condition at the left boundary of a hollow particle (r > 0)."
    )

def test_ale_advection_upwinding_stencil():
    """
    X-Ray for Advective Instability (Spurious Oscillations).
    Proves that the ALE grid velocity term uses a stable upwind scheme
    rather than a centered difference scheme.
    """
    model = ALEAdvectionPDE()
    engine = Engine(model=model, target="cpu", mock_execution=True)
    cpp = engine.cpp_source

    # Isolate the line compiling the ALE moving mesh term
    ale_term_segments = [line for line in cpp.split("\n") if "std::max(1e-12" in line]
    assert len(ale_term_segments) > 0, "Verify baseline: ALE advection term was injected."
    
    ale_line = ale_term_segments[0]

    # Verify that the advection component relies on velocity-directed upwinding.
    # An upwind stencil emits a ternary velocity check: (v > 0.0 ? ... : ...)
    assert "> 0.0 ?" in ale_line, "ALE advection missing velocity direction conditional."
    
    # Extract the true/false branches of the velocity conditional to ensure neither
    # branch accidentally divides by a centered difference denominator (2.0 * dx).
    # This prevents false positives if a diffusion term exists elsewhere on the line.
    upwind_match = re.search(r'> 0\.0 \?([^:]+):([^)]+)', ale_line)
    assert upwind_match is not None, "Failed to parse the upwind ternary operator branches."
    
    backward_diff_branch, forward_diff_branch = upwind_match.groups()
    
    assert "2.0 *" not in backward_diff_branch and "2.0 *" not in forward_diff_branch, (
        "Numerical Instability Flaw: ALE advection uses a centered-difference "
        "stencil within its directional branches, guaranteeing spurious oscillations."
    )