"""
Compiler Exactness Oracle

This suite proves that the AST-to-C++ compiler perfectly respects spherical 
coordinate geometry, spatial scaling, and boundary extractions without any 
off-by-one errors or volume distortions.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine

class SphericalFVMOracle(fx.PDE):
    """
    Manufactured Analytical Solution for Spherical Diffusion.
    If c(r, t) = r^2 + 6*t
    Then:
    dt(c) = 6.0
    grad(c) = 2*r
    
    The divergence in spherical coordinates is:
    div(grad(c)) = (1/r^2) * d/dr( r^2 * 2r ) = (1/r^2) * d/dr( 2r^3 ) = 6.0
    
    Therefore, dt(c) == div(grad(c)) is an EXACT mathematical truth.
    If the compiler mistakenly treats the sphere as a 1D slab (Cartesian), 
    div(grad(c)) would equal 2.0, and the residual would fail massively.
    """
    r = fx.Domain(bounds=(0, 1.0), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    
    # 0D target to test the boundary extraction AST logic
    surface_val = fx.State(domain=None, name="surface_val")

    def math(self):
        flux = -fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                # dt(c) - div(grad(c)) = 0 -> dt(c) - (-div(flux)) = 0
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r),
                
                # Check if boundary extraction grabs the exact edge node correctly
                self.surface_val: self.surface_val == self.c.boundary("right", domain=self.r)
            },
            "boundaries": {
                # grad(r^2) at r=1.0 is 2.0. So flux = -2.0.
                flux: {"left": 0.0, "right": -2.0}
            },
            "initial_conditions": {
                # Initialize c(r, 0) = r^2
                self.c: self.r.coords ** 2,
                self.surface_val: 0.0
            }
        }

def test_spherical_fvm_and_boundary_extraction_exactness():
    engine = Engine(model=SphericalFVMOracle(), target="cpu", mock_execution=False)
    
    # We want to check the instantaneous residual at t=0.
    y0, ydot0, _, _, _ = engine._extract_metadata()
    y0 = np.array(y0)
    ydot0 = np.zeros_like(y0)
    
    # Because c(r,t) = r^2 + 6t, the true derivative dt(c) MUST be 6.0 everywhere.
    off_c, size_c = engine.layout.state_offsets["c"]
    ydot0[off_c : off_c + size_c] = 6.0
    
    res = engine.evaluate_residual(y0.tolist(), ydot0.tolist(), parameters={})
    
    # 1. Check Spherical FVM Divergence
    # Residual F = ydot - rhs. If the compiler is perfectly exact, F == 0.0
    c_residuals = res[off_c : off_c + size_c]
    np.testing.assert_allclose(
        c_residuals, 0.0, atol=1e-12, 
        err_msg="COMPILER BUG: Spherical FVM divergence failed! The compiler is calculating the wrong cell volumes/areas."
    )
    
    # 2. Check Boundary Node Extraction
    # The true analytical value of c at the right boundary (r=1.0) is 1.0^2 = 1.0.
    off_surf, _ = engine.layout.state_offsets["surface_val"]
    
    # Residual of the algebraic observer eq: F = 0.0 - surface_val_extracted. 
    # Therefore, the extracted value is -F.
    extracted_surface = -res[off_surf]
    
    assert extracted_surface == pytest.approx(1.0, rel=1e-12), \
        "COMPILER BUG: Boundary extraction `c.boundary('right')` grabbed the wrong node or failed to evaluate!"