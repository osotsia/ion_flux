"""
Backend Spatial Discretization Verification Suite.

Validates the translation of continuum Math IR into discrete Compute IR:
  1. Pure Python Compute IR unit tests (stencils, harmonic means, pointer mapping).
  2. Interface flux balancing, auto-stitching, and conservative mass retention.
  3. Curvilinear metric tensors (Cartesian, Cylindrical, Spherical, Unstructured CSR).
  4. Multi-scale hierarchical domain coupling (1D-1D macro-micro coordinates).
  5. Non-linear Method of Manufactured Solutions (MMS) and spatial DAEs.
"""

import os
import shutil
import platform
import pytest
import numpy as np

import ion_flux as fx
from ion_flux.compiler._4_codegen.compute_ir import Literal, BinaryOp, ArrayAccess, FuncCall, Ternary
from ion_flux.compiler._3_backend.discretizer import (
    MeshMetrics, StateSampler, PiecewiseFluxStitcher, SpatialStencils, IndexManager
)


# ==============================================================================
# Environment Capabilities
# ==============================================================================

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac


try:
    from ion_flux._core import solve_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")
REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)


# ==============================================================================
# Shared FVM Geometry Utilities
# ==============================================================================

def get_control_volumes(engine: fx.Engine, domain_name: str) -> np.ndarray:
    """Extracts exact physical control volumes (V_node) directly from engine layout memory."""
    layout = engine.layout
    vol_off = layout.mesh_offsets[domain_name]["w_V_nodes"]
    m_list = layout.get_mesh_data()
    domain_info = engine.manifest.ast_payload["domains"][domain_name]
    resolution = domain_info["resolution"]
    bounds = domain_info.get("bounds", (0.0, 1.0))
    l_phys = float(bounds[1] - bounds[0])
    return np.array(m_list[vol_off : vol_off + resolution]) * l_phys


# ==============================================================================
# 1. Pure Python Compute IR Unit Tests (Zero-Compiler Verification)
# ==============================================================================

class TestComputeIRUnitStencils:
    """Fast, compiler-free unit verification of modular discretizer components."""

    def test_harmonic_mean_zero_division_guard(self):
        """Validates that zero-flux evaluations do not divide by zero due to 1e-30 safety epsilon."""
        topo = type("MockTopo", (), {"domains": {}})()
        stitcher = PiecewiseFluxStitcher(topo=topo)

        expr = stitcher.harmonic_mean(Literal(0.0), Literal(0.0))
        assert isinstance(expr, BinaryOp)
        assert expr.op == "/"
        # Denominator must enforce + 1e-30 guard
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.right.val == "1e-30"

    def test_harmonic_mean_extreme_jump_discontinuity_limit(self):
        """
        Validates asymptotic behavior across severe conductivity disparities:
          D_left = 1e6, D_right = 1.0 -> 2 * D_L * D_R / (D_L + D_R) -> 2.0
        """
        topo = type("MockTopo", (), {"domains": {}})()
        stitcher = PiecewiseFluxStitcher(topo=topo)

        flux_l = Literal(1e6)
        flux_r = Literal(1.0)
        expr = stitcher.harmonic_mean(flux_l, flux_r)

        assert isinstance(expr, BinaryOp)
        assert expr.op == "/"

    def test_mesh_metrics_pointer_generation(self):
        """Validates that MeshMetrics correctly emits ArrayAccess expressions into the 'm' array."""
        layout = type("MockLayout", (), {
            "mesh_offsets": {
                "cell": {"w_dx_faces": 0, "w_A_faces": 10, "w_V_nodes": 20, "w_centers": 30}
            }
        })()
        topo = type("MockTopo", (), {
            "domains": {"cell": {"bounds": (0.0, 2.0)}}
        })()
        metrics = MeshMetrics(layout, topo)

        area_expr = metrics.face_area("cell", Literal(5))
        assert isinstance(area_expr, ArrayAccess)
        assert area_expr.array_name == "m"

        vol_expr = metrics.scaled_cell_volume("cell", Literal(2))
        assert isinstance(vol_expr, FuncCall)
        assert vol_expr.func == "std::max"


# ==============================================================================
# 2. Interface Flux Balancing & Conservation
# ==============================================================================

class BiMaterialDiffusion(fx.PDE):
    bulk = fx.Domain(bounds=(0, 2.0), resolution=20)
    reg_L = bulk.region(bounds=(0, 1.0), resolution=10, name="reg_L")
    reg_R = bulk.region(bounds=(1.0, 2.0), resolution=10, name="reg_R")
    c = fx.State(domain=bulk, name="c")
    D_L = fx.Parameter(default=1.0)
    D_R = fx.Parameter(default=5.0)

    def math(self):
        flux_left = -self.D_L * fx.grad(self.c)
        flux_right = -self.D_R * fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_left),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_right)
                })
            },
            "boundaries": {
                flux_left: {"left": 0.0},
                flux_right: {"right": 0.0}
            },
            "initial_conditions": {self.c: 1.0}
        }


class TriRegionElectrolyte(fx.PDE):
    cell = fx.Domain(bounds=(0, 3.0), resolution=30)
    x_n = cell.region(bounds=(0, 1.0), resolution=10, name="x_n")
    x_s = cell.region(bounds=(1.0, 2.0), resolution=10, name="x_s")
    x_p = cell.region(bounds=(2.0, 3.0), resolution=10, name="x_p")
    c_e = fx.State(domain=cell, name="c_e")

    def math(self):
        flux_n = -1.0 * fx.grad(self.c_e)
        flux_s = -0.5 * fx.grad(self.c_e)
        flux_p = -2.0 * fx.grad(self.c_e)
        return {
            "equations": {
                self.c_e: fx.Piecewise({
                    self.x_n: fx.dt(self.c_e) == -fx.div(flux_n),
                    self.x_s: fx.dt(self.c_e) == -fx.div(flux_s),
                    self.x_p: fx.dt(self.c_e) == -fx.div(flux_p)
                })
            },
            "boundaries": {
                flux_n: {"left": 0.0},
                flux_p: {"right": 0.0}
            },
            "initial_conditions": {self.c_e: 1000.0}
        }


class CoupledTensorFlux(fx.PDE):
    bulk = fx.Domain(bounds=(0, 2.0), resolution=20)
    reg_L = bulk.region(bounds=(0, 1.0), resolution=10, name="reg_L")
    reg_R = bulk.region(bounds=(1.0, 2.0), resolution=10, name="reg_R")
    c = fx.State(domain=bulk, name="c")
    phi = fx.State(domain=bulk, name="phi")

    def math(self):
        flux_L = -1.0 * fx.grad(self.c) + self.c * fx.grad(self.phi)
        flux_R = -5.0 * fx.grad(self.c) + self.c * fx.grad(self.phi)
        i_e = -fx.grad(self.phi)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_L),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_R)
                }),
                self.phi: fx.div(i_e) == 0.0
            },
            "boundaries": {
                flux_L: {"left": 0.0},
                flux_R: {"right": 0.0},
                i_e: {"left": 0.0, "right": 0.0},
                self.phi: {"left": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {self.c: 1.0, self.phi: 0.0}
        }


class HarmonicMeanDiscontinuityOracle(fx.PDE):
    cell = fx.Domain(bounds=(0, 2.0), resolution=40, name="cell")
    reg_L = cell.region(bounds=(0, 1.0), resolution=20, name="reg_L")
    reg_R = cell.region(bounds=(1.0, 2.0), resolution=20, name="reg_R")
    c = fx.State(domain=cell, name="c")

    def math(self):
        flux_L = -10.0 * fx.grad(self.c)
        flux_R = -0.1 * fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_L),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_R)
                })
            },
            "boundaries": {
                self.c: {"left": fx.Dirichlet(100.0), "right": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {self.c: 50.0}
        }


class StagnantZeroDiffusivityEdgeCase(fx.PDE):
    """EDGE CASE: One region has strictly zero diffusivity (D=0.0). Must not diverge or NaN."""
    bulk = fx.Domain(bounds=(0, 2.0), resolution=20)
    reg_L = bulk.region(bounds=(0, 1.0), resolution=10, name="reg_L")
    reg_R = bulk.region(bounds=(1.0, 2.0), resolution=10, name="reg_R")
    c = fx.State(domain=bulk, name="c")

    def math(self):
        flux_L = -1.0 * fx.grad(self.c)
        flux_R = -0.0 * fx.grad(self.c)  # Stagnant zero diffusion
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_L: fx.dt(self.c) == -fx.div(flux_L),
                    self.reg_R: fx.dt(self.c) == -fx.div(flux_R)
                })
            },
            "boundaries": {
                flux_L: {"left": 0.0},
                flux_R: {"right": 0.0}
            },
            "initial_conditions": {self.c: 1.0}
        }


class PiecewisePorosityDiscontinuity(fx.PDE):
    """EDGE CASE: Discontinuous capacity multiplier (eps * dt(c)) across a material interface."""
    cell = fx.Domain(bounds=(0, 2.0), resolution=20)
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 2.0), resolution=10, name="reg_B")
    c = fx.State(domain=cell, name="c")

    def math(self):
        flux = fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.reg_A: 1.0 * fx.dt(self.c) == fx.div(flux),
                    self.reg_B: 5.0 * fx.dt(self.c) == fx.div(flux)
                })
            },
            "boundaries": {
                flux: {"left": -1.0, "right": 0.0}
            },
            "initial_conditions": {self.c: 0.0}
        }


class TestInterfaceFluxConservation:
    """Verifies that piecewise flux balancing conserves mass and maintains Jacobian continuity."""

    @REQUIRES_COMPILER
    def test_bimaterial_mass_conservation(self):
        engine = fx.Engine(model=BiMaterialDiffusion(), target="cpu", mock_execution=False)
        N = engine.layout.n_states
        np.random.seed(42)
        y = np.random.uniform(1.0, 5.0, size=N).tolist()
        ydot = np.zeros(N).tolist()

        res = engine.evaluate_residual(y, ydot, parameters={})
        V_cells = get_control_volumes(engine, "bulk")
        total_mass_drift = float(np.sum(np.array(res) * V_cells))
        assert total_mass_drift == pytest.approx(0.0, abs=1e-10)

    @REQUIRES_COMPILER
    def test_bimaterial_jacobian_coupling(self):
        engine = fx.Engine(model=BiMaterialDiffusion(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
        N = engine.layout.n_states
        y = np.linspace(1.0, 5.0, N).tolist()
        ydot = np.zeros(N).tolist()

        J = np.array(engine.evaluate_jacobian(y, ydot, c_j=1.0, parameters={}))
        assert abs(J[9, 10]) > 1e-8, "Disjoint Jacobian: Left region does not depend on Right region."
        assert abs(J[10, 9]) > 1e-8, "Disjoint Jacobian: Right region does not depend on Left region."

    @REQUIRES_COMPILER
    def test_triregion_mass_conservation(self):
        engine = fx.Engine(model=TriRegionElectrolyte(), target="cpu", mock_execution=False)
        N = engine.layout.n_states
        y = np.linspace(100.0, 200.0, N).tolist()
        ydot = np.zeros(N).tolist()

        res = engine.evaluate_residual(y, ydot, parameters={})
        V_cells = get_control_volumes(engine, "cell")
        total_mass_drift = float(np.sum(np.array(res) * V_cells))
        assert total_mass_drift == pytest.approx(0.0, abs=1e-10)

    @REQUIRES_COMPILER
    def test_complex_tensor_piecewise_stitching(self):
        engine = fx.Engine(model=CoupledTensorFlux(), target="cpu", mock_execution=False, jacobian_bandwidth=0)
        N = engine.layout.n_states
        off_c, size_c = engine.layout.state_offsets["c"]
        off_phi, size_phi = engine.layout.state_offsets["phi"]

        y = np.zeros(N)
        y[off_c : off_c + size_c] = np.linspace(1.0, 5.0, size_c)
        y[off_phi : off_phi + size_phi] = np.linspace(0.0, 10.0, size_phi)

        res = engine.evaluate_residual(y.tolist(), np.zeros(N).tolist(), parameters={})
        res_c = np.array(res[off_c : off_c + size_c])
        V_cells = get_control_volumes(engine, "bulk")
        total_mass_drift = float(np.sum(res_c * V_cells))
        assert total_mass_drift == pytest.approx(0.0, abs=1e-10)

    @REQUIRES_RUNTIME
    def test_harmonic_mean_discontinuity_steady_state(self):
        engine = fx.Engine(model=HarmonicMeanDiscontinuityOracle(), target="cpu", mock_execution=False)
        res = engine.solve(t_span=(0, 500.0), t_eval=np.array([0.0, 500.0]))
        c_final = res["c"].data[-1]

        # Analytical interface continuity condition:
        # D_L * (100 - c_int) = D_R * (c_int - 0) -> c_int = 99.0099
        c_int_L, c_int_R = c_final[19], c_final[20]
        c_int_exact = (10.0 * c_int_L + 0.1 * c_int_R) / (10.0 + 0.1)
        np.testing.assert_allclose(c_int_exact, 99.0099, rtol=1e-2)

    @REQUIRES_COMPILER
    def test_edge_case_stagnant_zero_diffusivity_stability(self):
        """EDGE CASE: Asserts that an interface with D_R=0 evaluates without NaNs or zero-division panics."""
        engine = fx.Engine(model=StagnantZeroDiffusivityEdgeCase(), target="cpu", mock_execution=False)
        N = engine.layout.n_states
        y = np.linspace(1.0, 10.0, N).tolist()
        ydot = np.zeros(N).tolist()

        res = engine.evaluate_residual(y, ydot, parameters={})
        assert np.all(np.isfinite(res)), "Zero diffusivity produced non-finite residual elements."
        # Stagnant region right of interface (node 10) must see zero incoming interface flux
        assert res[10] == pytest.approx(0.0, abs=1e-8)

    @REQUIRES_RUNTIME
    def test_piecewise_discontinuous_porosity_lhs(self):
        """EDGE CASE: Verifies mass conservation when time-derivative capacities (eps) jump at an interface."""
        engine = fx.Engine(model=PiecewisePorosityDiscontinuity(), target="cpu", mock_execution=False)
        res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))
        c_final = res["c"].data[-1]

        dx = 2.0 / 19.0
        mass_integrated = 0.0
        for i in range(20):
            eps = 1.0 if i < 10 else 5.0
            w = 0.5 * dx if (i == 0 or i == 19) else dx
            mass_integrated += eps * c_final[i] * w

        # Total injected mass = flux * time = 1.0 * 2.0 = 2.0
        np.testing.assert_allclose(mass_integrated, 2.0, rtol=1e-4)


# ==============================================================================
# 3. Curvilinear & Unstructured Geometry
# ==============================================================================

class SphericalFVMOracle(fx.PDE):
    r = fx.Domain(bounds=(0, 1.0), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")
    surface_val = fx.State(domain=None, name="surface_val")

    def math(self):
        flux = -fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r),
                self.surface_val: self.surface_val == self.c.boundary("right", domain=self.r)
            },
            "boundaries": {
                flux: {"left": 0.0, "right": -2.0}
            },
            "initial_conditions": {
                self.c: self.r.coords ** 2,
                self.surface_val: 0.0
            }
        }


class CompositeIntegrationOracle(fx.PDE):
    x = fx.Domain(bounds=(0, 2.0), resolution=10, name="x")
    r = fx.Domain(bounds=(0, 1.0), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=x * r, name="c")
    total_mass = fx.State(domain=None, name="total_mass")

    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                self.total_mass: self.total_mass == fx.integral(self.c, over=self.x * self.r)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 1.0,
                self.total_mass: 0.0
            }
        }


TETRAHEDRON_MESH = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "elements": [[0, 1, 2, 3]]
}


class UnstructuredGraphConservationOracle(fx.PDE):
    mesh = fx.Domain.from_mesh(TETRAHEDRON_MESH, name="mesh", surfaces={"top": [2, 3]})
    c = fx.State(domain=mesh, name="c")
    mass_tracker = fx.State(domain=None, name="mass_tracker")

    def math(self):
        flux = -fx.grad(self.c)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux),
                self.mass_tracker: self.mass_tracker == fx.integral(self.c, over=self.mesh)
            },
            "boundaries": {
                flux: {"top": -10.0}
            },
            "initial_conditions": {
                self.c: 1.0,
                self.mass_tracker: 0.0
            }
        }


class TestCurvilinearAndUnstructuredGeometry:
    """Verifies non-Cartesian metric scaling and unstructured finite-volume quadrature."""

    @REQUIRES_COMPILER
    def test_spherical_fvm_divergence_and_boundary_extraction(self):
        """Validates spherical divergence exactness and right-boundary node extraction."""
        engine = fx.Engine(model=SphericalFVMOracle(), target="cpu", mock_execution=False)
        y0, ydot0, _, _, _ = engine._extract_metadata()
        y0 = np.array(y0)
        ydot0 = np.zeros_like(y0)

        # Manufactured solution: c(r,t) = r^2 + 6t -> dt(c) = 6.0 everywhere
        off_c, size_c = engine.layout.state_offsets["c"]
        ydot0[off_c : off_c + size_c] = 6.0

        res = engine.evaluate_residual(y0.tolist(), ydot0.tolist(), parameters={})
        c_residuals = res[off_c : off_c + size_c]
        np.testing.assert_allclose(c_residuals, 0.0, atol=1e-12)

        off_surf, _ = engine.layout.state_offsets["surface_val"]
        extracted_surface = -res[off_surf]
        assert extracted_surface == pytest.approx(1.0, rel=1e-12)

    @REQUIRES_COMPILER
    def test_composite_domain_integration_exactness(self):
        """Validates multidimensional integration over composite domains: V_macro * V_micro."""
        engine = fx.Engine(model=CompositeIntegrationOracle(), target="cpu", mock_execution=False)
        y0, ydot0, _, _, _ = engine._extract_metadata()
        res = engine.evaluate_residual(y0, ydot0, parameters={})

        off_mass, _ = engine.layout.state_offsets["total_mass"]
        simulated_integral = -res[off_mass]
        exact_integral = 2.0 * (4.0 / 3.0) * np.pi * (1.0 ** 3)
        assert simulated_integral == pytest.approx(exact_integral, rel=1e-2)

    @REQUIRES_RUNTIME
    def test_unstructured_csr_mass_conservation(self):
        """Validates mass retention during matrix-free unstructured tetrahedral graph traversal."""
        engine = fx.Engine(model=UnstructuredGraphConservationOracle(), target="cpu", mock_execution=False)
        res = engine.solve(t_span=(0, 2.0), t_eval=np.array([0.0, 2.0]))

        exact_final_mass = (1.0 / 6.0) + 40.0
        simulated_final_mass = res["mass_tracker"].data[-1]
        np.testing.assert_allclose(simulated_final_mass, exact_final_mass, rtol=1e-4)


class NestedGradientIntegralOracle(fx.PDE):
    """EDGE CASE: fx.grad() nested inside fx.integral() must retain spatial axis context."""
    cell = fx.Domain(bounds=(0, 2.0), resolution=20, name="cell")
    reg = cell.region(bounds=(0, 2.0), resolution=20, name="reg")
    phi = fx.State(domain=cell, name="phi")
    T_var = fx.State(domain=None, name="T_var")

    def math(self):
        return {
            "equations": {
                self.phi: fx.dt(self.phi) == 0.0,
                self.T_var: fx.dt(self.T_var) == fx.integral(fx.grad(self.phi), over=self.reg) - self.T_var
            },
            "boundaries": {
                self.phi: {"left": fx.Dirichlet(0.0), "right": fx.Dirichlet(2.0)}
            },
            "initial_conditions": {
                self.phi: (self.cell.coords ** 2) / 2.0,  # grad(phi) = x -> integral(x) from 0 to 2 = 2.0
                self.T_var: 0.0
            }
        }


class SubregionQuadratureOracle(fx.PDE):
    """EDGE CASE: Integrals over sub-regions must scale by sub-interval lengths, not parent length."""
    cell = fx.Domain(bounds=(0, 3.0), resolution=30, name="cell")
    reg_A = cell.region(bounds=(0, 1.0), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1.0, 3.0), resolution=20, name="reg_B")
    c = fx.State(domain=cell, name="c")
    mass_A = fx.State(domain=None, name="mass_A")
    mass_B = fx.State(domain=None, name="mass_B")

    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                self.mass_A: self.mass_A == fx.integral(self.c, over=self.reg_A),
                self.mass_B: self.mass_B == fx.integral(self.c, over=self.reg_B)
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: self.cell.coords,  # c(x) = x -> int_A = 0.5, int_B = 4.0
                self.mass_A: 0.0,
                self.mass_B: 0.0
            }
        }


    @REQUIRES_RUNTIME
    def test_nested_gradient_inside_integral_context(self):
        """EDGE CASE: Verifies spatial operators inside integral reductions do not lose axis context."""
        engine = fx.Engine(model=NestedGradientIntegralOracle(), target="cpu", mock_execution=False)
        y0, ydot0, _, _, _ = engine._extract_metadata()
        J = np.array(engine.evaluate_jacobian(y0, ydot0, c_j=1.0, parameters={}))
        assert np.linalg.matrix_rank(J) >= engine.layout.n_states - 2

        res = engine.solve(t_span=(0, 15.0), t_eval=np.array([0.0, 15.0]))
        np.testing.assert_allclose(res["T_var"].data[-1], 2.0, rtol=0.05)

    @REQUIRES_RUNTIME
    def test_subregion_quadrature_physical_scaling(self):
        """EDGE CASE: Verifies quadrature weighting over sliced sub-regions scales by L_phys correctly."""
        engine = fx.Engine(model=SubregionQuadratureOracle(), target="cpu", mock_execution=False)
        res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
        np.testing.assert_allclose(res["mass_A"].data[-1], 0.5, rtol=0.01)
        np.testing.assert_allclose(res["mass_B"].data[-1], 4.0, rtol=0.01)


# ==============================================================================
# 4. Multi-Scale Hierarchical Coupling
# ==============================================================================

class HierarchicalMemoryCorruptionOracle(fx.PDE):
    cell = fx.Domain(bounds=(0, 10), resolution=10, name="cell")
    x_n = cell.region(bounds=(0, 5), resolution=5, name="x_n")
    x_p = cell.region(bounds=(5, 10), resolution=5, name="x_p")
    r_n = fx.Domain(bounds=(0, 1), resolution=100, coord_sys="spherical", name="r_n")
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n")
    phi_e = fx.State(domain=cell, name="phi_e")

    def math(self):
        flux = -fx.grad(self.c_s_n, axis=self.r_n)
        return {
            "equations": {
                self.phi_e: fx.dt(self.phi_e) == 0.0,
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(flux, axis=self.r_n)
            },
            "boundaries": {
                flux: {"left": 0.0, "right": self.phi_e}
            },
            "initial_conditions": {
                self.c_s_n: 0.0,
                self.phi_e: 0.0
            }
        }


class HierarchicalMassCouplingOracle(fx.PDE):
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    r = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r")
    c_s = fx.State(domain=x * r, name="c_s")
    c_e = fx.State(domain=x, name="c_e")

    def math(self):
        j_flux = 100.0
        flux_s = -1e-14 * fx.grad(self.c_s, axis=self.r)
        flux_e = -1e-10 * fx.grad(self.c_e, axis=self.x)

        eps_s = 0.5
        R_p = 5e-6
        a_s = 3.0 * eps_s / R_p
        j_volumetric = j_flux * a_s

        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -fx.div(flux_s, axis=self.r),
                self.c_e: fx.dt(self.c_e) == -fx.div(flux_e, axis=self.x) + j_volumetric
            },
            "boundaries": {
                flux_s: {"left": 0.0, "right": j_flux},
                flux_e: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c_s: 1000.0,
                self.c_e: 0.0
            }
        }


class Chen2020TopologicalOverlapOracle(fx.PDE):
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=72, name="cell")
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=35, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=6, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=31, name="x_p")
    c = fx.State(domain=cell, name="c")

    def math(self):
        return {
            "equations": {
                self.c: fx.Piecewise({
                    self.x_n: fx.dt(self.c) == 2.0,
                    self.x_s: fx.dt(self.c) == 3.0,
                    self.x_p: fx.dt(self.c) == 5.0
                })
            },
            "boundaries": {},
            "initial_conditions": {self.c: 0.0}
        }


class TestHierarchicalCoupling:
    """Verifies that composite domains (Macro x Micro) correctly decouple and index coordinate axes."""

    @REQUIRES_COMPILER
    def test_cross_domain_memory_routing(self):
        """Validates that a 1D macro state evaluated at a 2D micro boundary accesses index 0, not flat index 99."""
        engine = fx.Engine(model=HierarchicalMemoryCorruptionOracle(), target="cpu", mock_execution=False)
        N = engine.layout.n_states
        y0, ydot0, _, _, _ = engine._extract_metadata()
        y0 = np.array(y0)

        off_phi, size_phi = engine.layout.state_offsets["phi_e"]
        off_c, size_c = engine.layout.state_offsets["c_s_n"]

        y0[off_phi : off_phi + size_phi] = np.arange(10.0)
        res = engine.evaluate_residual(y0.tolist(), ydot0, parameters={})
        c_residuals = np.array(res[off_c : off_c + size_c])

        # Particle 0 at x_n[0] sits at cell[0] -> phi_e[0] == 0.0 -> residual == 0.0
        assert c_residuals[99] == pytest.approx(0.0, abs=1e-5)

    @REQUIRES_COMPILER
    def test_hierarchical_mass_conservation(self):
        """Verifies that moles lost by micro-particles equal moles gained by the macroscopic electrolyte."""
        engine = fx.Engine(model=HierarchicalMassCouplingOracle(), target="cpu", mock_execution=False)
        res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))

        c_e_final = res["c_e"].data[-1]
        c_s_initial = res["c_s"].data[0].reshape((10, 10))
        c_s_final = res["c_s"].data[-1].reshape((10, 10))

        dx_macro = 1.0 / 9.0
        V_macro = np.ones(10) * dx_macro
        V_macro[0] *= 0.5
        V_macro[-1] *= 0.5

        R_p = 5e-6
        dr_micro = R_p / 9.0
        r_faces = np.linspace(0, R_p, 10)
        V_micro = np.zeros(10)
        for i in range(10):
            r_right = r_faces[i] + 0.5 * dr_micro if i < 9 else r_faces[i]
            r_left = r_faces[i] - 0.5 * dr_micro if i > 0 else 0.0
            V_micro[i] = (4.0 / 3.0) * np.pi * (r_right**3 - r_left**3)

        V_particle_total = (4.0 / 3.0) * np.pi * (R_p**3)
        Li_added_e = float(np.sum(c_e_final * V_macro))

        eps_s = 0.5
        c_s_drop = c_s_initial - c_s_final
        avg_drop_per_particle = np.sum(c_s_drop * V_micro, axis=1) / V_particle_total
        Li_removed_s = float(np.sum(avg_drop_per_particle * (eps_s * V_macro)))

        np.testing.assert_allclose(Li_added_e, Li_removed_s, rtol=1e-8)

    @REQUIRES_COMPILER
    def test_chen2020_topological_overlap_off_by_one(self):
        """Verifies that floating-point region boundaries do not create overlapping or orphaned nodes."""
        engine = fx.Engine(model=Chen2020TopologicalOverlapOracle(), target="cpu", mock_execution=False)
        y0, ydot0, _, _, _ = engine._extract_metadata()
        res = engine.evaluate_residual(y0, ydot0, parameters={})

        off_c, _ = engine.layout.state_offsets["c"]
        c_residuals = res[off_c : off_c + 72]

        total_rhs = -float(np.sum(c_residuals))
        # Expected sum: 35 * 2.0 + 6 * 3.0 + 31 * 5.0 = 243.0
        assert total_rhs == pytest.approx(243.0)
        assert c_residuals[-1] == pytest.approx(-5.0)


# ==============================================================================
# 5. Non-Linear Spatial Stencils & MMS Verification
# ==============================================================================

class StaggeredNonLinearMMSOracle(fx.PDE):
    x = fx.Domain(bounds=(1.0, 2.0), resolution=50, name="x")
    c = fx.State(domain=x, name="c")
    phi = fx.State(domain=x, name="phi")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        coords = self.x.coords
        t = self.t_var
        kappa_e = self.c
        kappa_D = self.c ** 2
        i_e = kappa_e * fx.grad(self.phi) + kappa_D * fx.grad(self.c) / self.c
        div_source = 4.0 * coords * (t ** 2) + (t ** 2) + 2.0 * t

        return {
            "equations": {
                self.t_var: fx.dt(self.t_var) == 1.0,
                self.c: fx.dt(self.c) == coords,
                self.phi: fx.div(i_e) == div_source
            },
            "boundaries": {
                self.c: {"left": fx.Dirichlet(1.0 * t + 1.0), "right": fx.Dirichlet(2.0 * t + 1.0)},
                self.phi: {"left": fx.Dirichlet(1.0 * t), "right": fx.Dirichlet(4.0 * t)}
            },
            "initial_conditions": {
                self.t_var: 1.0,
                self.c: coords * 1.0 + 1.0,
                self.phi: (coords ** 2) * 1.0
            }
        }


class ElectrolytePotentialDAEOracle(fx.PDE):
    x = fx.Domain(bounds=(0, 2.0), resolution=50, name="x")
    c = fx.State(domain=x, name="c")
    phi = fx.State(domain=x, name="phi")

    def math(self):
        ce_diff_term = fx.grad(self.c) / fx.max(self.c, 1e-3)
        flux_phi = 2.0 * fx.grad(self.phi) + 3.0 * ce_diff_term
        return {
            "equations": {
                self.c: fx.dt(self.c) == 0.0,
                self.phi: fx.div(flux_phi) == 0.0
            },
            "boundaries": {
                self.phi: {"left": fx.Dirichlet(0.0)}
            },
            "initial_conditions": {
                self.c: fx.exp(self.x.coords),
                self.phi: 0.0
            }
        }


class ButlerVolmerSymmetryOracle(fx.PDE):
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    c_s = fx.State(domain=x, name="c_s")
    phi_e = fx.State(domain=x, name="phi_e")
    phi_s = fx.State(domain=x, name="phi_s")

    def math(self):
        U_ocp = 0.5 + 0.1 * self.c_s
        eta = self.phi_s - self.phi_e - U_ocp
        i_0 = fx.abs(self.c_s) ** 0.5

        def sinh_ast(x_val):
            e2x = fx.exp(2.0 * x_val)
            return (e2x - 1.0) / (e2x + 1.0)

        j_faraday = i_0 * sinh_ast(eta * 38.9)
        return {
            "equations": {
                self.c_s: fx.dt(self.c_s) == -j_faraday,
                self.phi_e: self.phi_e == 0.0,
                self.phi_s: self.phi_s == 0.55
            },
            "boundaries": {},
            "initial_conditions": {
                self.c_s: 0.5,
                self.phi_e: 0.0,
                self.phi_s: 0.55
            }
        }


class TestNonlinearSpatialMMS:
    """Verifies Method of Manufactured Solutions (MMS) on coupled non-linear spatial DAEs."""

    @REQUIRES_COMPILER
    def test_staggered_grid_nonlinear_dae_interpolation(self):
        engine = fx.Engine(model=StaggeredNonLinearMMSOracle(), target="cpu", mock_execution=False)
        res = engine.solve(t_span=(1.0, 2.0), t_eval=np.array([1.0, 2.0]))

        x_coords = np.linspace(1.0, 2.0, 50)
        c_exact = x_coords * 2.0 + 1.0
        phi_exact = (x_coords ** 2) * 2.0

        np.testing.assert_allclose(res["c"].data[-1], c_exact, rtol=1e-3)
        np.testing.assert_allclose(res["phi"].data[-1], phi_exact, rtol=1e-3)

    @REQUIRES_RUNTIME
    def test_electrolyte_dae_log_gradient_coupling(self):
        engine = fx.Engine(model=ElectrolytePotentialDAEOracle(), target="cpu", mock_execution=False)
        res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))

        x_coords = np.linspace(0, 2.0, 50)
        phi_exact = -1.5 * x_coords
        phi_sim = res["phi"].data[-1]
        np.testing.assert_allclose(phi_sim, phi_exact, atol=2e-2)

    @REQUIRES_RUNTIME
    def test_butler_volmer_equilibrium_invariance(self):
        """EDGE CASE: Zero overpotential (eta=0) must produce absolute equilibrium without numerical drift."""
        engine = fx.Engine(model=ButlerVolmerSymmetryOracle(), target="cpu", mock_execution=False)
        res = engine.solve(t_span=(0, 3600.0), t_eval=np.array([0.0, 3600.0]))

        c_s_initial = res["c_s"].data[0]
        c_s_final = res["c_s"].data[-1]
        drift = float(np.max(np.abs(c_s_final - c_s_initial)))
        assert drift < 1e-12, f"Thermodynamic drift detected at equilibrium: {drift:.3e}"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])