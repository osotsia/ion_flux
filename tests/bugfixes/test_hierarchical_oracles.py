"""
End-to-End Industry Oracles: The Hierarchical Crucible

This suite provides Method of Manufactured Solutions (MMS) and exact 
thermodynamic stability oracles specifically tailored for the highly coupled, 
non-linear phenomena observed in Doyle-Fuller-Newman (DFN) models.
"""

import pytest
import numpy as np
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine

# ==============================================================================
# Environment Configuration
# ==============================================================================

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

# ==============================================================================
# ORACLE 1: Memory Corruption / Misrouting in Cross-Domain Boundaries
# ==============================================================================

class HierarchicalMemoryCorruptionOracle(fx.PDE):
    """
    Isolates the out-of-bounds memory fetch when a micro-particle boundary 
    depends on a global macroscopic state array.
    """
    cell = fx.Domain(bounds=(0, 10), resolution=10, name="cell")
    x_n = cell.region(bounds=(0, 5), resolution=5, name="x_n")
    x_p = cell.region(bounds=(5, 10), resolution=5, name="x_p")
    
    r_n = fx.Domain(bounds=(0, 1), resolution=100, coord_sys="spherical", name="r_n")
    
    # 2D Field (Size: 5 * 100 = 500 nodes)
    c_s_n = fx.State(domain=x_n * r_n, name="c_s_n") 
    # 1D Global Field (Size: 10 nodes)
    phi_e = fx.State(domain=cell, name="phi_e")      
    
    def math(self):
        flux = -fx.grad(self.c_s_n, axis=self.r_n)
        
        return {
            "equations": {
                self.phi_e: fx.dt(self.phi_e) == 0.0,
                self.c_s_n: fx.dt(self.c_s_n) == -fx.div(flux, axis=self.r_n)
            },
            "boundaries": {
                # Evaluated over x_n * r_n, but referencing cell.
                flux: {"left": 0.0, "right": self.phi_e}
            },
            "initial_conditions": {
                self.c_s_n: 0.0,
                self.phi_e: 0.0
            }
        }

@REQUIRES_COMPILER
def test_cross_domain_memory_corruption():
    """
    PROBE: Proves that `phi_e` is incorrectly indexed using the flat 2D `c_s_n` 
    index. Because of the C++ CLAMP macro, the index does not segfault but maps 
    to the entirely wrong physical location.
    """
    engine = Engine(model=HierarchicalMemoryCorruptionOracle(), target="cpu", mock_execution=False)
    
    N = engine.layout.n_states
    y0, ydot0, _, _, _ = engine._extract_metadata()
    y0 = np.array(y0)
    
    off_phi, size_phi = engine.layout.state_offsets["phi_e"]
    off_c, size_c = engine.layout.state_offsets["c_s_n"]
    
    # Establish a known linear gradient across phi_e: [0, 1, 2, ..., 9]
    y0[off_phi : off_phi + size_phi] = np.arange(10.0)
    
    res = engine.evaluate_residual(y0.tolist(), ydot0, parameters={})
    c_residuals = np.array(res[off_c : off_c + size_c])
    
    # Analyze the residual at the rightmost boundary of the FIRST particle (x_n node 0).
    # Its flat index in the c_s_n array is 99.
    res_0 = c_residuals[99]
    
    # Mathematical Truth: The first particle sits at x_n[0], which corresponds to cell[0].
    # Therefore, phi_e should evaluate to 0.0, yielding a residual of exactly 0.0.
    expected_res_correct = 0.0
    
    assert res_0 == pytest.approx(expected_res_correct, abs=1e-5), \
        f"Memory Misrouting Confirmed: Expected the correct mapping to fetch phi_e[0] (Residual 0.0), " \
        f"but got a residual of {res_0:.1f}. The compiler is fetching out-of-bounds memory."


# ==============================================================================
# ORACLE 2: Flat-Line Composite Integration
# ==============================================================================

class CompositeIntegrationOracle(fx.PDE):
    """
    Isolates the integration scaling of multi-scale domains.
    Mathematically: ∫ 1.0 dV = V_macro * V_micro
    """
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

@REQUIRES_COMPILER
def test_composite_domain_integration_failure():
    """
    PROBE: Proves the compiler lacks support for 2D composite integration. 
    It will currently crash with a KeyError during AST translation, and must be 
    patched to support nested nested loops yielding `V_macro * V_micro`.
    """
    engine = Engine(model=CompositeIntegrationOracle(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0, parameters={})
    
    off_mass, _ = engine.layout.state_offsets["total_mass"]
    
    # The evaluated integral is `-res` of the algebraic equation
    simulated_integral = -res[off_mass]
    
    # Exact Analytical Volume:
    # V_x = 2.0
    # V_r = (4/3) * pi * (1.0)^3 = 4.18879
    # Total volume = 2.0 * 4.18879 = 8.37758
    exact_integral = 2.0 * (4.0/3.0) * np.pi * (1.0)**3
    
    assert simulated_integral == pytest.approx(exact_integral, rel=1e-2), \
        f"Composite Integration Bug Confirmed! Expected analytical mass {exact_integral:.3f}, " \
        f"but got {simulated_integral:.3f}."


# ==============================================================================
# ORACLE 3: FVM Spherical Exactness
# ==============================================================================

class SphericalFVMOracle(fx.PDE):
    """
    Manufactured Analytical Solution.
    c(r, t) = r^2 + 6*t
    dt(c) = 6.0
    div(grad(c)) = 6.0
    """
    r = fx.Domain(bounds=(0, 1.0), resolution=10, coord_sys="spherical", name="r")
    c = fx.State(domain=r, name="c")

    def math(self):
        flux = -fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                # dt(c) - div(grad(c)) = 0 -> dt(c) - (-div(flux)) = 0
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r),
            },
            "boundaries": {
                # grad(r^2) at r=1.0 is 2.0. So flux = -2.0.
                flux: {"left": 0.0, "right": -2.0}
            },
            "initial_conditions": {
                # Initialize c(r, 0) = r^2
                self.c: self.r.coords ** 2,
            }
        }

@REQUIRES_COMPILER
def test_spherical_fvm_volume_exactness():
    """Ensures the compiler has no off-by-one errors regarding Spherical geometry arrays."""
    engine = Engine(model=SphericalFVMOracle(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _, _ = engine._extract_metadata()
    y0 = np.array(y0)
    ydot0 = np.zeros_like(y0)
    
    off_c, size_c = engine.layout.state_offsets["c"]
    ydot0[off_c : off_c + size_c] = 6.0
    
    res = engine.evaluate_residual(y0.tolist(), ydot0.tolist(), parameters={})
    c_residuals = res[off_c : off_c + size_c]
    
    np.testing.assert_allclose(
        c_residuals, 0.0, atol=1e-12, 
        err_msg="COMPILER BUG: Spherical FVM divergence failed."
    )


# ==============================================================================
# ORACLE 4: Non-Linear Staggered Grid DAE Crucible
# ==============================================================================

class StaggeredNonLinearMMSOracle(fx.PDE):
    """
    Manufactures a highly non-linear spatial DAE designed to expose face-interpolation 
    errors in the FVM lowering pass.
    """
    x = fx.Domain(bounds=(1.0, 2.0), resolution=50, name="x")
    
    c = fx.State(domain=x, name="c")
    phi = fx.State(domain=x, name="phi")
    t_var = fx.State(domain=None, name="t_var")

    def math(self):
        # FIX: Explicitly shift coords to match the [1.0, 2.0] physical bounds
        coords = self.x.coords + 1.0
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
                self.c: {
                    "left": fx.Dirichlet(1.0 * t + 1.0), 
                    "right": fx.Dirichlet(2.0 * t + 1.0)
                },
                self.phi: {
                    "left": fx.Dirichlet(1.0 * t), 
                    "right": fx.Dirichlet(4.0 * t)
                }
            },
            "initial_conditions": {
                self.t_var: 1.0, 
                self.c: coords * 1.0 + 1.0,
                self.phi: (coords ** 2) * 1.0
            }
        }

@REQUIRES_COMPILER
def test_oracle_staggered_grid_nonlinear_dae_interpolation():
    """
    PROBE: Fails if the AST-to-C++ translator misapplies cell-center states to 
    cell-face flux evaluations during non-linear tensor assembly.
    """
    engine = Engine(model=StaggeredNonLinearMMSOracle(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(1.0, 2.0), t_eval=np.array([1.0, 2.0]))
    
    x_coords = np.linspace(1.0, 2.0, 50)
    c_exact = x_coords * 2.0 + 1.0
    phi_exact = (x_coords ** 2) * 2.0
    
    np.testing.assert_allclose(res["c"].data[-1], c_exact, rtol=1e-3)
    np.testing.assert_allclose(res["phi"].data[-1], phi_exact, rtol=1e-3,
        err_msg="Non-Linear DAE Failed: Face interpolation for kappa_e or grad(c)/c is flawed.")


# ==============================================================================
# ORACLE 5: Hierarchical Inter-Domain Mass Mapping Crucible
# ==============================================================================

class HierarchicalMassCouplingOracle(fx.PDE):
    """
    A minimal topology tracking lithium flux extracted from a micro-spherical 
    particle and injected into a macro-Cartesian mesh. Evaluates if the FVM 
    geometric translation factor (a_s) is mathematically conserved in the native arrays.
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    r = fx.Domain(bounds=(0, 5e-6), resolution=10, coord_sys="spherical", name="r")
    macro_micro = x * r
    
    c_s = fx.State(domain=macro_micro, name="c_s")
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

@REQUIRES_COMPILER
def test_oracle_hierarchical_mass_coupling_crucible():
    """
    PROBE: Directly calculates exact NumPy FVM volumes and dots them against the 
    simulated arrays to strictly account for every single mole of lithium transferred 
    across the scale gap.
    """
    engine = Engine(model=HierarchicalMassCouplingOracle(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 1.0), t_eval=np.array([0.0, 1.0]))
    
    c_e_final = res["c_e"].data[-1]
    c_s_initial = res["c_s"].data[0].reshape((10, 10))
    c_s_final = res["c_s"].data[-1].reshape((10, 10))
    
    # 1. Exact Macro Volumes (dx = 1.0 / 9)
    dx_macro = 1.0 / 9.0
    V_macro = np.ones(10) * dx_macro
    V_macro[0] *= 0.5
    V_macro[-1] *= 0.5
    
    # 2. Exact Micro Volumes (Spherical, dr = 5e-6 / 9)
    R_p = 5e-6
    dr_micro = R_p / 9.0
    r_faces = np.linspace(0, R_p, 10)
    
    V_micro = np.zeros(10)
    for i in range(10):
        r_right = r_faces[i] + 0.5*dr_micro if i < 9 else r_faces[i]
        r_left = r_faces[i] - 0.5*dr_micro if i > 0 else 0.0
        V_micro[i] = (4.0/3.0) * np.pi * (r_right**3 - r_left**3)
        
    V_particle_total = (4.0/3.0) * np.pi * R_p**3
    
    # 3. Mass Balance Accounting
    Li_added_e = np.sum(c_e_final * V_macro)
    
    eps_s = 0.5
    c_s_drop = c_s_initial - c_s_final
    avg_drop_per_particle = np.sum(c_s_drop * V_micro, axis=1) / V_particle_total
    Li_removed_s = np.sum(avg_drop_per_particle * (eps_s * V_macro))
    
    np.testing.assert_allclose(Li_added_e, Li_removed_s, rtol=1e-8,
        err_msg="Hierarchical Mass Leak Detected! The mass of lithium entering the macroscopic "
                "mesh does not equal the mass leaving the microscopic mesh.")


# ==============================================================================
# ORACLE 6: Chen2020 Topological Overlap (The Off-By-One Bug)
# ==============================================================================

class Chen2020TopologicalOverlapOracle(fx.PDE):
    """
    Exposes the floating-point `int(round(...))` bug in Domain.region.
    Using the exact mesh sizing from the Chen2020 DFN model (35, 6, 31 nodes).
    """
    cell = fx.Domain(bounds=(0, 172.8e-6), resolution=72, name="cell")
    x_n = cell.region(bounds=(0, 85.2e-6), resolution=35, name="x_n")
    x_s = cell.region(bounds=(85.2e-6, 97.2e-6), resolution=6, name="x_s")
    x_p = cell.region(bounds=(97.2e-6, 172.8e-6), resolution=31, name="x_p")
    
    c = fx.State(domain=cell, name="c")

    def math(self):
        return {
            "equations": {
                # Assign a distinct prime number to each region's derivative
                self.c: fx.Piecewise({
                    self.x_n: fx.dt(self.c) == 2.0,
                    self.x_s: fx.dt(self.c) == 3.0,
                    self.x_p: fx.dt(self.c) == 5.0
                })
            },
            "boundaries": {},
            "initial_conditions": {
                self.c: 0.0
            }
        }

@REQUIRES_COMPILER
def test_oracle_chen2020_topological_overlap_off_by_one():
    """
    PROBE: Fails if the `Domain.region` math creates overlapping indices 
    or orphans nodes at the right boundary due to Python 3 float rounding.
    """
    engine = Engine(model=Chen2020TopologicalOverlapOracle(), target="cpu", mock_execution=False)
    
    y0, ydot0, _, _, _ = engine._extract_metadata()
    res = engine.evaluate_residual(y0, ydot0, parameters={})
    
    off_c, _ = engine.layout.state_offsets["c"]
    c_residuals = res[off_c : off_c + 72]
    
    # Residual = ydot - RHS = 0.0 - RHS
    # Total sum of RHS should be 35*2 + 6*3 + 31*5 = 70 + 18 + 155 = 243
    total_rhs = -np.sum(c_residuals)
    
    assert total_rhs == pytest.approx(243.0), \
        f"Topological Off-By-One Bug Confirmed! Expected total derivative sum of 243.0, " \
        f"but got {total_rhs}. The sub-mesh indices are overlapping and overwriting each other!"
        
    # Specifically check the final node. If it's orphaned, it evaluates to 0.0 instead of 5.0
    assert c_residuals[-1] == pytest.approx(-5.0), \
        "The rightmost node of the cell was orphaned (never evaluated) because the cathode region shifted left!"


# ==============================================================================
# ORACLE 7: EIS Mass Matrix Extraction (Engine Bug)
# ==============================================================================

class CapacitiveImpedanceOracle(fx.PDE):
    """
    Proves the Engine's Analytical EIS solver incorrectly extracts the Mass Matrix.
    By allowing non-unit multipliers on the time derivative (C * dt(V)), the 
    true mass matrix M = C. The engine currently hardcodes M = id_arr (1.0).
    """
    V = fx.State(domain=None, name="V")
    C_cap = fx.Parameter(default=5.0, name="C_cap")
    R = fx.Parameter(default=2.0, name="R")
    i_app = fx.Parameter(default=1.0, name="i_app")
    
    def math(self):
        return {
            "equations": {
                # Implicit capacity: M = 5.0
                self.V: self.C_cap * fx.dt(self.V) == self.i_app - self.V / self.R
            },
            "boundaries": {},
            "initial_conditions": {self.V: 0.0}
        }

@REQUIRES_COMPILER
def test_oracle_eis_mass_matrix_extraction():
    """
    PROBE: Compares the simulated EIS against the exact analytical Transfer Function.
    """
    engine = Engine(model=CapacitiveImpedanceOracle(), target="cpu", mock_execution=False)
    session = engine.start_session(parameters={"C_cap": 5.0, "R": 2.0, "i_app": 1.0})
    session.reach_steady_state()
    
    w_arr = np.array([0.1, 1.0, 10.0])
    eis_res = session.solve_eis(w_arr, input_var="i_app", output_var="V")
    
    Z_sim = eis_res["Z_real"].data + 1j * eis_res["Z_imag"].data
    
    # Exact Analytical Transfer Function: Z(w) = R / (1 + j * w * R * C)
    # Note: solve_eis treats the input array as frequencies in Hz, converting 
    # to rad/s natively. We must do the same for the analytical truth.
    w_rad = w_arr * 2 * np.pi
    Z_exact = 2.0 / (1.0 + 1j * w_rad * 2.0 * 5.0)
    
    np.testing.assert_allclose(
        np.real(Z_sim), np.real(Z_exact), rtol=1e-4,
        err_msg="EIS Mass Matrix Bug! The engine is hardcoding M=1.0 instead of extracting M=5.0."
    )


# ==============================================================================
# ORACLE 8: Continuous Adjoint VJP Sensitivities (Engine Bug)
# ==============================================================================

class AdjointCapacityOracle(fx.PDE):
    """
    Proves the Continuous Adjoint solver (adjoint.rs) also suffers from the 
    hardcoded Mass Matrix bug.
    """
    y = fx.State(domain=None, name="y")
    C_cap = fx.Parameter(default=2.0, name="C_cap")
    k = fx.Parameter(default=1.0, name="k")
    
    def math(self):
        return {
            "equations": {
                self.y: self.C_cap * fx.dt(self.y) == -self.k * self.y
            },
            "boundaries": {},
            "initial_conditions": {self.y: 1.0}
        }

@REQUIRES_COMPILER
def test_oracle_adjoint_mass_matrix_vjp():
    """
    PROBE: Compares the Enzyme-derived continuous Adjoint gradient to an exact 
    Scipy-derived analytical ground truth.
    """
    engine = Engine(model=AdjointCapacityOracle(), target="cpu", mock_execution=False)
    t_eval = np.linspace(0, 5.0, 50)
    
    # Forward Pass
    res = engine.solve(t_eval=t_eval, parameters={"C_cap": 2.0, "k": 1.0}, requires_grad=["C_cap"])
    
    # Loss = Sum( y^2 )
    y_sim = res["y"].data
    loss_val = float(np.sum(y_sim ** 2))
    
    # Manual backprop to inject into the engine
    dl_dy = 2.0 * y_sim
    res.trajectory["requires_grad"] = ["C_cap"]
    
    # Trigger native Adjoint pass
    loss_obj = fx.metrics.Loss(loss_val, engine=engine, trajectory=res.trajectory, dl_dy_mapped=np.expand_dims(dl_dy, axis=1))
    grads = loss_obj.backward()
    
    simulated_grad = grads["C_cap"]
    
    # Exact Analytical Oracle
    # y(t) = exp(-k * t / C)
    # dLoss/dC = sum( d/dC [exp(-2 * k * t / C)] )
    # dLoss/dC = sum( exp(-2 * k * t / C) * (2 * k * t / C^2) )
    exact_grad = np.sum( np.exp(-2.0 * 1.0 * t_eval / 2.0) * (2.0 * 1.0 * t_eval / (2.0**2)) )
    
    np.testing.assert_allclose(
        simulated_grad, exact_grad, rtol=2e-2,
        err_msg="Adjoint Mass Matrix Bug! The VJP loop in Rust is likely ignoring the capacity multiplier."
    )

# ==============================================================================
# ORACLE 9: Variable Transference Mass Leak (Literature Inconsistency)
# ==============================================================================

class VariableTransferenceLeakProbe(fx.PDE):
    """
    Proves that `(1 - t_+) * j` leaks mass when t_plus is a spatial field.
    The strictly conservative form is: -div(-D*grad(c) + t_+*i_e / F) + j/F
    """
    x = fx.Domain(bounds=(0, 1.0), resolution=10, name="x")
    
    c_leaky = fx.State(domain=x, name="c_leaky")
    c_strict = fx.State(domain=x, name="c_strict")
    
    def math(self):
        F = 96485.0
        j_val = 1000.0
        
        # Create a variable t_plus field
        t_plus = 0.2 + 0.1 * self.x.coords 
        
        # Manufactured i_e gradient to trigger the leak
        i_e = 10.0 * self.x.coords 
        
        # 1. The published formulation from Table S2 of O'Regan 2022
        flux_leaky = -1e-10 * fx.grad(self.c_leaky)
        eq_leaky = -fx.div(flux_leaky) + (1.0 - t_plus) * j_val / F
        
        # 2. The mathematically strict, physically conservative formulation
        flux_strict = -1e-10 * fx.grad(self.c_strict) + (t_plus * i_e) / F
        eq_strict = -fx.div(flux_strict) + j_val / F
        
        return {
            "equations": {
                self.c_leaky: fx.dt(self.c_leaky) == eq_leaky,
                self.c_strict: fx.dt(self.c_strict) == eq_strict
            },
            "boundaries": {
                flux_leaky: {"left": 0.0, "right": 0.0},
                flux_strict: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c_leaky: 1000.0,
                self.c_strict: 1000.0
            }
        }

@pytest.mark.skip(reason="The O'Regan 2022 paper publishes a non-conservative electrolyte mass equation.")
@REQUIRES_COMPILER
def test_oracle_literature_transference_mass_leak():
    """
    PROBE: Integrates total electrolyte mass. If it drifts from the exact analytical 
    influx of `j / F`, mass has leaked. 
    
    This test is skipped. The user script faithfully reproduces the DFN equations
    published in the O'Regan 2022 paper, but the paper itself contains an error.
    """
    engine = Engine(model=VariableTransferenceLeakProbe(), target="cpu", mock_execution=False)
    
    res = engine.solve(t_span=(0, 10.0), t_eval=np.array([0.0, 10.0]))
    
    # Exact analytical mass added = j_val / F * time * length
    exact_mass_added = (1000.0 / 96485.0) * 10.0 * 1.0
    
    mass_leaky = np.mean(res["c_leaky"].data[-1]) - 1000.0
    mass_strict = np.mean(res["c_strict"].data[-1]) - 1000.0
    
    np.testing.assert_allclose(mass_strict, exact_mass_added, rtol=1e-4)
    
    np.testing.assert_allclose(
        mass_leaky, exact_mass_added, rtol=1e-4,
        err_msg="Literature Inconsistency! The user faithfully implemented Table S2 of O'Regan 2022, "
                "which uses the simplified source term `(1 - t_plus) * j / F`. However, because Eq 21 "
                "defines t_plus as a spatial variable, factoring it outside the divergence operator "
                "mathematically omits the `i_e * grad(t_plus)` term, causing a global mass leak."
    )

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])