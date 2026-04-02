"""
Frontend DSL: Semantic API and AST Capture

This suite validates the strict V2 Declarative API (`equations`, `boundaries`, `initial_conditions`).
It ensures Python operator overloading generates the correct Abstract Syntax Tree (AST), 
that submodels maintain namespace isolation, and that complex topologies (like ALE moving meshes)
are flagged accurately for the compiler.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.runtime.engine import Engine, RUST_FFI_AVAILABLE
from ion_flux.dsl.ast import validate_ast

# ==============================================================================
# Model 1: The Heavyweight (Piecewise, Hierarchical Domains, DAEs)
# ==============================================================================

class ComprehensiveBatteryModel(fx.PDE):
    """Validates Piecewise regions, Spatial DAEs, Terminal Multiplexing, and Dirichlet boundaries."""
    cell = fx.Domain(bounds=(0, 2), resolution=20)
    reg_A = cell.region(bounds=(0, 1), resolution=10, name="reg_A")
    reg_B = cell.region(bounds=(1, 2), resolution=10, name="reg_B")
    
    r = fx.Domain(bounds=(0, 1), resolution=5, coord_sys="spherical")
    
    c_e = fx.State(domain=cell, name="c_e")
    phi_e = fx.State(domain=cell, name="phi_e")
    c_s = fx.State(domain=reg_A * r, name="c_s")
    
    V_cell = fx.State(name="V_cell")
    i_app = fx.State(name="i_app")
    
    terminal = fx.Terminal(current=i_app, voltage=V_cell)
    
    def math(self):
        N_e = -fx.grad(self.c_e)
        i_e = -fx.grad(self.phi_e)
        N_s = -fx.grad(self.c_s, axis=self.r)
        
        return {
            "equations": {
                # 1. Piecewise PDE mapping
                self.c_e: fx.Piecewise({
                    self.reg_A: fx.dt(self.c_e) == -fx.div(N_e) - 1.0,
                    self.reg_B: fx.dt(self.c_e) == -fx.div(N_e)
                }),
                # 2. Pure Spatial DAE (No dt)
                self.phi_e: fx.div(i_e) == 0.0,
                # 3. Macro-Micro PDE
                self.c_s: fx.dt(self.c_s) == -fx.div(N_s, axis=self.r),
                # 4. 0D Algebraic
                self.V_cell: self.V_cell == 4.2 - self.phi_e.right
            },
            "boundaries": {
                N_e: {"left": 0.0, "right": 0.0},
                self.phi_e: {"left": fx.Dirichlet(0.0)},
                N_s: {"right": -1.0}
            },
            "initial_conditions": {
                self.c_e: 1000.0,
                self.phi_e: 0.0,
                self.c_s: 500.0,
                self.V_cell: 4.2,
                self.i_app: 0.0
            }
        }

# ==============================================================================
# Model 2: The Composer (Submodels, Namespacing, Merge Mechanics)
# ==============================================================================

class SubParticle(fx.PDE):
    """Reusable 1D diffusion submodel."""
    r = fx.Domain(bounds=(0, 1), resolution=5, coord_sys="spherical")
    c = fx.State(domain=r)
    D = fx.Parameter(default=1.0)
    
    def math(self, flux_bnd: fx.Node):
        flux = -self.D * fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r)
            },
            "boundaries": {
                flux: {"left": 0.0, "right": flux_bnd}
            },
            "initial_conditions": {
                self.c: 0.0
            }
        }

class CompositeCell(fx.PDE):
    """Validates fx.merge, deepcopy isolation, and prefixing."""
    anode = SubParticle()
    cathode = SubParticle()
    macro_v = fx.State(domain=None)
    
    def math(self):
        macro_sys = {
            "equations": {
                self.macro_v: fx.dt(self.macro_v) == self.anode.c.right - self.cathode.c.right
            },
            "boundaries": {},
            "initial_conditions": {
                self.macro_v: 1.0
            }
        }
        return fx.merge(macro_sys, self.anode.math(1.0), self.cathode.math(-1.0))

class BadSharedDomainParent(fx.PDE):
    """Tests isolation boundary failures (sharing external domains)."""
    macro_x = fx.Domain(bounds=(0, 1), resolution=5)
    
    class SharedDomainConsumer(fx.PDE):
        c = fx.State()
        def __init__(self, shared_domain: fx.Domain):
            super().__init__()
            self.x = shared_domain
            self.c.domain = self.x
        def math(self):
            return {
                "equations": {self.c: fx.dt(self.c) == fx.grad(self.c)},
                "boundaries": {}, "initial_conditions": {self.c: 1.0}
            }
            
    def __init__(self):
        super().__init__()
        self.sub_a = self.SharedDomainConsumer(self.macro_x)
        self.sub_b = self.SharedDomainConsumer(self.macro_x)
        
    def math(self):
        return fx.merge(self.sub_a.math(), self.sub_b.math())

# ==============================================================================
# Model 3: Advanced Math & Arbitrary Lagrangian-Eulerian (ALE) Moving Meshes
# ==============================================================================

class AdvancedMathAndALE(fx.PDE):
    """Validates Relational triggers, Integrals, and Dynamic Boundary tracking."""
    x = fx.Domain(bounds=(0, 1), resolution=5)
    c = fx.State(domain=x)
    L = fx.State(domain=None)
    
    def math(self):
        flux = -fx.grad(self.c)
        reaction = fx.exp(-1.0 / fx.max(self.c, 0.1)) * fx.sin(self.c)
        trigger = self.L > 2.0
        
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux) + reaction,
                self.L: fx.dt(self.L) == fx.integral(self.c, over=self.x) * trigger
            },
            "boundaries": {
                self.x: {"right": self.L}, # Triggers ALE Moving Boundary Mode!
                flux: {"left": 0.0, "right": 0.0}
            },
            "initial_conditions": {
                self.c: 1.0,
                self.L: 1.0
            }
        }


# ==============================================================================
# Tests
# ==============================================================================

def test_ast_capture_and_semantic_buckets():
    """Validates that explicit API targeting maps to the correct structural JSON buckets."""
    model = ComprehensiveBatteryModel()
    ast = model.ast()
    
    assert "equations" in ast
    assert "boundaries" in ast
    assert "initial_conditions" in ast
    assert "domains" in ast
    
    # 1. Terminal injection verification
    # User declared 4 equations (c_e, phi_e, c_s, V_cell). Terminal dynamically injected 1 (i_app).
    assert len(ast["equations"]) == 5
    
    # 2. Piecewise structure validation
    ce_eq = next(eq for eq in ast["equations"] if eq["state"] == "c_e")
    assert ce_eq["type"] == "piecewise"
    assert len(ce_eq["regions"]) == 2
    
    # 3. Spatial DAE capture
    phi_eq = next(eq for eq in ast["equations"] if eq["state"] == "phi_e")
    assert phi_eq["type"] == "standard"
    assert phi_eq["eq"]["left"]["op"] == "div"  # The 0 == div(...) maps the div to the left
    
    # 4. Dirichlet boundary overrides
    bc_phi = next(bc for bc in ast["boundaries"] if bc.get("state") == "phi_e")
    assert bc_phi["type"] == "dirichlet"
    assert bc_phi["bcs"]["left"]["type"] == "Scalar"


def test_model_composition_and_namespacing():
    """Validates submodel deepcopying, fx.merge mechanics, and parameter prefixing."""
    model = CompositeCell()
    ast = model.ast()
    
    # States should have been natively prefixed by the submodel attribute names
    states_in_eqs = {eq["state"] for eq in ast["equations"]}
    assert "anode_c" in states_in_eqs
    assert "cathode_c" in states_in_eqs
    assert "macro_v" in states_in_eqs
    
    # Domains should be cloned and prefixed to prevent overlapping topological graphs
    assert "anode_r" in ast["domains"]
    assert "cathode_r" in ast["domains"]
    assert "r" not in ast["domains"]
    
    # Parameters should be prefixed and extracted successfully
    engine = Engine(model, target="cpu", mock_execution=True)
    assert "anode_D" in engine.layout.param_offsets
    assert "cathode_D" in engine.layout.param_offsets


def test_shared_domain_deepcopy_isolation_boundary():
    """
    Validates that passing a parent domain to a submodel safely triggers a deepcopy.
    The submodel receives its own isolated coordinate space, preventing silent mesh conflicts.
    """
    model = BadSharedDomainParent()
    
    # The parent's original domain remains untouched
    assert model.macro_x.name == "macro_x"
    # But internal submodel references were copied and prefixed
    assert model.sub_a.x.name == "sub_a_macro_x"
    assert model.sub_b.x.name == "sub_b_macro_x"
    
    ast = model.ast()
    assert "sub_a_macro_x" in ast["domains"]
    assert "sub_b_macro_x" in ast["domains"]


def test_advanced_math_and_moving_boundaries():
    """Validates relational triggers, integral generation, and ALE boundary bindings."""
    model = AdvancedMathAndALE()
    ast = model.ast()
    
    # 1. ALE Moving Boundary Tagging
    ale_bc = next(bc for bc in ast["boundaries"] if bc.get("type") == "moving_domain")
    assert ale_bc["domain"] == "x"
    assert ale_bc["bcs"]["right"]["name"] == "L" # Bound directly to a State Unknown
    
    # 2. Advanced Math & Trigger Mapping
    l_eq = next(eq for eq in ast["equations"] if eq["state"] == "L")
    rhs = l_eq["eq"]["right"]
    
    assert rhs["type"] == "BinaryOp" and rhs["op"] == "mul"
    assert rhs["left"]["op"] == "integral"
    assert rhs["right"]["op"] == "gt"


@pytest.mark.skipif(not RUST_FFI_AVAILABLE, reason="Requires native JIT compilation.")
def test_dae_masking_extraction():
    """
    Validates that the compiler uses the AST output to generate the correct 
    binary Differential-Algebraic (DAE) masking vectors for the native numerical solver.
    """
    model = ComprehensiveBatteryModel()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    _, _, id_arr, _ = engine._extract_metadata()
    id_arr = np.array(id_arr)
    
    # 1. c_e is a standard PDE (Should be masked as 1.0)
    off_ce, size_ce = engine.layout.state_offsets["c_e"]
    assert np.all(id_arr[off_ce : off_ce + size_ce] == 1.0)
    
    # 2. phi_e is a spatial DAE (Lacks fx.dt(), should be masked as 0.0)
    off_phi, size_phi = engine.layout.state_offsets["phi_e"]
    assert np.all(id_arr[off_phi : off_phi + size_phi] == 0.0)
    
    # 3. V_cell is a 0D Algebraic constraint (0.0)
    off_v, _ = engine.layout.state_offsets["V_cell"]
    assert id_arr[off_v] == 0.0


def test_ast_validation_schema_rejections():
    """Validates the compiler throws exact, actionable errors when users malform dicts."""
    
    # 1. Invalid root buckets
    bad_payload = {"pdes": []}
    with pytest.raises(ValueError, match="Invalid equation bucket"):
        validate_ast(bad_payload)
        
    # 2. Missing states in equations
    bad_eq_payload = {
        "equations": [{"type": "standard"}], 
        "boundaries": [],
        "initial_conditions": []
    }
    with pytest.raises(ValueError, match="Equations must explicitly define 'state' and 'type'"):
        validate_ast(bad_eq_payload)
        
    # 3. Malformed boundaries
    bad_bc_payload = {
        "equations": [],
        "boundaries": [{"type": "dirichlet"}], # Missing "bcs" dict
        "initial_conditions": []
    }
    with pytest.raises(ValueError, match="Boundaries must explicitly define 'type' and 'bcs'"):
        validate_ast(bad_bc_payload)