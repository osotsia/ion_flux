
import pytest
import numpy as np
import ion_flux as fx
from ion_flux import Engine

# ==============================================================================
# 1. Happy Path: Composing and Executing a Modular System
# ==============================================================================

class SubParticle(fx.PDE):
    """Reusable 1D diffusion submodel."""
    r = fx.Domain(bounds=(0, 1), resolution=5, coord_sys="spherical")
    c = fx.State(domain=r)
    D = fx.Parameter(default=1.0)
    
    def math(self, flux_boundary: fx.Node):
        flux = -self.D * fx.grad(self.c, axis=self.r)
        return {
            "equations": {
                self.c: fx.dt(self.c) == -fx.div(flux, axis=self.r)
            },
            "boundaries": {
                flux: {"left": 0.0, "right": flux_boundary}
            },
            "initial_conditions": {}
        }

class ModularCell(fx.PDE):
    """Parent model instantiating two isolated submodels."""
    anode = SubParticle()
    cathode = SubParticle()
    macro_v = fx.State(domain=None)
    
    def math(self):
        # Extract boundaries from the prefixed submodels
        c_anode_surf = self.anode.c.right
        c_cathode_surf = self.cathode.c.right
        
        macro_sys = {
            "equations": {
                self.macro_v: fx.dt(self.macro_v) == -0.1 * self.macro_v
            },
            "boundaries": {},
            "initial_conditions": {
                self.macro_v: 5.0,
                self.anode.c: 100.0,
                self.cathode.c: 200.0
            }
        }
        
        # Merge ASTs, passing the macro state as the boundary condition
        return fx.merge(
            macro_sys,
            self.anode.math(flux_boundary=self.macro_v),
            self.cathode.math(flux_boundary=-self.macro_v)
        )

def test_happy_path_modular_composition():
    """Validates that a composed model compiles, layouts memory correctly, and executes."""
    model = ModularCell()
    engine = Engine(model=model, target="cpu:serial", mock_execution=False)
        
    # 1. Verify recursive layout extraction
    assert "macro_v" in engine.layout.state_offsets
    assert "anode_c" in engine.layout.state_offsets
    assert "cathode_c" in engine.layout.state_offsets
    
    assert "anode_D" in engine.layout.param_offsets
    assert "cathode_D" in engine.layout.param_offsets
    
    # 2. Verify topological isolation (domains were deepcopied and prefixed)
    assert "anode_r" in engine.cpp_source
    assert "cathode_r" in engine.cpp_source
    
    # 3. Verify execution and mathematical coupling
    res = engine.solve(t_span=(0, 1.0))
    assert res.status == "completed"
    
    # Anode received positive flux (macro_v > 0), so concentration should decrease
    assert res["anode_c"].data[-1, -1] < 100.0
    # Cathode received negative flux, so concentration should increase
    assert res["cathode_c"].data[-1, -1] > 200.0


# ==============================================================================
# 2. Category 1: Deep Nesting & Namespacing Integrity
# ==============================================================================

class Level3(fx.PDE):
    y = fx.State()
    def math(self): return {
        "equations": {self.y: fx.dt(self.y) == -self.y}, 
        "boundaries": {}, 
        "initial_conditions": {self.y: 1.0}
    }


class Level2(fx.PDE):
    sub = Level3()
    def math(self): return self.sub.math()

class Level1(fx.PDE):
    child = Level2()
    def math(self): return self.child.math()

def test_deep_nesting_recursive_namespaces():
    """Validates that `_apply_namespace` correctly stacks prefixes across deep hierarchies."""
    model = Level1()
    engine = Engine(model=model, target="cpu:serial", mock_execution=True)

    # The state 'y' in Level3 should be prefixed by both parent attribute names
    assert "child_sub_y" in engine.layout.state_offsets

    # The AST should reflect the properly mangled name.
    # Note: lhs is `dt(y)`, which is a UnaryOp. The State sits inside `child`.
    ast = model.ast()
    assert ast["global"][0]["lhs"]["child"]["name"] == "child_sub_y"


# ==============================================================================
# 3. Category 2: Merge Mechanics & Empty Payloads
# ==============================================================================

def test_merge_handles_empty_and_none_payloads():
    """Validates `fx.merge` resilience against empty dictionaries or None returns."""
    sys_a = {"global": [{"lhs": {"type": "State"}, "rhs": {"type": "Scalar"}}]}
    sys_b = {}
    sys_c = None
    sys_d = {"regions": {"x": []}, "boundaries": []}
    
    merged = fx.merge(sys_a, sys_b, sys_c, sys_d) # type: ignore
    
    assert "global" in merged
    assert len(merged["global"]) == 1
    assert "regions" in merged
    assert "x" in merged["regions"]
    assert len(merged["boundaries"]) == 0


# ==============================================================================
# 4. Category 3: Memory Isolation Boundaries (The Shared Domain Problem)
# ==============================================================================

class SharedDomainConsumer(fx.PDE):
    c = fx.State()
    def __init__(self, shared_domain: fx.Domain):
        super().__init__()
        # Anti-Pattern: Storing an external domain as an instance attribute
        self.x = shared_domain
        self.c.domain = self.x
        
    def math(self):
        return {
            "equations": {
                self.c: fx.dt(self.c) == fx.grad(self.c)
            },
            "boundaries": {},
            "initial_conditions": {}
        }

class BadSharedDomainParent(fx.PDE):
    macro_x = fx.Domain(bounds=(0, 1), resolution=5)
    
    def __init__(self):
        super().__init__()
        # Instantiating submodels and passing the parent domain
        self.sub_a = SharedDomainConsumer(self.macro_x)
        self.sub_b = SharedDomainConsumer(self.macro_x)
        
    def math(self):
        return fx.merge(self.sub_a.math(), self.sub_b.math())

def test_shared_domain_deepcopy_isolation_boundary():
    """
    Failure Mode: Validates the strict deepcopy boundary behavior.
    If a submodel stores a parent's Domain as an attribute, `_bind_declarations`
    will deepcopy it and aggressively rename it, destroying the topological
    link to the parent.
    """
    model = BadSharedDomainParent()
    
    # Because `sub_a` and `sub_b` were processed by the parent's `_bind_declarations`,
    # their internal references to `macro_x` were copied and prefixed.
    assert model.sub_a.x.name == "sub_a_macro_x"
    assert model.sub_b.x.name == "sub_b_macro_x"
    
    # The parent's original domain remains untouched
    assert model.macro_x.name == "macro_x"
    
    # When generating the AST, the domains are split, breaking spatial coupling
    ast = model.ast()
    regions = ast.get("regions", {})
    
    # Proof of isolation: There are two separate regions instead of one unified macro_x.
    # The PDE.ast() method serializes domain keys into strings.
    assert "sub_a_macro_x" in regions.keys()
    assert "sub_b_macro_x" in regions.keys()
    assert "macro_x" not in regions.keys()

# ==============================================================================