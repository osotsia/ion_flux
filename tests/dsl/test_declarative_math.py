import ion_flux as fx

class AdvancedMathModel(fx.PDE):
    T = fx.State()
    V = fx.State()
    limit = fx.Parameter(default=4.2)
    
    def math(self):
        reaction_rate = fx.exp(-1.0 / self.T)
        clamped_voltage = fx.max(self.V, self.limit)
        trigger = self.V >= self.limit  
        
        return {
            "global": [
                fx.dt(self.T) == reaction_rate * fx.sin(self.T),
                self.V == clamped_voltage - self.limit
            ]
        }

def test_pde_ast_capture(heat_model):
    ast = heat_model.ast()
    
    # Check that 4 equations were captured across all buckets (1 PDE, 2 BCs, 1 IC)
    total_eqs = len(ast.get("global", [])) + len(ast.get("boundaries", [])) + sum(len(eqs) for eqs in ast.get("regions", {}).values())
    assert total_eqs == 4
    
    # Find the time derivative equation (routed to the regions bucket under 'rod')
    pde_eq = next(eq for eq in ast["regions"]["rod"] if eq["lhs"].get("op") == "dt")
    assert pde_eq["lhs"]["child"]["name"] == "T"

def test_advanced_math_operators():
    model = AdvancedMathModel()
    ast = model.ast()
    
    # Check trigger logic overloading (>= translates to "ge" binary operator)
    dt_eq = next(eq for eq in ast["global"] if eq["lhs"].get("op") == "dt")
    rhs = dt_eq["rhs"]
    assert rhs["type"] == "BinaryOp" and rhs["op"] == "mul"
    assert rhs["left"]["op"] == "exp"
    assert rhs["right"]["op"] == "sin"

    # The Voltage constraint is algebraic, so it routes to global
    v_eq = next(eq for eq in ast["global"] if eq["lhs"].get("name") == "V")
    assert v_eq["rhs"]["left"]["op"] == "max"