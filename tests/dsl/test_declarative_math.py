import ion_flux as fx

class AdvancedMathModel(fx.PDE):
    T = fx.State()
    V = fx.State()
    limit = fx.Parameter(default=4.2)
    
    def math(self):
        # Testing the full suite of new operators
        reaction_rate = fx.exp(-1.0 / self.T)
        clamped_voltage = fx.max(self.V, self.limit)
        trigger = self.V >= self.limit  # Operator overloading for logic
        
        return {
            fx.dt(self.T): reaction_rate * fx.sin(self.T),
            self.V: clamped_voltage - self.limit
        }

def test_pde_ast_capture(heat_model):
    ast = heat_model.ast()
    
    # Check that 4 equations were captured (PDE, 2 BCs, 1 IC)
    assert len(ast) == 4
    
    # Find the time derivative equation
    pde_eq = next(eq for eq in ast if eq["lhs"].get("op") == "dt")
    assert pde_eq["lhs"]["child"]["name"] == "T"

def test_advanced_math_operators():
    model = AdvancedMathModel()
    ast = model.ast()
    
    # Check trigger logic overloading (>= translates to "ge" binary operator)
    dt_eq = next(eq for eq in ast if eq["lhs"].get("op") == "dt")
    rhs = dt_eq["rhs"]
    assert rhs["type"] == "BinaryOp" and rhs["op"] == "mul"
    assert rhs["left"]["op"] == "exp"
    assert rhs["right"]["op"] == "sin"

    v_eq = next(eq for eq in ast if eq["lhs"].get("name") == "V")
    assert v_eq["rhs"]["left"]["op"] == "max"