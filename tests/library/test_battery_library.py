import pytest
import numpy as np
from ion_flux.battery import DFN, parameters
from ion_flux import Engine
import ion_flux.protocols as protocols

@pytest.fixture
def dfn_engine():
    # Compile the DFN once for all tests in this module
    model = DFN(options={"thermal": "lumped"})
    return Engine(model=model, target="cpu")

def test_dfn_flat_parameter_overrides(dfn_engine):
    base_params = parameters.Chen2020()
    
    # Test updating deeply nested parameters via flat string API
    overrides = {
        "neg_elec.porosity": 0.25,
        "electrolyte.initial_concentration": 1200.0
    }
    
    protocol = protocols.ConstantCurrent(c_rate=1.0, until_voltage=2.5)
    
    result = dfn_engine.solve(protocol=protocol, parameters={**base_params, **overrides})
    
    assert result.status == "completed"
    assert result["Voltage [V]"].data[-1] <= 2.501 # Honors the voltage cutoff event

def test_drive_cycle_protocol_injection(dfn_engine):
    base_params = parameters.Chen2020()
    
    # Mock a high-frequency 100-second drive cycle
    time_array = np.linspace(0, 100, 1000)
    current_array = np.sin(time_array) * 5.0 
    
    protocol = protocols.CurrentProfile(time=time_array, current=current_array)
    
    result = dfn_engine.solve(protocol=protocol, parameters=base_params)
    
    assert result.status == "completed"
    assert len(result["Time [s]"].data) >= 1000 # Solver stepped through the profile