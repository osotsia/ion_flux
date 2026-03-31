"""
E2E Integration: Battery Library API

Validates the public API wrappers around the core engine, including flat-string
parameter injection and dynamic drive cycle array evaluation.
"""

import pytest
import numpy as np
import ion_flux as fx
from ion_flux.battery import DFN, parameters
from ion_flux import Engine
from ion_flux.protocols import Sequence, CC

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
    
    protocol = Sequence([CC(rate=1.0, until=dfn_engine.model.V <= 2.5)])
    
    result = dfn_engine.solve(protocol=protocol, parameters={**base_params, **overrides})
    
    assert result.status == "completed"
    assert result["V"].data[-1] <= 2.501 # Honors the voltage cutoff event natively

def test_drive_cycle_protocol_injection(dfn_engine):
    base_params = parameters.Chen2020()
    
    # Mock a high-frequency 100-second drive cycle
    time_array = np.linspace(0, 100, 1000)
    
    # In V2, dynamic array inputs are evaluated efficiently via native t_eval injection
    result = dfn_engine.solve(t_eval=time_array, parameters=base_params)
    
    assert result.status == "completed"
    assert len(result["Time [s]"].data) == 1000 # Solver seamlessly swept the profile trajectory