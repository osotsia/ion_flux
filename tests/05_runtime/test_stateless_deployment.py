import pytest
import numpy as np
import shutil
import platform
import os
import sys
import ion_flux as fx

# Ensure models directory is in path for E2E tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

REQUIRES_COMPILER = pytest.mark.skipif(not _has_compiler(), reason="Requires native C++ toolchain.")

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

REQUIRES_RUNTIME = pytest.mark.skipif(
    not _has_compiler() or not RUST_FFI_AVAILABLE, 
    reason="Requires native C++ toolchain and compiled Rust backend."
)

import pytest
import asyncio
import os
import shutil
import platform
import numpy as np
import ion_flux as fx
from ion_flux.runtime.scheduler import MultiTenantScheduler
from ion_flux.protocols import Sequence, CC, CV, Rest
from ion_flux._core import solve_ida_native



class AdjointAndEISModel(fx.PDE):
    """Validates Analytical EIS (Frequency Domain) and Exact VJP Adjoints."""
    V = fx.State(domain=None, name="V")
    R = fx.Parameter(default=10.0, name="R")
    C = fx.Parameter(default=0.1, name="C")
    i_app = fx.Parameter(default=1.0, name="i_app")
    
    def math(self):
        return {
            "equations": {
                self.V: fx.dt(self.V) == (self.i_app - self.V / self.R) / self.C
            },
            "boundaries": {},
            "initial_conditions": {
                self.V: 0.0
            }
        }




@REQUIRES_RUNTIME
def test_stateless_binary_deployment(tmp_path):
    """Validates 0ms cold-start `.so` deployments bypassing AST reconstruction."""
    engine = fx.Engine(model=AdjointAndEISModel(), target="cpu", mock_execution=False)
    
    export_file = tmp_path / "model_prod.so"
    engine.export_binary(str(export_file))
    
    assert os.path.exists(str(export_file) + ".meta.json")
    
    # Instantiate instantly without Clang or AST parsing
    stateless_engine = fx.Engine.load(str(export_file), target="cpu:serial")
    
    assert stateless_engine.mock_execution is False
    assert stateless_engine.layout.n_states == engine.layout.n_states
    assert stateless_engine.layout.get_param_offset("R") == engine.layout.get_param_offset("R")
    
    # Solve directly via FFI
    res = stateless_engine.solve(t_span=(0, 1.0), parameters={"R": 5.0})
    assert res.status == "completed"
