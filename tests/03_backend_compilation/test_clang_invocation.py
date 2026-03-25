"""
Backend Compilation: Clang Invocation

Tests the subprocess invocation of the Clang compiler, generation of portable
.so binaries, and basic Foreign Function Interface (FFI) loading.
"""

import pytest
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine
from ion_flux.compiler.invocation import NativeCompiler

class CoupledDynamics(fx.PDE):
    y0 = fx.State()
    y1 = fx.State()
    p_alpha = fx.Parameter(default=1.5)
    p_beta = fx.Parameter(default=0.0)
    
    def math(self):
        return {
            "global": [
                fx.dt(self.y0) == self.p_alpha * (self.y1 - self.y0),
                self.y1 == self.y1 - fx.abs(self.y0) * (self.y0 ** 2) + self.p_beta
            ]
        }

def _has_compiler() -> bool:
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

def _has_enzyme() -> bool:
    return bool(NativeCompiler().enzyme_plugin)

@pytest.mark.skipif(not _has_compiler(), reason="No C++ toolchain found on the host machine.")
def test_native_compilation_and_execution():
    model = CoupledDynamics()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    y = [-2.0, 5.0]
    ydot = [0.0, 0.0]
    params = {"p_alpha": 2.0, "p_beta": 1.0}
    
    res = engine.evaluate_residual(y, ydot, parameters=params)
    assert len(res) == 2
    assert res[0] == pytest.approx(-14.0, rel=1e-5)
    assert res[1] == pytest.approx(7.0, rel=1e-5)

@pytest.mark.skipif(not _has_compiler() or not _has_enzyme(), reason="Enzyme plugin required for Analytical Jacobian.")
def test_native_dense_jacobian_execution():
    model = CoupledDynamics()
    engine = Engine(model=model, target="cpu", mock_execution=False, jacobian_bandwidth=0)
    
    y = [-2.0, 5.0]
    ydot = [0.0, 0.0]
    params = {"p_alpha": 2.0, "p_beta": 1.0}
    
    jac = engine.evaluate_jacobian(y, ydot, c_j=0.5, parameters=params)
    
    assert jac[0][0] == pytest.approx(2.5, rel=1e-5)
    assert jac[0][1] == pytest.approx(-2.0, rel=1e-5)
    assert jac[1][0] == pytest.approx(-12.0, rel=1e-5)
    assert jac[1][1] == pytest.approx(0.0, rel=1e-5)

@pytest.mark.skipif(not _has_compiler() or not _has_enzyme(), reason="Enzyme plugin required for Analytical Jacobian.")
def test_native_banded_jacobian_execution():
    """Validates the Curtis-Powell-Reid graph coloring algorithm logic generates correct results."""
    model = CoupledDynamics()
    # Explicitly use bandwidth=1 (Tridiagonal) to invoke CPR graph-coloring
    engine = Engine(model=model, target="cpu", mock_execution=False, jacobian_bandwidth=1)
    
    y = [-2.0, 5.0]
    ydot = [0.0, 0.0]
    params = {"p_alpha": 2.0, "p_beta": 1.0}
    
    jac = engine.evaluate_jacobian(y, ydot, c_j=0.5, parameters=params)
    
    # CPR output should perfectly match the Dense output for small N
    assert jac[0][0] == pytest.approx(2.5, rel=1e-5)
    assert jac[0][1] == pytest.approx(-2.0, rel=1e-5)
    assert jac[1][0] == pytest.approx(-12.0, rel=1e-5)
    assert jac[1][1] == pytest.approx(0.0, rel=1e-5)
