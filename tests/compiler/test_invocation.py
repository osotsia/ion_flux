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
            fx.dt(self.y0): self.p_alpha * (self.y1 - self.y0),
            self.y1: self.y1 - fx.abs(self.y0) * (self.y0 ** 2) + self.p_beta
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
def test_native_jacobian_execution():
    model = CoupledDynamics()
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    y = [-2.0, 5.0]
    ydot = [0.0, 0.0]
    params = {"p_alpha": 2.0, "p_beta": 1.0}
    c_j = 0.5
    
    jac = engine.evaluate_jacobian(y, ydot, c_j, parameters=params)
    
    # Analytical Verification of LHS - RHS mapping:
    # F0 = ydot[0] - [p_alpha * (y1 - y0)]
    # F1 = y1 - [y1 - |y0| * y0^2 + p_beta] = |y0| * y0^2 - p_beta
    # dF0/dy0 = p_alpha = 2.0; dF0/dydot0 = 1.0 => J00 = 2.0 + 0.5(1.0) = 2.5
    # dF0/dy1 = -p_alpha = -2.0; dF0/dydot1 = 0.0 => J01 = -2.0
    # dF1/dy0 (for y0 < 0, F1 = -y0^3 - p_beta) = -3y0^2 = -3(4) = -12.0 => J10 = -12.0
    # dF1/dy1 = 0.0 => J11 = 0.0
    
    assert len(jac) == 2
    assert len(jac[0]) == 2
    
    assert jac[0][0] == pytest.approx(2.5, rel=1e-5)
    assert jac[0][1] == pytest.approx(-2.0, rel=1e-5)
    assert jac[1][0] == pytest.approx(-12.0, rel=1e-5)
    assert jac[1][1] == pytest.approx(0.0, rel=1e-5)

@pytest.mark.skipif(not _has_compiler(), reason="No C++ toolchain found on the host machine.")
def test_native_compiler_caching():
    model = CoupledDynamics()
    engine_1 = Engine(model=model, target="cpu", mock_execution=False)
    
    import os
    import hashlib
    
    source_hash = hashlib.sha256(engine_1.cpp_source.encode('utf-8')).hexdigest()[:16]
    ext = ".dylib" if platform.system() == "darwin" else ".so"
    source_path = engine_1.runtime.lib_path.replace(f"lib_res_{source_hash}{ext}", f"res_{source_hash}.cpp")
    
    if os.path.exists(source_path):
        os.remove(source_path)
        
    engine_2 = Engine(model=model, target="cpu", mock_execution=False)
    assert engine_2.runtime.lib_path == engine_1.runtime.lib_path
