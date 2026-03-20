import pytest
import shutil
import platform
import ion_flux as fx
from ion_flux.runtime.engine import Engine


class CoupledDynamics(fx.PDE):
    """
    A non-linear testing system to validate C++ AST operations and parameter binding.
    y0_dot = p_alpha * (y1 - y0)
    0 = y1 - abs(y0) * y0^2 + p_beta
    """
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
    # Check standard paths or Homebrew macOS paths
    has_std = bool(shutil.which("clang++") or shutil.which("g++"))
    has_mac = platform.system() == "darwin" and (
        shutil.os.path.exists("/opt/homebrew/opt/llvm/bin/clang++") or 
        shutil.os.path.exists("/usr/local/opt/llvm/bin/clang++")
    )
    return has_std or has_mac

@pytest.mark.skipif(not _has_compiler(), reason="No C++ toolchain found on the host machine.")
def test_native_compilation_and_execution():
    model = CoupledDynamics()
    
    # Engine invokes NativeCompiler directly when mock_execution=False
    engine = Engine(model=model, target="cpu", mock_execution=False)
    
    # Test vectors
    y = [-2.0, 5.0]
    ydot = [0.0, 0.0]
    params = {"p_alpha": 2.0, "p_beta": 1.0}
    
    # Execute compiled C++ function natively via ctypes
    res = engine.evaluate_residual(y, ydot, parameters=params)
    
    # Expected analytical results:
    # ----------------------------
    # Equation 0 (ODE): res[0] = ydot[0] - [ p_alpha * (y1 - y0) ]
    # res[0] = 0.0 - [ 2.0 * (5.0 - (-2.0)) ]
    # res[0] = 0.0 - [ 2.0 * 7.0 ] = -14.0
    
    # Equation 1 (DAE): res[1] = y1 - [ y1 - abs(y0) * (y0^2) + p_beta ]
    # res[1] = 5.0 - [ 5.0 - abs(-2.0) * ((-2.0)^2) + 1.0 ]
    # res[1] = 5.0 - [ 5.0 - 2.0 * 4.0 + 1.0 ]
    # res[1] = 5.0 - [ 5.0 - 8.0 + 1.0 ] = 5.0 - [-2.0] = 7.0
    
    assert len(res) == 2
    assert res[0] == pytest.approx(-14.0, rel=1e-5)
    assert res[1] == pytest.approx(7.0, rel=1e-5)


@pytest.mark.skipif(not _has_compiler(), reason="No C++ toolchain found on the host machine.")
def test_native_compiler_caching():
    model = CoupledDynamics()
    engine_1 = Engine(model=model, target="cpu", mock_execution=False)
    
    # Compiling the exact same topology should bypass the Clang invocation and instantly load
    # By deleting the source file, we prove that Engine_2 loaded the .so from cache
    import os
    import hashlib
    
    source_hash = hashlib.sha256(engine_1.cpp_source.encode('utf-8')).hexdigest()[:16]
    ext = ".dylib" if platform.system() == "darwin" else ".so"
    source_path = engine_1.runtime.lib_path.replace(f"lib_res_{source_hash}{ext}", f"res_{source_hash}.cpp")
    
    if os.path.exists(source_path):
        os.remove(source_path)
        
    engine_2 = Engine(model=model, target="cpu", mock_execution=False)
    
    # Will throw an exception if the Clang sub-process attempted to run without the source file
    assert engine_2.runtime.lib_path == engine_1.runtime.lib_path
