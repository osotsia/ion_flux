import os
import sys
import glob
import hashlib
import subprocess
import shutil
import tempfile
import uuid
import logging
from ion_flux.compiler._5_toolchain.ffi_runtime import NativeRuntime

logger = logging.getLogger(__name__)

class NativeCompiler:
    """Manages the Clang/LLVM toolchain invocation and caching of emitted C++ strings."""
    def __init__(self, cache_dir: str = None):
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = tempfile.mkdtemp(prefix="ion_flux_jit_")
            
        self.bundled_toolchain_dir = os.path.expanduser("~/.cache/ion_flux/toolchain")
        
        self.compiler_cmd = self._find_bundled_compiler()
        self.enzyme_plugin = self._find_bundled_plugin()
        
        if not self.compiler_cmd:
            raise RuntimeError(
                "Hermetic C++ toolchain not found. Native execution requires the bundled LLVM/Enzyme toolchain. "
                "Execute `ion-flux install-toolchain` in your terminal to install it."
            )
        if not self.enzyme_plugin:
            raise RuntimeError(
                "Enzyme AD plugin not found. Exact analytical Jacobians require the bundled plugin. "
                "Execute `ion-flux install-toolchain` in your terminal to install it."
            )

    def _find_bundled_compiler(self) -> str:
        bundled_clang = os.path.join(self.bundled_toolchain_dir, "bin", "clang++")
        if os.path.exists(bundled_clang) and os.access(bundled_clang, os.X_OK):
            return bundled_clang
        return ""

    def _find_system_compiler(self) -> str:
        if sys.platform == "darwin":
            for path in ["/opt/homebrew/opt/llvm/bin/clang++", "/usr/local/opt/llvm/bin/clang++"]:
                if os.path.exists(path): return path
        return shutil.which("clang++") or shutil.which("g++") or ""

    def _find_bundled_plugin(self) -> str:
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        bundled_matches = glob.glob(os.path.join(self.bundled_toolchain_dir, "lib", f"ClangEnzyme*{ext}"))
        if bundled_matches:
            return bundled_matches[0]
        return ""

    def _find_system_plugin(self) -> str:
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        if sys.platform == "darwin":
            for base in ["/opt/homebrew/lib", "/usr/local/lib"]:
                matches = glob.glob(os.path.join(base, f"ClangEnzyme*{ext}"))
                if matches: return matches[0]
        elif sys.platform == "linux":
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            if conda_prefix:
                matches = glob.glob(os.path.join(conda_prefix, "lib", f"ClangEnzyme*{ext}"))
                if matches: return matches[0]
            for base in ["/usr/lib", "/usr/local/lib"]:
                matches = glob.glob(os.path.join(base, f"ClangEnzyme*{ext}"))
                if matches: return matches[0]
        return ""

    def compile(self, cpp_source: str, n_states: int) -> NativeRuntime:
        if not self.compiler_cmd:
            raise RuntimeError("C++ toolchain is unavailable on this host.")

        source_hash = hashlib.sha256(cpp_source.encode('utf-8')).hexdigest()[:16]
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib_name = f"lib_res_{source_hash}{ext}"
        lib_path = os.path.join(self.cache_dir, lib_name)
        
        if os.path.exists(lib_path):
            return NativeRuntime(lib_path, n_states)
            
        tmp_uuid = uuid.uuid4().hex
        source_path = os.path.join(self.cache_dir, f"res_{source_hash}_{tmp_uuid}.cpp")
        tmp_lib_path = os.path.join(self.cache_dir, f"lib_{source_hash}_{tmp_uuid}{ext}")

        with open(source_path, "w") as f:
            f.write(cpp_source)
            
        def attempt_compile(compiler: str, plugin: str) -> bool:
            cmd = [compiler, "-O3", "-fPIC", "-shared", "-o", tmp_lib_path, source_path]
            
            if "#pragma omp" in cpp_source:
                cmd.append("-fopenmp")
                if sys.platform == "darwin":
                    cmd.extend([
                        "-lomp", 
                        # M-Series Macs (Keg-only paths)
                        "-I/opt/homebrew/opt/libomp/include", 
                        "-L/opt/homebrew/opt/libomp/lib",
                        "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
                        # Intel Macs (Keg-only paths)
                        "-I/usr/local/opt/libomp/include", 
                        "-L/usr/local/opt/libomp/lib",
                        "-Wl,-rpath,/usr/local/opt/libomp/lib"
                    ])
                elif sys.platform == "linux":
                    cmd.extend(["-static-libgcc", "-static-libstdc++", "-Wl,-Bstatic", "-lgomp", "-Wl,-Bdynamic"])
            
            cmd.insert(1, f"-fplugin={plugin}")
            cmd.insert(2, "-DENZYME_ACTIVE")
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                os.replace(tmp_lib_path, lib_path)
                return True
            except subprocess.CalledProcessError as e:
                self.last_error = f"cmd: {' '.join(cmd)}\nstderr:\n{e.stderr}"
                return False

        success = attempt_compile(self.compiler_cmd, self.enzyme_plugin)

        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(tmp_lib_path):
            os.remove(tmp_lib_path)

        if not success:
            raise RuntimeError(f"Hermetic compilation failed.\n{getattr(self, 'last_error', 'Unknown error')}")
            
        return NativeRuntime(lib_path, n_states)