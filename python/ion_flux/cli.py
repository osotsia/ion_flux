import argparse
import urllib.request
import subprocess
import os
import sys
import platform
import shutil
import stat
import re
import time
import threading
from typing import Tuple

# =============================================================================
# Terminal Utilities
# =============================================================================

class ProgressBar:
    """
    Terminal progress bar styled consistently with the native Rust orchestrator.
    
    Features a Background Daemon Thread to animate a bouncing block while 
    waiting for blocking I/O (like tar extraction or CMake config), ensuring 
    the UI never appears frozen from the user's perspective.
    """
    def __init__(self, name: str):
        self.name = name[:4].strip() # Constrain to 4 chars for visual alignment
        self._last_draw = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def update(self, pct: float, suffix: str = "", force: bool = False) -> None:
        """Draws a deterministic fractional progress bar."""
        now = time.time()
        # Throttle redraws to ~20 FPS to prevent terminal flickering during rapid I/O
        if not force and now - self._last_draw < 0.05:
            return
        self._last_draw = now
        
        pct = max(0.0, min(1.0, pct))
        filled = int(pct * 30)
        bar = "█" * filled + "-" * (30 - filled)
        
        # Suffix is padded to clear trailing characters if the text length shrinks
        sys.stdout.write(f"\r▶ {self.name:<4} [{bar}] {pct*100:5.1f}% | {suffix:<45}")
        sys.stdout.flush()

    def start_indeterminate(self, suffix: str) -> None:
        """Spawns a daemon thread to animate a bouncing block during blocking I/O."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, args=(suffix,))
        self._thread.daemon = True  # Ensures thread dies instantly on Ctrl+C
        self._thread.start()

    def _animate(self, suffix: str) -> None:
        ticks = 0
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        
        while not self._stop_event.is_set():
            # Bouncing block math (Bar width 30, Block width 3)
            cycle = 54 
            pos = ticks % cycle
            if pos >= 27:
                pos = 54 - pos
            
            bar_list = ["-"] * 30
            for i in range(3):
                if 0 <= pos + i < 30:
                    bar_list[pos + i] = "█"
            bar = "".join(bar_list)
            
            spinner = spinner_chars[ticks % 10]
            sys.stdout.write(f"\r▶ {self.name:<4} [{bar}]  ---.-% | {spinner} {suffix:<43}")
            sys.stdout.flush()
            
            time.sleep(0.05)
            ticks += 1

    def stop_indeterminate(self, suffix: str = "Complete") -> None:
        """Joins the animation thread, returning control to the main thread."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
        self.finish(suffix)

    def finish(self, suffix: str = "Complete") -> None:
        """Locks the bar at 100% and advances to the next line."""
        sys.stdout.write(f"\r▶ {self.name:<4} [{'█'*30}] 100.0% | {suffix:<45}\n")
        sys.stdout.flush()


# =============================================================================
# Toolchain Installer
# =============================================================================

class ToolchainInstaller:
    """
    Orchestrates the hermetic installation of the LLVM compiler and Enzyme AD plugin.
    
    Why: Relying on system-level C++ compilers is notoriously fragile across different 
    OS package managers. By downloading a pre-compiled LLVM binary and compiling Enzyme 
    from source strictly against it, we guarantee a closed, mathematically exact 
    differentiable environment regardless of the host machine.
    """
    
    def __init__(self, llvm_version: str = "19.1.0", enzyme_version: str = "v0.0.256"):
        self.llvm_version = llvm_version
        self.enzyme_version = enzyme_version
        
        self.target_dir = os.path.expanduser("~/.cache/ion_flux/toolchain")
        self.llvm_dir = os.path.join(self.target_dir, "llvm")
        self.enzyme_src_dir = os.path.join(self.target_dir, "enzyme_src")
        
        self.llvm_tarball = os.path.join(self.target_dir, "llvm.tar.xz")
        self.enzyme_tarball = os.path.join(self.target_dir, "enzyme.tar.gz")

    def install(self) -> None:
        """The primary execution flow for the installation process."""
        if self._is_already_installed():
            print(f"Toolchain already satisfied at {self.target_dir}")
            return

        self._check_system_dependencies()
        self._prepare_directories()
        
        llvm_url, enzyme_url = self._resolve_download_urls()

        print("Fetching dependencies...")
        self._download_with_progress(llvm_url, self.llvm_tarball, "LLVM")
        self._extract_archive(self.llvm_tarball, self.llvm_dir, "Extr")
        
        self._download_with_progress(enzyme_url, self.enzyme_tarball, "Enzm")
        self._extract_archive(self.enzyme_tarball, self.enzyme_src_dir, "Extr")

        print("Building Automatic Differentiation plugin...")
        self._configure_cmake()
        self._compile_with_progress()

        print("Finalizing toolchain...")
        self._install_plugin_binaries()
        self._create_compiler_wrapper()
        self._prune_sysroot()
        self._scrub_symlinks()

        print(f"\n✅ Successfully installed hermetic C++ toolchain to {self.target_dir}")

    # --- Verification & Setup ---

    def _is_already_installed(self) -> bool:
        return os.path.exists(os.path.join(self.target_dir, "bin", "clang++"))

    def _check_system_dependencies(self) -> None:
        """Ensures the user has the bare minimum tools to bootstrap the compilation."""
        missing = [tool for tool in ["cmake", "ninja"] if shutil.which(tool) is None]
        if missing:
            print(f"Error: Missing required system dependencies to build the toolchain: {', '.join(missing)}")
            print("Please install them via your system package manager (e.g., `brew install cmake ninja` or `sudo apt install cmake ninja-build`).")
            sys.exit(1)

    def _prepare_directories(self) -> None:
        os.makedirs(os.path.join(self.target_dir, "bin"), exist_ok=True)
        os.makedirs(os.path.join(self.target_dir, "lib"), exist_ok=True)
        os.makedirs(self.llvm_dir, exist_ok=True)
        
        if os.path.exists(self.enzyme_src_dir):
            shutil.rmtree(self.enzyme_src_dir)
        os.makedirs(self.enzyme_src_dir, exist_ok=True)

    def _resolve_download_urls(self) -> Tuple[str, str]:
        """Maps the current hardware architecture to the correct LLVM release binaries."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == "darwin":
            arch = "ARM64" if machine == "arm64" else "X64"
            llvm_url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{self.llvm_version}/LLVM-{self.llvm_version}-macOS-{arch}.tar.xz"
        elif system == "linux":
            llvm_url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{self.llvm_version}/LLVM-{self.llvm_version}-Linux-X64.tar.xz"
        else:
            print(f"Unsupported system architecture: {system} {machine}")
            sys.exit(1)
            
        enzyme_url = f"https://github.com/EnzymeAD/Enzyme/archive/refs/tags/{self.enzyme_version}.tar.gz"
        return llvm_url, enzyme_url

    # --- I/O & Network ---

    def _download_with_progress(self, url: str, dest_path: str, bar_name: str) -> None:
        """Streams a download from a URL, updating the custom progress bar via a reporting hook."""
        pb = ProgressBar(bar_name)
        
        def reporthook(block_num: int, block_size: int, total_size: int):
            downloaded = block_num * block_size
            mb_downloaded = downloaded / 1048576
            
            if total_size > 0:
                pct = downloaded / total_size
                mb_total = total_size / 1048576
                # Force draw at 100% to guarantee it hits the end cleanly
                force = downloaded >= total_size
                pb.update(pct, f"{mb_downloaded:.1f}/{mb_total:.1f} MB", force=force)
            else:
                pb.update(0.0, f"{mb_downloaded:.1f} MB (Unknown Total)")
                
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ion_flux installer)'})
            with urllib.request.urlopen(req) as response:
                total_size = int(response.info().get("Content-Length", -1))
                
                with open(dest_path, 'wb') as out_file:
                    downloaded = 0
                    while True:
                        chunk = response.read(16384)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        
                        # Calculate fake block variables to feed the standard reporthook logic
                        reporthook(downloaded // 16384, 16384, total_size)
                        
            pb.finish("Complete")
        except Exception as e:
            print(f"\nError downloading {bar_name}: {e}")
            sys.exit(1)

    def _extract_archive(self, tarball_path: str, dest_dir: str, bar_name: str) -> None:
        """
        Extracts the archive. Spawns a daemon thread to animate the terminal 
        since `tar` can block the Python thread for upwards of 20 seconds for large releases.
        """
        pb = ProgressBar(bar_name)
        pb.start_indeterminate("Extracting archive...")
        
        # Use "-xf" for xz and "-xzf" for gzip based on file extension
        compression_flag = "-xzf" if tarball_path.endswith(".gz") else "-xf"
        
        try:
            subprocess.run(["tar", compression_flag, tarball_path, "-C", dest_dir, "--strip-components=1"], check=True)
            pb.stop_indeterminate("Extracted")
        except Exception as e:
            pb.stop_indeterminate("Failed")
            print(f"Error extracting {tarball_path}: {e}")
            sys.exit(1)
        finally:
            if os.path.exists(tarball_path):
                os.remove(tarball_path)

    # --- Compilation & Configuration ---

    def _configure_cmake(self) -> None:
        """Configures the Enzyme build system to securely link against our downloaded LLVM."""
        cc = os.path.join(self.llvm_dir, "bin", "clang")
        cxx = os.path.join(self.llvm_dir, "bin", "clang++")
        self.enzyme_build_dir = os.path.join(self.enzyme_src_dir, "build")
        
        pb = ProgressBar("Cmak")
        pb.start_indeterminate("Configuring build system...")
        
        cmake_args = [
            "cmake", "-G", "Ninja", "-S", os.path.join(self.enzyme_src_dir, "enzyme"), "-B", self.enzyme_build_dir,
            f"-DLLVM_DIR={os.path.join(self.llvm_dir, 'lib', 'cmake', 'llvm')}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}"
        ]
        
        # Linux requires explicit static linking to lld to prevent host glibc mismatches
        if platform.system().lower() == "linux":
            cmake_args.extend([
                f"-DCMAKE_LINKER={os.path.join(self.llvm_dir, 'bin', 'ld.lld')}",
                f"-DCMAKE_AR={os.path.join(self.llvm_dir, 'bin', 'llvm-ar')}",
                f"-DCMAKE_RANLIB={os.path.join(self.llvm_dir, 'bin', 'llvm-ranlib')}",
                "-DCMAKE_C_FLAGS=-fuse-ld=lld",
                "-DCMAKE_CXX_FLAGS=-fuse-ld=lld"
            ])
            
        try:
            subprocess.run(cmake_args, check=True, capture_output=True)
            pb.stop_indeterminate("Configured")
        except subprocess.CalledProcessError as e:
            pb.stop_indeterminate("Failed")
            print(f"CMake Configuration Failed:\n{e.stderr.decode()}")
            sys.exit(1)

    def _compile_with_progress(self) -> None:
        """
        Executes the Ninja build while intercepting stdout line-by-line to drive the progress bar.
        Extracts step fractions (e.g., [42/96]) and truncates the active task description.
        """
        pb = ProgressBar("Comp")
        ninja_pattern = re.compile(r"^\[\s*(\d+)/\s*(\d+)\]\s*(.*)")
        
        process = subprocess.Popen(
            ["ninja", "-C", self.enzyme_build_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1 # Request line buffering
        )
        
        for line in iter(process.stdout.readline, ''):
            match = ninja_pattern.search(line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                task = match.group(3).strip()
                
                # Truncate overly long C++ file paths to prevent text wrapping
                display_task = task[:38] + "..." if len(task) > 38 else task
                pct = current / total if total > 0 else 0.0
                
                # Force draw on the last item to guarantee 100% rendering before finish()
                force = (current == total)
                pb.update(pct, f"{current}/{total} | {display_task}", force=force)
                
        process.wait()
        
        if process.returncode != 0:
            print("\nEnzyme Compilation failed!")
            sys.exit(1)
            
        pb.finish("Compiled")

    # --- Finalization & Cleanup ---

    def _install_plugin_binaries(self) -> None:
        """Hunts down the compiled shared object and relocates it to our lib bin."""
        for root, dirs, files in os.walk(self.enzyme_build_dir):
            for file in files:
                if file.startswith("ClangEnzyme") and (file.endswith(".so") or file.endswith(".dylib")):
                    shutil.copy(os.path.join(root, file), os.path.join(self.target_dir, "lib", file))
        
        shutil.rmtree(self.enzyme_src_dir)

    def _create_compiler_wrapper(self) -> None:
        """
        Creates a `clang++` bash wrapper. 
        Why: On macOS, host SDK paths (like CoreFoundation or standard library headers) frequently 
        change locations across OS updates. This wrapper dynamically injects `--isysroot` via `xcrun`
        upon every invocation, ensuring the hermetic LLVM can always find the host's Apple headers.
        """
        wrapper_path = os.path.join(self.target_dir, "bin", "clang++")
        with open(wrapper_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write('DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\n')
            f.write('OS=$(uname -s)\n')
            f.write('if [ "$OS" = "Darwin" ]; then\n')
            f.write('  SDK_PATH=$(xcrun --show-sdk-path 2>/dev/null || echo "")\n')
            f.write('  if [ -n "$SDK_PATH" ]; then\n')
            f.write('    SYSROOT_FLAG="-isysroot $SDK_PATH"\n')
            f.write('  else\n')
            f.write('    SYSROOT_FLAG=""\n')
            f.write('  fi\n')
            f.write('  exec "$DIR/../llvm/bin/clang++" $SYSROOT_FLAG "$@"\n')
            f.write('else\n')
            f.write('  exec "$DIR/../llvm/bin/clang++" "$@"\n')
            f.write('fi\n')
            
        st = os.stat(wrapper_path)
        os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)

    def _prune_sysroot(self) -> None:
        """Aggressively drops the disk footprint of LLVM by deleting unrelated build tools/docs."""
        llvm_bin = os.path.join(self.llvm_dir, "bin")
        keep_bins = {"clang", "clang++", "clang-19", "ld.lld", "llvm-ar", "llvm-ranlib"}
        
        for f in os.listdir(llvm_bin):
            path = os.path.join(llvm_bin, f)
            if os.path.isfile(path) and not any(f.startswith(k) for k in keep_bins):
                os.remove(path)
                
        for d in ["share", "libexec", "docs"]:
            path = os.path.join(self.llvm_dir, d)
            if os.path.exists(path):
                shutil.rmtree(path)

    def _scrub_symlinks(self) -> None:
        """
        Cleans up dead symlinks inside the LLVM folder.
        Why: Python's setuptools (Maturin/Wheels) will violently crash with OS Error 2 
        if it encounters a broken symlink while attempting to bundle the package.
        """
        for root, dirs, files in os.walk(self.target_dir):
            for f in files + dirs:
                path = os.path.join(root, f)
                if os.path.islink(path) and not os.path.exists(path):
                    os.unlink(path)

# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    parser = argparse.ArgumentParser(prog="ion-flux", description="ion_flux orchestration utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    install_parser = subparsers.add_parser("install-toolchain", help="Fetch LLVM and compile Enzyme LLVM plugin from source")
    
    args = parser.parse_args()
    
    if args.command == "install-toolchain":
        installer = ToolchainInstaller()
        installer.install()

if __name__ == "__main__":
    main()