import argparse
import urllib.request
import subprocess
import os
import sys
import platform
import shutil
import stat

def get_toolchain_dir() -> str:
    """Resolve the standardized user-level cache directory for the toolchain."""
    return os.path.expanduser("~/.cache/ion_flux/toolchain")

def check_system_dependencies():
    """Ensure the user has the required build tools to compile Enzyme."""
    missing = []
    for tool in ["cmake", "ninja"]:
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print(f"Error: Missing required system dependencies to build the toolchain: {', '.join(missing)}")
        print("Please install them via your system package manager (e.g., `brew install cmake ninja` or `sudo apt install cmake ninja-build`).")
        sys.exit(1)

def install_toolchain(llvm_version: str = "19.1.0", enzyme_version: str = "v0.0.256"):
    """Fetches LLVM binaries and compiles Enzyme from source against them."""
    target_dir = get_toolchain_dir()
    
    if os.path.exists(os.path.join(target_dir, "bin", "clang++")):
        print(f"Toolchain already satisfied at {target_dir}")
        return

    check_system_dependencies()
    
    os.makedirs(os.path.join(target_dir, "bin"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "lib"), exist_ok=True)
    llvm_dir = os.path.join(target_dir, "llvm")
    os.makedirs(llvm_dir, exist_ok=True)
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "darwin":
        arch = "ARM64" if machine == "arm64" else "X64"
        llvm_url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{llvm_version}/LLVM-{llvm_version}-macOS-{arch}.tar.xz"
    elif system == "linux":
        llvm_url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{llvm_version}/LLVM-{llvm_version}-Linux-X64.tar.xz"
    else:
        print(f"Unsupported system architecture: {system} {machine}")
        sys.exit(1)

    tarball_path = os.path.join(target_dir, "llvm.tar.xz")
    
    print(f"Downloading LLVM {llvm_version} from {llvm_url}...")
    try:
        urllib.request.urlretrieve(llvm_url, tarball_path)
    except Exception as e:
        print(f"Error downloading LLVM: {e}")
        sys.exit(1)

    print("Extracting LLVM...")
    try:
        # Using subprocess tar is vastly faster and handles symlinks better than Python's native tarfile
        subprocess.run(["tar", "-xf", tarball_path, "-C", llvm_dir, "--strip-components=1"], check=True)
    except Exception as e:
        print(f"Error extracting LLVM: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(tarball_path):
            os.remove(tarball_path)

    print(f"Downloading Enzyme {enzyme_version}...")
    enzyme_src = os.path.join(target_dir, "enzyme_src")
    if os.path.exists(enzyme_src):
        shutil.rmtree(enzyme_src)
    os.makedirs(enzyme_src, exist_ok=True)

    enzyme_tarball = os.path.join(target_dir, "enzyme.tar.gz")
    enzyme_url = f"https://github.com/EnzymeAD/Enzyme/archive/refs/tags/{enzyme_version}.tar.gz"

    try:
        urllib.request.urlretrieve(enzyme_url, enzyme_tarball)
        subprocess.run(["tar", "-xzf", enzyme_tarball, "-C", enzyme_src, "--strip-components=1"], check=True)
    except Exception as e:
        print(f"Error downloading or extracting Enzyme: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(enzyme_tarball):
            os.remove(enzyme_tarball)

    print("Configuring Enzyme build with CMake...")
    cc = os.path.join(llvm_dir, "bin", "clang")
    cxx = os.path.join(llvm_dir, "bin", "clang++")
    enzyme_build = os.path.join(enzyme_src, "build")
    
    cmake_args = [
        "cmake", "-G", "Ninja", "-S", os.path.join(enzyme_src, "enzyme"), "-B", enzyme_build,
        f"-DLLVM_DIR={os.path.join(llvm_dir, 'lib', 'cmake', 'llvm')}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={cc}",
        f"-DCMAKE_CXX_COMPILER={cxx}"
    ]
    
    if system == "linux":
        cmake_args.extend([
            f"-DCMAKE_LINKER={os.path.join(llvm_dir, 'bin', 'ld.lld')}",
            f"-DCMAKE_AR={os.path.join(llvm_dir, 'bin', 'llvm-ar')}",
            f"-DCMAKE_RANLIB={os.path.join(llvm_dir, 'bin', 'llvm-ranlib')}",
            "-DCMAKE_C_FLAGS=-fuse-ld=lld",
            "-DCMAKE_CXX_FLAGS=-fuse-ld=lld"
        ])
        
    subprocess.run(cmake_args, check=True)

    print("Compiling Enzyme (This may take a few minutes)...")
    subprocess.run(["ninja", "-C", enzyme_build], check=True)

    print("Installing Enzyme plugin...")
    for root, dirs, files in os.walk(enzyme_build):
        for file in files:
            if file.startswith("ClangEnzyme") and (file.endswith(".so") or file.endswith(".dylib")):
                shutil.copy(os.path.join(root, file), os.path.join(target_dir, "lib", file))
    
    shutil.rmtree(enzyme_src)

    print("Writing clang++ compiler wrapper...")
    wrapper_path = os.path.join(target_dir, "bin", "clang++")
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

    print("Pruning LLVM sysroot to reduce disk footprint...")
    llvm_bin = os.path.join(llvm_dir, "bin")
    keep_bins = {"clang", "clang++", "clang-19", "ld.lld", "llvm-ar", "llvm-ranlib"}
    for f in os.listdir(llvm_bin):
        path = os.path.join(llvm_bin, f)
        if os.path.isfile(path) and not any(f.startswith(k) for k in keep_bins):
            os.remove(path)
            
    for d in ["share", "libexec", "docs"]:
        path = os.path.join(llvm_dir, d)
        if os.path.exists(path):
            shutil.rmtree(path)

    print("Scrubbing broken symlinks...")
    for root, dirs, files in os.walk(target_dir):
        for f in files + dirs:
            path = os.path.join(root, f)
            if os.path.islink(path) and not os.path.exists(path):
                os.unlink(path)

    print(f"\n✅ Successfully installed hermetic C++ toolchain to {target_dir}")

def main():
    parser = argparse.ArgumentParser(prog="ion-flux", description="ion_flux orchestration utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    install_parser = subparsers.add_parser("install-toolchain", help="Fetch LLVM and compile Enzyme LLVM plugin from source")
    
    args = parser.parse_args()
    
    if args.command == "install-toolchain":
        install_toolchain()

if __name__ == "__main__":
    main()