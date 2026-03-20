fn main() {
    // macOS Apple Silicon Homebrew paths
    println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/homebrew/lib");
    
    // macOS Intel Homebrew paths
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
    
    // Linux standard paths
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

    // Dynamically link against the SUNDIALS shared libraries
    println!("cargo:rustc-link-lib=sundials_ida");
    println!("cargo:rustc-link-lib=sundials_nvecserial");
    println!("cargo:rustc-link-lib=sundials_sunmatrixdense");
    println!("cargo:rustc-link-lib=sundials_sunlinsoldense");
    println!("cargo:rustc-link-lib=sundials_core"); 
}
