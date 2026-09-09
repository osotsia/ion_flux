fn main() {
    // macOS Apple Silicon Homebrew paths
    println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
    
    // macOS Intel / Linux compiled source paths
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    
    // Linux standard paths
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

    // Statically link against the SUNDIALS archives
    println!("cargo:rustc-link-lib=static=sundials_ida");
    println!("cargo:rustc-link-lib=static=sundials_nvecserial");
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixdense");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsoldense");
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixband");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolband");
    println!("cargo:rustc-link-lib=static=sundials_core"); 
}