// With a prebuilt MLX (vendor/mlx-sys), libmlx.dylib is loaded through @rpath.
fn main() {
    println!("cargo:rerun-if-env-changed=MLX_SYS_PREBUILT_DIR");
    if let Some(dir) = std::env::var_os("MLX_SYS_PREBUILT_DIR") {
        let lib = std::path::Path::new(&dir).join("lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}
