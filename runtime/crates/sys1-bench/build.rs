//! sys1-bench and sys1-probe link laya-mlx. With a prebuilt MLX (`MLX_SYS_PREBUILT_DIR`, see
//! vendor/mlx-sys/build.rs), libmlx.dylib is loaded through @rpath, and cargo passes a
//! dependency's `rustc-link-arg` only to that dependency's own targets, so this binary crate
//! emits the rpath itself (crates/laya-mlx/build.rs says more). The directory is canonicalized
//! first so that a relative `MLX_SYS_PREBUILT_DIR` does not become a relative rpath.
fn main() {
    println!("cargo:rerun-if-env-changed=MLX_SYS_PREBUILT_DIR");
    if let Some(dir) = std::env::var_os("MLX_SYS_PREBUILT_DIR") {
        let dir = std::path::Path::new(&dir).canonicalize().unwrap_or_else(|e| {
            panic!("MLX_SYS_PREBUILT_DIR={} cannot be resolved to an absolute path ({e})", dir.to_string_lossy())
        });
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.join("lib").display());
    }
}
