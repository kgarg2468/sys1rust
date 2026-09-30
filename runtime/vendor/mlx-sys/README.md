# mlx-sys

Rust bindings to the mlx-c API. Generated using bindgen.

The crate version is independent of its native source tuple. This revision targets mlx-c
`c74db5307cc8ce122f48d97ef951b30578674e7f`, whose CMake configuration pins MLX `v0.32.2`.

## Metal library location

Metal builds place `mlx.metallib` in `~/.mlx/lib/<mlx-c-key>/`, where `<mlx-c-key>` is the
first 12 characters of the pinned mlx-c commit. Packaged source without Git metadata uses a
deterministic hash of the mlx-c headers and CMake configuration instead. The stable location
allows binaries produced by `cargo install` to keep loading the library after Cargo removes its
temporary build directory.

Set `MLX_RS_METAL_PATH` to use a different directory verbatim as CMake's `MLX_METAL_PATH`.
When this override is set, the build does not read or write `HOME`, which supports sandboxed and
Nix builds.

## Prebuilt MLX (sys1rust)

Set `MLX_SYS_PREBUILT_DIR` to an MLX install such as the `mlx` Python wheel's `site-packages/mlx`
directory. The build then compiles only mlx-c and links it against `lib/libmlx.dylib` from that
directory. The install must be MLX `0.32.2`; the build reads
`share/cmake/MLX/MLXConfigVersion.cmake` and fails on any other version. The path is resolved to
an absolute path, and the build reruns when `lib/libmlx.dylib` or the version file changes.

The two locations above do not apply to prebuilt builds. `MLX_RS_METAL_PATH` is ignored, and
`libmlx.dylib` loads `mlx.metallib` from its own directory (`lib/`), so keep the two files
together. The final binary needs an rpath to that `lib/` directory.
