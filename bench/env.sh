# Source before any benchmark or runtime build work:  source bench/env.sh
# Points every tool cache into bench/ (git-ignored), so one folder holds everything the benchmark creates.
# Set BENCH_ROOT first to reuse the caches of another checkout.
if [ -n "${BASH_SOURCE:-}" ]; then _env_file="${BASH_SOURCE[0]}"; else _env_file="$0"; fi
export BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "$_env_file")" && pwd)}"
unset _env_file
export HF_HOME=$BENCH_ROOT/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export UV_CACHE_DIR=$BENCH_ROOT/.cache/uv
export UV_PYTHON_INSTALL_DIR=$BENCH_ROOT/.cache/uv-python
export UV_TOOL_DIR=$BENCH_ROOT/.cache/uv-tools
export UV_TOOL_BIN_DIR=$BENCH_ROOT/bin
export PIP_CACHE_DIR=$BENCH_ROOT/.cache/pip
export XDG_CACHE_HOME=$BENCH_ROOT/.cache/xdg
export TORCHINDUCTOR_CACHE_DIR=$BENCH_ROOT/.cache/torchinductor
export npm_config_cache=$BENCH_ROOT/.cache/npm
export GOPATH=$BENCH_ROOT/.cache/go
export GOMODCACHE=$BENCH_ROOT/.cache/go/mod
# Cargo registry, git checkouts and `cargo install` output. Rustup toolchains stay in ~/.rustup.
export CARGO_HOME=$BENCH_ROOT/.cargo
export PATH=$BENCH_ROOT/bin:$CARGO_HOME/bin:$HOME/.cargo/bin:$PATH
# SwiftPM: always pass these flags, e.g. swift build $SWIFTPM_FLAGS
export SWIFTPM_FLAGS="--cache-path $BENCH_ROOT/.cache/swiftpm --config-path $BENCH_ROOT/.cache/swiftpm-config --security-path $BENCH_ROOT/.cache/swiftpm-security"
# xcodebuild: always pass -derivedDataPath $XCODE_DERIVED
export XCODE_DERIVED=$BENCH_ROOT/.cache/DerivedData
export CARGO_TARGET_DIR=$BENCH_ROOT/.cache/cargo-target
export TMPDIR=$BENCH_ROOT/tmp
export MLX_RS_METAL_PATH=$BENCH_ROOT/.cache/mlx-metal
# Prebuilt MLX from the mlx Python wheel (same MLX 0.32.2 as the laya-mlx control); see runtime/vendor/mlx-sys.
export MLX_SYS_PREBUILT_DIR=$BENCH_ROOT/contenders/laya-mlx/.venv/lib/python3.12/site-packages/mlx
