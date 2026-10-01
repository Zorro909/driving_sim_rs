#!/bin/sh
# THREADS=1 builds the shared-memory package with a scoped nightly toolchain.
# TEST_THREADS=1 enables test-only diagnostics.
# wasm-bindgen CLI must match Cargo.lock. See wasm/README.md.
set -eu
cd "$(dirname "$0")/.."
PROFILE=${PROFILE:-release}
TARGET=${TARGET:-web}
THREADS=${THREADS:-0}
TEST_THREADS=${TEST_THREADS:-0}
for value in "$THREADS" "$TEST_THREADS"; do
    case "$value" in 0|1) ;; *) echo 'THREADS and TEST_THREADS must be 0 or 1.' >&2; exit 1 ;; esac
done
SOURCE_ID=$(wasm/source-id.sh)
BINDGEN_VERSION=$(wasm/bindgen-version.sh)
if [ "$(wasm-bindgen --version)" != "wasm-bindgen $BINDGEN_VERSION" ]; then
    echo "Install the matching CLI: cargo install wasm-bindgen-cli --version $BINDGEN_VERSION --locked" >&2
    exit 1
fi
FEATURES=wasm
[ "$TEST_THREADS" = 1 ] && FEATURES=$FEATURES,wasm-thread-test
DIR=$PROFILE
[ "$PROFILE" = dev ] && DIR=debug
TOOLCHAIN=$(rustc --version)
MEMORY=null
if [ "$THREADS" = 1 ]; then
    if [ "$TARGET" != web ]; then
        echo 'Threaded WASM requires TARGET=web. Use THREADS=0 for nodejs, bundler or deno.' >&2
        exit 1
    fi
    OUT=${OUT:-wasm/pkg-threads}
    BUILD_DIR=${THREAD_TARGET_DIR:-target/wasm-threads}
    TOOLCHAIN=$(cat wasm/thread-toolchain)
    MEMORY=1073741824
    FEATURES=$FEATURES,wasm-threads
    # RUSTFLAGS overrides .cargo/config.toml, including the WebGPU cfg.
    RUSTFLAGS="--cfg=web_sys_unstable_apis -C target-feature=+atomics,+bulk-memory -C link-arg=--shared-memory -C link-arg=--max-memory=$MEMORY -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base" \
        cargo +"$TOOLCHAIN" rustc --locked -Z build-std=panic_abort,std \
        --target-dir "$BUILD_DIR" --profile "$PROFILE" --lib --crate-type cdylib \
        --target wasm32-unknown-unknown --no-default-features --features "$FEATURES"
else
    OUT=${OUT:-wasm/pkg}
    BUILD_DIR=${CARGO_TARGET_DIR:-target}
    cargo rustc --locked --target-dir "$BUILD_DIR" --profile "$PROFILE" --lib --crate-type cdylib \
        --target wasm32-unknown-unknown --no-default-features --features "$FEATURES"
fi
# Generate into a clean temporary directory. A failed build leaves the old
# package intact; removed dependencies cannot leave stale snippets behind.
STAGING=$(mktemp -d "${TMPDIR:-/tmp}/altd-wasm.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT HUP INT TERM
wasm-bindgen --target "$TARGET" --out-dir "$STAGING" --out-name altd_sim \
    "$BUILD_DIR/wasm32-unknown-unknown/$DIR/altd_sim.wasm"
OPTIMIZER=null
if command -v wasm-opt >/dev/null 2>&1 && [ "$PROFILE" = release ]; then
    OPTIMIZER="\"$(wasm-opt --version)\""
    if [ "$THREADS" = 1 ]; then
        wasm-opt -O3 --enable-threads --enable-bulk-memory -o "$STAGING/altd_sim_bg.wasm" "$STAGING/altd_sim_bg.wasm"
    else
        wasm-opt -O3 -o "$STAGING/altd_sim_bg.wasm" "$STAGING/altd_sim_bg.wasm"
    fi
fi
THREADED=false; [ "$THREADS" = 1 ] && THREADED=true
DIAGNOSTICS=false; [ "$TEST_THREADS" = 1 ] && DIAGNOSTICS=true
printf '{"source":"%s","threads":%s,"threadDiagnostics":%s,"toolchain":"%s","maxMemoryBytes":%s,"profile":"%s","target":"%s","bindgen":"%s","optimizer":%s}\n' \
    "$SOURCE_ID" "$THREADED" "$DIAGNOSTICS" "$TOOLCHAIN" "$MEMORY" "$PROFILE" "$TARGET" "$BINDGEN_VERSION" "$OPTIMIZER" > "$STAGING/features.json"
# OUT is replaced as a whole. Only replace an empty directory or an earlier
# package, never a source or parent directory given by mistake.
if [ -e "$OUT" ]; then
    if [ ! -d "$OUT" ] || { [ -n "$(ls -A "$OUT")" ] && [ ! -f "$OUT/altd_sim.js" ]; }; then
        echo "Refusing to replace $OUT: it is not an earlier altd_sim package." >&2
        exit 1
    fi
fi
mkdir -p "$(dirname "$OUT")"
rm -rf "$OUT"
mv "$STAGING" "$OUT"
cat "$OUT/features.json"
