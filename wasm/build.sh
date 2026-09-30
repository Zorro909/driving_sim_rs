#!/bin/sh
# Builds the WebAssembly library into wasm/pkg: altd_sim.js (ES module glue)
# and altd_sim_bg.wasm. Needs the wasm32-unknown-unknown target and the
# wasm-bindgen CLI of the version in Cargo.lock:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version "$(wasm/bindgen-version.sh)"
# PROFILE=dev builds without optimizations; TARGET=nodejs|bundler|deno|web
# selects the wasm-bindgen output flavour (default web).
set -eu
cd "$(dirname "$0")/.."
PROFILE=${PROFILE:-release}
TARGET=${TARGET:-web}
OUT=${OUT:-wasm/pkg}
DIR=$PROFILE
[ "$PROFILE" = dev ] && DIR=debug
cargo rustc --profile "$PROFILE" --lib --crate-type cdylib --target wasm32-unknown-unknown --no-default-features --features wasm
wasm-bindgen --target "$TARGET" --out-dir "$OUT" --out-name altd_sim "target/wasm32-unknown-unknown/$DIR/altd_sim.wasm"
if command -v wasm-opt >/dev/null 2>&1 && [ "$PROFILE" = release ]; then
    wasm-opt -O3 -o "$OUT/altd_sim_bg.wasm" "$OUT/altd_sim_bg.wasm"
fi
ls -l "$OUT"
