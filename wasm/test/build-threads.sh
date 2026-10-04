#!/bin/sh
# Keep integration diagnostics out of distributable packages.
set -eu
cd "$(dirname "$0")/../.."
TEST_THREADS=1 THREADS=1 OUT=target/wasm-thread-test/threads wasm/build.sh
TEST_THREADS=1 THREADS=0 OUT=target/wasm-thread-test/single wasm/build.sh
cargo run --locked --release --example wasm_thread_reference
