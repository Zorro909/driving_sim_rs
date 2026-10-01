#!/bin/sh
# Stable source fingerprint for detecting mixed serial/threaded packages.
set -eu
cd "$(dirname "$0")/.."
{
    for file in Cargo.toml Cargo.lock .cargo/config.toml wasm/build.sh wasm/source-id.sh wasm/bindgen-version.sh wasm/thread-toolchain; do
        sha256sum "$file"
    done
    find src -type f | LC_ALL=C sort | while IFS= read -r file; do
        sha256sum "$file"
    done
} | sha256sum | cut -d ' ' -f 1
