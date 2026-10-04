#!/bin/sh
# Prints the wasm-bindgen version of Cargo.lock; the CLI must match it exactly.
cd "$(dirname "$0")/.."
awk '/^name = "wasm-bindgen"$/ { getline; sub(/^version = "/, ""); sub(/"$/, ""); print; exit }' Cargo.lock
