//! JSON reports and atomic run-file writes.

use serde_json::Value;
use std::path::Path;

pub(super) fn load(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub(super) fn write_report(path: &Path, value: &Value) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create report directory");
        }
    }
    std::fs::write(path, serde_json::to_string_pretty(value).unwrap() + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// Print a report without its bulky per-row key.
pub(super) fn print_without(value: &Value, skip: &str) {
    let mut map = value.as_object().unwrap().clone();
    map.shift_remove(skip);
    println!("{}", serde_json::to_string_pretty(&Value::Object(map)).unwrap());
}

pub(super) fn write_atomic(path: &Path, bytes: &[u8]) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).unwrap_or_else(|e| panic!("{}: {e}", tmp.display()));
    std::fs::rename(&tmp, path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

pub(super) fn write_atomic_report(path: &Path, value: &Value) {
    write_atomic(path, (serde_json::to_string_pretty(value).unwrap() + "\n").as_bytes());
}

pub(super) fn load_optional(path: &Path) -> Option<Value> {
    path.exists().then(|| load(path))
}
