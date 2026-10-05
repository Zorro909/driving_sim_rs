//! JSON reports and atomic run-file writes.

use serde_json::Value;
use std::path::Path;

pub(super) fn load(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A default input compiled into the binary, so packaged CLIs work outside the source checkout.
pub(super) struct Builtin {
    /// Repository path of the file, recorded in run.json as `builtin:<path>`.
    path: &'static str,
    text: &'static str,
}

pub(super) const RALLY_NETWORK: Builtin = Builtin {
    path: "assets/networks/rally.json",
    text: include_str!("../../../assets/networks/rally.json"),
};
pub(super) const RALLY_MODEL: Builtin = Builtin {
    path: "assets/models/rally.json",
    text: include_str!("../../../assets/models/rally.json"),
};

/// Loads `path`, or `builtin` when the option was omitted.
pub(super) fn load_or(path: Option<&Path>, builtin: &Builtin) -> Value {
    match path {
        Some(path) => load(path),
        None => serde_json::from_str(builtin.text).expect("built-in input"),
    }
}

/// The run.json record of an input loaded by [`load_or`].
pub(super) fn describe_input(path: Option<&Path>, builtin: &Builtin) -> Value {
    match path {
        Some(path) => serde_json::json!(path),
        None => format!("builtin:{}", builtin.path).into(),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_inputs_match_the_repository_assets() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for builtin in [&RALLY_NETWORK, &RALLY_MODEL] {
            assert_eq!(load_or(None, builtin), load(&root.join(builtin.path)));
            assert_eq!(describe_input(None, builtin), format!("builtin:{}", builtin.path));
        }
        let explicit = root.join(RALLY_MODEL.path);
        assert_eq!(
            describe_input(Some(&explicit), &RALLY_MODEL),
            serde_json::json!(explicit)
        );
    }
}
