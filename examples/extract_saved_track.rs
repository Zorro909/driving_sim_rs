//! Import an uncompressed saved-track JSON using a geometry-free vehicle template.
//! Usage: extract_saved_track TRACK NAME TEMPLATE OUTPUT
use altd_sim::track::random_track::saved_track_scene;
use serde_json::Value;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 5 || args[1] == "--help" {
        eprintln!("usage: extract_saved_track TRACK NAME TEMPLATE OUTPUT");
        if args.get(1).is_some_and(|a| a == "--help") {
            return;
        }
        std::process::exit(2);
    }
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let template: Value = serde_json::from_str(&std::fs::read_to_string(&args[3]).unwrap()).unwrap();
    let scene = saved_track_scene(
        &saved,
        &args[2],
        &template,
        altd_sim::math::profile::MathProfile::Proton,
    )
    .unwrap();
    std::fs::write(&args[4], serde_json::to_string_pretty(&scene).unwrap() + "\n").unwrap();
}
