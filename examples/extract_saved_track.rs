use altd_sim::random_track::saved_track_scene;
use serde_json::Value;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(
        args.len(),
        5,
        "usage: extract_saved_track TRACK NAME TEMPLATE OUTPUT"
    );
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let template: Value =
        serde_json::from_str(&std::fs::read_to_string(&args[3]).unwrap()).unwrap();
    let scene = saved_track_scene(&saved, &args[2], &template).unwrap();
    std::fs::write(
        &args[4],
        serde_json::to_string_pretty(&scene).unwrap() + "\n",
    )
    .unwrap();
}
