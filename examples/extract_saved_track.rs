use altd_sim::random_track::{
    blocks, Connection, GeneratedTile, GeneratedTrack, RandomTrackConfig,
};
use serde_json::Value;

fn coords(text: &str) -> [i32; 2] {
    let values = text
        .strip_prefix("Vector2i(")
        .and_then(|value| value.strip_suffix(')'))
        .expect("Vector2i(x, y)")
        .split(',')
        .map(|value| value.trim().parse().expect("integer coordinate"))
        .collect::<Vec<_>>();
    [values[0], values[1]]
}

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
    let tiles = saved["Tiles"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(position, tile)| {
            let atlas = tile["Block"]["TileSetId"]["AtlasId"].as_i64().unwrap() as i32;
            let alternative = tile["Block"]["TileSetId"]["AlternativeId"]
                .as_i64()
                .unwrap() as i32;
            let block = blocks()
                .iter()
                .position(|candidate| {
                    candidate.atlas == atlas && candidate.alternative == alternative
                })
                .expect("saved tile exists in the extracted block catalog");
            GeneratedTile {
                position: coords(position),
                block,
            }
        })
        .collect();
    let connection = &saved["StartTileConnection"];
    let track = GeneratedTrack {
        name: args[2].clone(),
        config: RandomTrackConfig {
            length: 0,
            allow_double: true,
            start: None,
            start_direction: None,
            surfaces: None,
            distribution: 0,
        },
        start: coords(saved["StartTileCoords"].as_str().unwrap()),
        connection: Connection {
            side: connection["Side"].as_u64().unwrap() as usize,
            index: connection["Index"].as_i64().unwrap() as i32,
            kind: connection["Type"].as_u64().unwrap() as usize,
        },
        tiles,
    };
    let scene = track.to_scene(&template);
    std::fs::write(
        &args[4],
        serde_json::to_string_pretty(&scene).unwrap() + "\n",
    )
    .unwrap();
}
