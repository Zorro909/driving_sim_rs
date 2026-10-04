//! Generated saved-track imports preserve training, resets and reproduction.
use altd_sim::{
    track::random_track::{blocks, place_tile_map, saved_track_scene, GeneratedTrack},
    training::session::{Session, SessionOptions, Spawn},
};
use serde_json::{json, Value};

#[path = "../examples/support/generated.rs"]
mod generated;

/// Encode generated tiles in the format accepted by the saved-track importer.
fn saved(track: &GeneratedTrack) -> Value {
    let tiles: serde_json::Map<String, Value> = track
        .tiles
        .iter()
        .map(|tile| {
            let key = format!("Vector2i({}, {})", tile.position[0], tile.position[1]);
            let block = &blocks()[tile.block];
            (
                key.clone(),
                json!({"Block": {"TileSetId": {
            "AtlasId": block.atlas, "AlternativeId": block.alternative}}, "Coords": key}),
            )
        })
        .collect();
    json!({"Tiles": tiles,
        "StartTileCoords": format!("Vector2i({}, {})", track.start[0], track.start[1]),
        "StartTileConnection": {"Side": track.connection.side,
            "Index": track.connection.index, "Type": track.connection.kind}})
}

fn options(spawn: Option<Spawn>) -> SessionOptions {
    let mut options = SessionOptions::from_json(
        r#"{"population":32,"seed":5,"batchCount":2,"eliminateOnWall":false,"eliminateWhenIdle":false}"#,
    )
    .unwrap();
    options.spawn = spawn;
    options
}

#[test]
fn saved_track_import_matches_generated_scene_for_four_generations() {
    let track = generated::track("rally", 1);
    let template = generated::template("rally");
    let network = generated::network("rally");
    let model = generated::model("rally");
    let mut direct = track.to_scene(&template);
    // A saved track places collision geometry at the game's TileMap origin.
    // The expected scene uses the same documented placement explicitly.
    place_tile_map(&mut direct, [3, 0]);
    direct["track"]["native_broadphase"] = json!(false);
    let saved_fixture: Value = serde_json::from_str(include_str!("fixtures/generated_saved_track.json")).unwrap();
    assert_eq!(saved_fixture, saved(&track), "fixed generated saved-track fixture");
    let converted = saved_track_scene(&saved_fixture, &track.name, &template).unwrap();
    assert_eq!(direct, converted, "saved import reconstructs every scene field");
    let pose = generated::spawn(&direct);
    let spawn = Spawn {
        position: serde_json::from_value(pose["position"].clone()).unwrap(),
        rotation: pose["rotation"].as_f64().unwrap(),
    };
    for converted_spawn in [Some(spawn), None] {
        let mut x = Session::new(&direct, &network, &model, options(Some(spawn))).unwrap();
        let mut y = Session::new(&converted, &network, &model, options(converted_spawn)).unwrap();
        let shape = [network["inputs"].as_array().unwrap().len(), 16, 5];
        x.start_with_shape(&shape).unwrap();
        y.start_with_shape(&shape).unwrap();
        for generation in 0..4 {
            for window in 0..30 {
                assert_eq!(x.advance(60, false).unwrap(), y.advance(60, false).unwrap());
                let bits = |session: &Session| session.car_states().into_iter().map(f64::to_bits).collect::<Vec<_>>();
                assert_eq!(bits(&x), bits(&y), "generation {generation}, window {window}");
            }
            assert_eq!(
                serde_json::to_value(x.generation_summary()).unwrap(),
                serde_json::to_value(y.generation_summary()).unwrap()
            );
            x.next_generation().unwrap();
            y.next_generation().unwrap();
            assert_eq!(
                x.checkpoint_bytes().unwrap(),
                y.checkpoint_bytes().unwrap(),
                "generation {generation}: reproduction and RNG"
            );
        }
    }
}

#[test]
fn saved_track_import_rejects_open_paths_and_unknown_tiles() {
    let track = generated::track("rally", 0);
    let template = generated::template("rally");
    let mut file = saved(&track);
    let key = format!(
        "Vector2i({}, {})",
        track.tiles[3].position[0], track.tiles[3].position[1]
    );
    file["Tiles"].as_object_mut().unwrap().shift_remove(&key);
    assert!(saved_track_scene(&file, "open", &template)
        .unwrap_err()
        .contains("leaves the track"));
    let mut file = saved(&track);
    file["Tiles"][&key]["Block"]["TileSetId"]["AtlasId"] = json!(999);
    assert!(saved_track_scene(&file, "unknown", &template)
        .unwrap_err()
        .contains("unknown block"));
}
