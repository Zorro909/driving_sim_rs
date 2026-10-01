//! Compares training on a track converted from the game's saved `.track`
//! JSON (random_track::saved_track_scene) with training on a captured scene.
//! Usage: saved_track_parity TRACK_JSON CAPTURED_SCENE NETWORK MODEL X Y ROTATION
use altd_sim::session::{Session, SessionOptions, Spawn};
use serde_json::{json, Value};

fn load(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (saved, captured, network, model) = (load(&a[0]), load(&a[1]), load(&a[2]), load(&a[3]));
    let spawn = Spawn { position: [a[4].parse().unwrap(), a[5].parse().unwrap()], rotation: a[6].parse().unwrap() };
    let template = json!({"vehicle": captured["vehicle"], "physics": captured["physics"]});
    let mut converted = altd_sim::random_track::saved_track_scene(&saved, "parity", &template).unwrap();
    // PARITY_TILE_MAP=0,0 shows the comparison detects a misplaced TileMap.
    if let Ok(at) = std::env::var("PARITY_TILE_MAP") {
        let p: Vec<i32> = at.split(',').map(|v| v.parse().unwrap()).collect();
        altd_sim::random_track::place_tile_map(&mut converted, [p[0], p[1]]);
    }
    println!("converted reset {} {}", converted["reset_position"], converted["reset_rotation"]);
    let options = |spawn: Option<Spawn>| {
        let mut o = SessionOptions::from_json(r#"{"population":32,"seed":5,"batchCount":2,"eliminateOnWall":false,"eliminateWhenIdle":false}"#).unwrap();
        o.spawn = spawn;
        o
    };
    for (label, converted_spawn) in [("captured spawn", Some(spawn)), ("scene reset pose", None)] {
        let mut x = Session::new(&captured, &network, &model, options(Some(spawn))).unwrap();
        let mut y = Session::new(&converted, &network, &model, options(converted_spawn)).unwrap();
        let shape = [network["inputs"].as_array().unwrap().len(), 16, 5];
        if network.get("weights").is_some() {
            x.start().unwrap();
            y.start().unwrap();
        } else {
            x.start_with_shape(&shape).unwrap();
            y.start_with_shape(&shape).unwrap();
        }
        let mut first = None;
        for generation in 0..4 {
            for window in 0..30 {
                x.advance(60, false).unwrap();
                y.advance(60, false).unwrap();
                if first.is_none() && x.car_states() != y.car_states() {
                    first = Some((generation, window * 60 + 60));
                }
            }
            let (sx, sy) = (x.generation_summary(), y.generation_summary());
            println!("{label}: generation {generation} best score {:.6} vs {:.6}, lapped {} vs {}, lap {:?} vs {:?}", sx.best_score, sy.best_score, sx.lapped, sy.lapped, sx.lap_time, sy.lap_time);
            x.next_generation().unwrap();
            y.next_generation().unwrap();
        }
        match first {
            None => println!("{label}: bit-identical over 4 generations of 1800 ticks"),
            Some((g, t)) => println!("{label}: first difference in generation {g} by tick {t}"),
        }
    }
}
